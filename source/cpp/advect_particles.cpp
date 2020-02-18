// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Vertex.h>

#include "advect_particles.h"
#include "utils.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
advect_particles::advect_particles(particles& P, FunctionSpace& U,
                                   Function& uhi, const std::string type1)
    : _P(&P), uh(&uhi), _element(U.element())
{
  // Following types are distinguished:
  // "open"       --> open boundary
  // "periodic"   --> periodic bc (additional info on extent required)
  // "closed"     --> closed boundary

  // This constructor cant take periodic:
  assert(type1 != "periodic");

  // Set facet info
  update_facets_info();

  // Set all external facets to type1
  set_bfacets(type1);

  // Set some other useful info
  _space_dimension = _element->space_dimension();
  _value_size_loc = 1;
  for (std::size_t i = 0; i < _element->value_rank(); i++)
    _value_size_loc *= _element->value_dimension(i);
}
//-----------------------------------------------------------------------------
// Using delegate constructors here
advect_particles::advect_particles(
    particles& P, FunctionSpace& U, Function& uhi, const std::string type1,
    Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits)
    : advect_particles::advect_particles(P, U, uhi, type1)
{
  std::size_t gdim = _P->mesh()->geometry().dim();

  // Then the only thing to do: check if type1 was "periodic"
  if (type1 == "periodic")
  {
    // Check if it has the right size, always has to come in pairs
    // TODO: do provided values make sense?
    if ((pbc_limits.size() % (gdim * 4)) != 0)
    {
      dolfin_error("advect_particles.cpp::advect_particles",
                   "construct periodic boundary information",
                   "Incorrect shape of pbc_limits provided?");
    }

    std::size_t num_rows = pbc_limits.size() / (gdim * 2);
    for (std::size_t i = 0; i < num_rows; i++)
    {
      std::vector<double> pbc_helper(gdim * 2);
      for (std::size_t j = 0; j < gdim * 2; j++)
        pbc_helper[j] = pbc_limits[i * gdim * 2 + j];

      pbc_lims.push_back(pbc_helper);
    }
    pbc_active = true;
  }
  else
  {
    dolfin_error("advect_particles.cpp::advect_particles",
                 "could not set pbc_lims",
                 "Did you provide limits for a non-periodic BC?");
  }
}
//-----------------------------------------------------------------------------
advect_particles::advect_particles(particles& P, FunctionSpace& U, Function& uhi,
                 const MeshFunction<std::size_t>& mesh_func)
    : _P(&P), uh(&uhi), _element(U.element())
{
  // Confirm that mesh_func contains no periodic boundary values (3)
  if (std::find(mesh_func.values(), mesh_func.values()+mesh_func.size(), 3)
        != mesh_func.values()+mesh_func.size())
    dolfin_error("advect_particles.cpp::advect_particles",
                 "construct advect_particles class",
                 "Periodic boundary value encountered in facet MeshFunction");

  // Set facet info
  update_facets_info();

  // Set facets information
  set_bfacets(mesh_func);

  // Set some other useful info
  _space_dimension = _element->space_dimension();
  _value_size_loc = 1;
  for (std::size_t i = 0; i < _element->value_rank(); i++)
    _value_size_loc *= _element->value_dimension(i);
}
//-----------------------------------------------------------------------------
advect_particles::advect_particles(particles& P, FunctionSpace& U, Function& uhi,
                                   const MeshFunction<std::size_t>& mesh_func,
                                   Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits)
    : _P(&P), uh(&uhi), _element(U.element())
{
  // Confirm that mesh_func does contain periodic boundary values?

  // Set facet info
  update_facets_info();

  // Set facets information
  set_bfacets(mesh_func);

  // Set periodic boundary info
  std::size_t gdim = _P->mesh()->geometry().dim();

  // Check if it has the right size, always has to come in pairs
  // TODO: do provided values make sense?
  if ((pbc_limits.size() % (gdim * 4)) != 0)
  {
    dolfin_error("advect_particles.cpp::advect_particles",
                 "construct periodic boundary information",
                 "Incorrect shape of pbc_limits provided?");
  }

  std::size_t num_rows = pbc_limits.size() / (gdim * 2);
  for (std::size_t i = 0; i < num_rows; i++)
  {
    std::vector<double> pbc_helper(gdim * 2);
    for (std::size_t j = 0; j < gdim * 2; j++)
      pbc_helper[j] = pbc_limits[i * gdim * 2 + j];

    pbc_lims.push_back(pbc_helper);
  }
  pbc_active = true;

  // Set some other useful info
  _space_dimension = _element->space_dimension();
  _value_size_loc = 1;
  for (std::size_t i = 0; i < _element->value_rank(); i++)
    _value_size_loc *= _element->value_dimension(i);
}
//-----------------------------------------------------------------------------
void advect_particles::update_facets_info()
{
  // Cache midpoint, and normal of each facet in mesh
  // Note that in DOLFIN simplicial cells, Facet f_i is opposite Vertex v_i,
  // etc.

  const Mesh* mesh = _P->mesh();
  std::size_t tdim = mesh->topology().dim();
  const std::size_t num_cell_facets = mesh->type().num_entities(tdim - 1);

  // Information for each facet of the mesh
  facets_info.resize(mesh->num_entities(tdim - 1));

  for (FacetIterator fi(*mesh); !fi.end(); ++fi)
  {
    // Get and store facet normal and facet midpoint
    Point facet_n = fi->normal();
    Point facet_mp = fi->midpoint();
    std::vector<bool> outward_normal;

    // FIXME: could just look at first cell only, simplifies code

    int i = 0;
    for (CellIterator ci(*fi); !ci.end(); ++ci)
    {
      const unsigned int* cell_facets = ci->entities(tdim - 1);

      // Find which facet this is in the cell
      const std::size_t local_index
          = std::find(cell_facets, cell_facets + num_cell_facets, fi->index())
            - cell_facets;
      assert(local_index < num_cell_facets);

      // Get cell vertex opposite facet
      Vertex v(*mesh, ci->entities(0)[local_index]);

      // Take vector from facet midpoint to opposite vertex
      // and compare to facet normal.
      const Point q = v.point() - facet_mp;
      const double dir = q.dot(facet_n);
      assert(std::abs(dir) > 1e-10);
      bool outward_pointing = (dir < 0);

      // Make sure that the facet normal is always outward pointing
      // from Cell 0.
      if (!outward_pointing and i == 0)
      {
        facet_n *= -1.0;
        outward_pointing = true;
      }

      // Store outward normal bool for safety check (below)
      outward_normal.push_back(outward_pointing);
      ++i;
    }

    // Safety check
    if (fi->num_entities(tdim) == 2)
    {
      if (outward_normal[0] == outward_normal[1])
      {
        dolfin_error(
            "advect_particles.cpp::update_facets_info",
            "get correct facet normal direction",
            "The normal cannot be of same direction for neighboring cells");
      }
    }

    // Store info in facets_info array
    const std::size_t index = fi->index();
    facets_info[index].midpoint = facet_mp;
    facets_info[index].normal = facet_n;
  } // End facet iterator
}
//-----------------------------------------------------------------------------
void advect_particles::set_bfacets(std::string btype)
{

  // Type of external facet to set on all external facets
  facet_t external_facet_type;
  if (btype == "closed")
    external_facet_type = facet_t::closed;
  else if (btype == "open")
    external_facet_type = facet_t::open;
  else if (btype == "periodic")
    external_facet_type = facet_t::periodic;
  else
  {
    dolfin_error("advect_particles.cpp", "set external facet type",
                 "Invalid value: %s", btype.c_str());
  }

  const Mesh* mesh = _P->mesh();
  const std::size_t tdim = mesh->topology().dim();
  for (FacetIterator fi(*mesh); !fi.end(); ++fi)
  {
    if (fi->num_global_entities(tdim) == 1)
      facets_info[fi->index()].type = external_facet_type;
    else
      facets_info[fi->index()].type = facet_t::internal;
  }
}
//-----------------------------------------------------------------------------
void advect_particles::set_bfacets(const MeshFunction<std::size_t>& mesh_func)
{
  const Mesh* mesh = _P->mesh();
  const std::size_t tdim = mesh->topology().dim();

  // Check if size matches number of facets in mesh
  assert(mesh_func.size() == mesh->num_facets());

  // Loop over facets to determine type
  for (FacetIterator fi(*mesh); !fi.end(); ++fi)
  {
    if (fi->num_global_entities(tdim) == 1)
    {
      if (mesh_func[fi->index()] == 1)
        facets_info[fi->index()].type = facet_t::closed;
      else if (mesh_func[fi->index()] == 2)
        facets_info[fi->index()].type = facet_t::open;
      else if (mesh_func[fi->index()] == 3)
        facets_info[fi->index()].type = facet_t::periodic;
      else
        dolfin_error("advect_particles.cpp", "set external facet type",
                     "Invalid value, must be 1, 2, or 3");
    }
    else
    {
      assert(mesh_func[fi->index()] == 0);
      facets_info[fi->index()].type = facet_t::internal;
    }
  }
}
//-----------------------------------------------------------------------------
void advect_particles::do_step(double dt)
{
  const Mesh* mesh = _P->mesh();
  const MPI_Comm mpi_comm = mesh->mpi_comm();
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology().dim();

  std::size_t num_processes = MPI::size(mpi_comm);

  // Needed for local reloc
  std::vector<std::array<std::size_t, 3>> reloc;

  for (CellIterator ci(*mesh); !ci.end(); ++ci)
  {
    std::vector<double> coeffs;
    // Restrict once per cell, once per timestep
    Utils::return_expansion_coeffs(coeffs, *ci, uh);

    // Loop over particles in cell
    for (unsigned int i = 0; i < _P->num_cell_particles(ci->index()); i++)
    {
      // FIXME: It might be better to use 'pointer iterator here' as we need to
      // erase from cell2part vector now we decrement iterator int when needed

      Eigen::MatrixXd basis_mat(_value_size_loc, _space_dimension);
      Utils::return_basis_matrix(basis_mat.data(), _P->x(ci->index(), i), *ci,
                                 _element);

      // Compute value at point using expansion coeffs and basis matrix, first
      // convert to Eigen matrix
      Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), _space_dimension);
      Eigen::VectorXd u_p = basis_mat * exp_coeffs;

      // Convert velocity to point
      Point up(gdim, u_p.data());

      std::size_t cidx_recv = ci->index();
      double dt_rem = dt;

      while (dt_rem > 1E-15)
      {
        // Returns facet which is intersected and the time it takes to do so
        std::tuple<std::size_t, double> intersect_info
            = time2intersect(cidx_recv, dt_rem, _P->x(ci->index(), i), up);
        const std::size_t target_facet = std::get<0>(intersect_info);
        const double dt_int = std::get<1>(intersect_info);

        if (target_facet == std::numeric_limits<unsigned int>::max())
        {
          // Then remain within cell, finish time step
          _P->push_particle(dt_rem, up, ci->index(), i);
          dt_rem = 0.0;
          // TODO: if step == last tstep: update particle position old to most
          // recent value If cidx_recv != ci->index(), particles crossed facet
          // and hence need to be relocated
          if (cidx_recv != ci->index())
            reloc.push_back({ci->index(), i, cidx_recv});
        }
        else
        {
          const Facet f(*mesh, target_facet);
          const unsigned int* facet_cells = f.entities(tdim);

          // Two options: if internal (==2) else if boundary
          if (f.num_entities(tdim) == 2)
          {
            // Then we cross facet which has a neighboring cell
            _P->push_particle(dt_int, up, ci->index(), i);

            cidx_recv = (facet_cells[0] == cidx_recv) ? facet_cells[1]
                                                      : facet_cells[0];

            // Update remaining time
            dt_rem -= dt_int;
            if (dt_rem < 1E-15)
            {
              // Then terminate
              dt_rem = 0.0;
              if (cidx_recv != ci->index())
                reloc.push_back({ci->index(), i, cidx_recv});
            }
          }
          else if (f.num_entities(tdim) == 1)
          {
            const facet_t ftype = facets_info[target_facet].type;
            // Then we hit a boundary, but which type?
            if (f.num_global_entities(tdim) == 2)
            {
              assert(ftype == facet_t::internal);
              // Then it is an internal boundary
              // Do a full push
              _P->push_particle(dt_rem, up, ci->index(), i);
              dt_rem *= 0.;

              if (pbc_active)
                pbc_limits_violation(ci->index(),
                                     i); // Check on sequence crossing internal
                                         // bc -> crossing periodic bc
              // TODO: do same for closed bcs to handle (unlikely event):
              // internal bc-> closed bc

              // Go to the particle communicator
              reloc.push_back(
                  {ci->index(), i, std::numeric_limits<unsigned int>::max()});
            }
            else if (ftype == facet_t::open)
            {
              // Particle leaves the domain. Simply erase!
              // FIXME: additional check that particle indeed leaves domain
              // (u\cdotn > 0)
              // Send to "off process" (should just disappear)
              //
              // Issue 12 Work around: do a full push to make sure that
              // particle is pushed outside domain
              _P->push_particle(dt_rem, up, ci->index(), i);

              // Then push back to relocate
              reloc.push_back(
                  {ci->index(), i, std::numeric_limits<unsigned int>::max()});
              dt_rem = 0.0;
            }
            else if (ftype == facet_t::closed)
            {
              // Closed BC
              apply_closed_bc(dt_int, up, ci->index(), i, target_facet);
              dt_rem -= dt_int;
            }
            else if (ftype == facet_t::periodic)
            {
              // Then periodic bc
              apply_periodic_bc(dt_rem, up, ci->index(), i, target_facet);
              if (num_processes > 1) // Behavior in parallel
                reloc.push_back(
                    {ci->index(), i, std::numeric_limits<unsigned int>::max()});
              else
              {
                // Behavior in serial
                std::size_t cell_id = _P->mesh()
                                          ->bounding_box_tree()
                                          ->compute_first_entity_collision(
                                              _P->x(ci->index(), i));
                reloc.push_back({ci->index(), i, cell_id});
              }
              dt_rem = 0.0;
            }
            else
            {
              dolfin_error("advect_particles.cpp::do_step",
                           "encountered unknown boundary",
                           "Only internal boundaries implemented yet");
            }
          }
          else
          {
            dolfin_error("advect_particles.cpp::do_step",
                         "found incorrect number of facets (<1 or > 2)",
                         "Unknown");
          }
        } // end else
      }   // end while
    }     // end for
  }       // end for

  // Relocate local and global
  _P->relocate(reloc);
}
//-----------------------------------------------------------------------------
std::tuple<std::size_t, double>
advect_particles::time2intersect(std::size_t cidx, double dt, const Point xp,
                                 const Point up)
{
  // Time to facet intersection
  const Mesh* mesh = _P->mesh();
  const std::size_t tdim = mesh->topology().dim();
  double dt_int = std::numeric_limits<double>::max();
  std::size_t target_facet = std::numeric_limits<unsigned int>::max();

  Cell c(*mesh, cidx);
  for (unsigned int i = 0; i < c.num_entities(tdim - 1); ++i)
  {
    std::size_t fidx = c.entities(tdim - 1)[i];
    Facet f(*mesh, fidx);

    Point normal = facets_info[fidx].normal;

    // Normal points outward from Cell 0, so reverse if this is Cell 1 of the
    // Facet
    if (f.entities(tdim)[0] != cidx)
      normal *= -1.0;

    // Compute distance to point. For procedure, see Haworth (2010). Though it
    // is slightly modified
    double h = f.distance(xp);

    // double dtintd = std::max(0., h / (up.dot(normal)) ); //See Haworth
    double denom = up.dot(normal);
    if (denom > 0. && denom < 1e-8)
      denom *= -1.; // If up orth to normal --> blows up timestep

    double dtintd = h / denom;
    // TODO: is this robust for: 1) very small h? 2) infinite number?
    if ((dtintd < dt_int && dtintd > 0. && h > 1E-10)
        || (h < 1E-10 && denom > 0.))
    {
      dt_int = dtintd;
      // Then hit a face or located exactly at a face with non-zero velocity in
      // outward normal direction
      if (dt_int <= dt)
      {
        target_facet = fidx;
      }
    }
  }
  // Store and return intersect info in tuple
  std::tuple<std::size_t, double> intersect_info(target_facet, dt_int);
  return intersect_info;
}
//-----------------------------------------------------------------------------
void advect_particles::apply_open_bc(std::size_t cidx, std::size_t pidx)
{
  _P->delete_particle(cidx, pidx);
}
//-----------------------------------------------------------------------------
void advect_particles::apply_closed_bc(double dt, Point& up, std::size_t cidx,
                                       std::size_t pidx, std::size_t fidx)
{
  // First push particle
  _P->push_particle(dt, up, cidx, pidx);
  // Mirror velocity
  Point normal = facets_info[fidx].normal;
  up -= 2 * (up.dot(normal)) * normal;
}
//-----------------------------------------------------------------------------
void advect_particles::apply_periodic_bc(double dt, Point& up, std::size_t cidx,
                                         std::size_t pidx, std::size_t fidx)
{
  const std::size_t gdim = _P->mesh()->geometry().dim();
  Point midpoint = facets_info[fidx].midpoint;
  std::size_t row_match = std::numeric_limits<unsigned int>::max();
  std::size_t row_friend;
  std::size_t component;
  bool hit = false;
  for (std::size_t i = 0; i < pbc_lims.size(); i++)
  {
    for (std::size_t j = 0; j < gdim; j++)
    {
      if (std::abs(midpoint[j] - pbc_lims[i][j * 2]) < 1E-10
          && std::abs(midpoint[j] - pbc_lims[i][j * 2 + 1]) < 1E-10)
      {
        // Then we most likely found a match, but check if midpoint coordinates
        // are in between the limits for the other coordinate directions
        hit = true;
        for (std::size_t k = 0; k < gdim; k++)
        {
          if (k == j)
            continue;
          // New formulation
          if (midpoint[k] <= pbc_lims[i][k * 2]
              || midpoint[k] >= pbc_lims[i][k * 2 + 1])
            hit = false;
        }
        if (hit)
        {
          row_match = i;
          component = j;
          goto break_me;
        }
      }
    }
  }

break_me:
  // Throw an error if rowmatch not set at this point
  if (row_match == std::numeric_limits<unsigned int>::max())
    dolfin_error("advect_particles.cpp::apply_periodic_bc",
                 "find matching periodic boundary info", "Unknown");
  // Column and matchin column come in pairs
  if (row_match % 2 == 0)
  {
    // Find the uneven friend
    row_friend = row_match + 1;
  }
  else
  {
    // Find the even friend
    row_friend = row_match - 1;
  }

  // For multistep/multistage (!?) schemes, you may need to copy the old
  // position before doing the actual push
  _P->push_particle(dt, up, cidx, pidx);

  // Point formulation
  Point x = _P->x(cidx, pidx);
  x[component] += pbc_lims[row_friend][component * 2]
                  - pbc_lims[row_match][component * 2];

  // Corners can be tricky, therefore include this test
  for (std::size_t i = 0; i < gdim; i++)
  {
    if (i == component)
      continue; // Skip this
    if (x[i] < pbc_lims[row_match][i * 2])
    {
      // Then we push the particle to the other end of domain
      x[i] += (pbc_lims[row_friend][i * 2 + 1] - pbc_lims[row_match][i * 2]);
    }
    else if (x[i] > pbc_lims[row_match][i * 2 + 1])
    {
      x[i] -= (pbc_lims[row_match][i * 2 + 1] - pbc_lims[row_friend][i * 2]);
    }
  }
  _P->set_property(cidx, pidx, 0, x);
}
//-----------------------------------------------------------------------------
void advect_particles::pbc_limits_violation(std::size_t cidx, std::size_t pidx)
{
  // This method guarantees that particles can cross internal bc -> periodic bc
  // in one time step without being deleted.
  // FIXME: more efficient implementation??
  // FIXME: can give troubles when domain decomposition results in one cell in
  // domain corner Check if periodic bcs are violated somewhere, if so, modify
  // particle position
  std::size_t gdim = _P->mesh()->geometry().dim();

  Point x = _P->x(cidx, pidx);

  for (std::size_t i = 0; i < pbc_lims.size() / 2; i++)
  {
    for (std::size_t j = 0; j < gdim; j++)
    {
      if (std::abs(pbc_lims[2 * i][2 * j] - pbc_lims[2 * i][2 * j + 1]) < 1E-13)
      {
        if (x[j] > pbc_lims[2 * i][2 * j] && x[j] > pbc_lims[2 * i + 1][2 * j])
        {
          x[j] -= (std::max(pbc_lims[2 * i][2 * j], pbc_lims[2 * i + 1][2 * j])
                   - std::min(pbc_lims[2 * i][2 * j],
                              pbc_lims[2 * i + 1][2 * j]));
          // Check whether the other bounds are violated, to handle corners
          // FIXME: cannot handle cases where domain of friend in one direction
          // is different from match, reason: looping over periodic bc pairs
          for (std::size_t k = 0; k < gdim; k++)
          {
            if (k == j)
              continue;
            if (x[k] < pbc_lims[2 * i][2 * k])
            {
              x[k] += (pbc_lims[2 * i + 1][2 * k + 1] - pbc_lims[2 * i][2 * k]);
            }
            else if (x[k] > pbc_lims[2 * i][2 * k + 1])
            {
              x[k] -= (pbc_lims[2 * i][2 * k + 1] - pbc_lims[2 * i + 1][2 * k]);
            }
          }
        }
        else if (x[j] < pbc_lims[2 * i][2 * j]
                 && x[j] < pbc_lims[2 * i + 1][2 * j])
        {
          x[j] += (std::max(pbc_lims[2 * i][2 * j], pbc_lims[2 * i + 1][2 * j])
                   - std::min(pbc_lims[2 * i][2 * j],
                              pbc_lims[2 * i + 1][2 * j]));
          // Check wheter the other bounds are violated, to handle corners
          for (std::size_t k = 0; k < gdim; k++)
          {
            if (k == j)
              continue;
            if (_P->x(cidx, pidx)[k] < pbc_lims[2 * i][2 * k])
            {
              x[k] += (pbc_lims[2 * i + 1][2 * k + 1] - pbc_lims[2 * i][2 * k]);
            }
            else if (x[k] > pbc_lims[2 * i][2 * k + 1])
            {
              x[k] -= (pbc_lims[2 * i][2 * k + 1] - pbc_lims[2 * i + 1][2 * k]);
            }
          }
        } // else do nothing
      }
    }
  }
  _P->set_property(cidx, pidx, 0, x);
}
//-----------------------------------------------------------------------------
void advect_particles::do_substep(
    double dt, Point& up, const std::size_t cidx, std::size_t pidx,
    const std::size_t step, const std::size_t num_steps,
    const std::size_t xp0_idx, const std::size_t up0_idx,
    std::vector<std::array<std::size_t, 3>>& reloc)
{
  double dt_rem = dt;

  const Mesh* mesh = _P->mesh();
  const std::size_t mpi_size = MPI::size(mesh->mpi_comm());
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology().dim();

  std::size_t cidx_recv = std::numeric_limits<unsigned int>::max();

  if (step == 0)
    cidx_recv = cidx;
  else
  {
    // The reason for doing this step is:
    // for the multistep (RK) schemes, the carried old position may not be the
    // same as the cell where the particle lives newest position is always
    // carried
    // TODO: Can we think of smarter implementation?
    cidx_recv = mesh->bounding_box_tree()->compute_first_entity_collision(
        _P->x(cidx, pidx));

    // One alternative might be:
    // Cell cell(*(_P->_mesh), cidx);
    // bool contain = cell.contains(_P->_cell2part[cidx][pidx][0])
    // If true  cidx_recv = cidx; and continue
    // if not: do entity collision

    // FIXME: this approach is robust for the internal points multistep schemes,
    // but what about multistage schemes and near closed/periodic bc's?
    if (cidx_recv == std::numeric_limits<unsigned int>::max())
    {
      _P->push_particle(dt_rem, up, cidx, pidx);
      if (pbc_active)
        pbc_limits_violation(cidx, pidx);

      if (step == (num_steps - 1))
      {
        // Copy current position to old position
        // so something like
        _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));
      }
      // Apparently, this always lead to a communicate, but why?
      reloc.push_back({cidx, pidx, std::numeric_limits<unsigned int>::max()});
      return; // Stop right here
    }
  }

  bool hit_cbc = false; // Hit closed boundary condition (?!)
  while (dt_rem > 1E-15)
  {
    // Returns facet which is intersected and the time it takes to do so
    std::tuple<std::size_t, double> intersect_info
        = time2intersect(cidx_recv, dt_rem, _P->x(cidx, pidx), up);
    const std::size_t target_facet = std::get<0>(intersect_info);
    const double dt_int = std::get<1>(intersect_info);

    if (target_facet == std::numeric_limits<unsigned int>::max())
    {
      // Then remain within cell, finish time step
      _P->push_particle(dt_rem, up, cidx, pidx);
      dt_rem = 0.0;

      if (step == (num_steps - 1))
        // Copy current position to old position
        _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

      // If cidx_recv != ci->index(), particles crossed facet and hence need to
      // be relocated
      if (cidx_recv != cidx)
        reloc.push_back({cidx, pidx, cidx_recv});
    }
    else
    {
      Facet f(*mesh, target_facet);
      const unsigned int* fcells = f.entities(tdim);

      // Two options: if internal (==2) else if boundary
      if (f.num_entities(tdim) == 2)
      {
        // Then we cross facet which has a neighboring cell
        _P->push_particle(dt_int, up, cidx, pidx);

        // Update index of receiving cell
        cidx_recv = (fcells[0] == cidx_recv) ? fcells[1] : fcells[0];

        // Update remaining time
        dt_rem -= dt_int;
        if (dt_rem < 1E-15)
        {
          // Then terminate
          dt_rem *= 0.;
          // Copy current position to old position
          if (step == (num_steps - 1))
            _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

          if (cidx_recv != cidx)
            reloc.push_back({cidx, pidx, cidx_recv});
        }
      }
      else if (f.num_entities(tdim) == 1)
      {
        const facet_t ftype = facets_info[target_facet].type;
        // Then we hit a boundary, but which type?
        if (f.num_global_entities(tdim) == 2)
        { // Internal boundary between processes
          assert(ftype == facet_t::internal);
          _P->push_particle(dt_rem, up, cidx, pidx);
          dt_rem = 0.0;

          // Updates particle position if pbc_limits is violated
          if (pbc_active)
            pbc_limits_violation(cidx, pidx);

          // Copy current position to old position
          if (step == (num_steps - 1) || hit_cbc)
            _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

          reloc.push_back(
              {cidx, pidx, std::numeric_limits<unsigned int>::max()});

          return; // Stop right here
        }
        else if (ftype == facet_t::open)
        {
          // Particle leaves the domain. Relocate to another process (particle
          // will be discarded)

          // Issue 12 work around: do full push to push particle outside
          // domain
          _P->push_particle(dt_rem, up, cidx, pidx);

          // Then push back to relocate
          reloc.push_back(
              {cidx, pidx, std::numeric_limits<unsigned int>::max()});
          dt_rem = 0.0;
        }
        else if (ftype == facet_t::closed)
        {
          apply_closed_bc(dt_int, up, cidx, pidx, target_facet);
          dt_rem -= dt_int;

          // TODO: CHECK THIS
          dt_rem
              += (1. - dti[step]) * (dt / dti[step]); // Make timestep complete
          // If we hit a closed bc, modify following, probably is first order:

          // TODO: UPDATE AS PARTICLE!
          std::vector<double> dummy_vel(gdim,
                                        std::numeric_limits<double>::max());
          _P->set_property(cidx, pidx, up0_idx, Point(gdim, dummy_vel.data()));

          hit_cbc = true;
        }
        else if (ftype == facet_t::periodic)
        {
          // TODO: add support for periodic bcs
          apply_periodic_bc(dt_rem, up, cidx, pidx, target_facet);

          // Copy current position to old position
          if (step == (num_steps - 1))
            _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

          // Behavior in parallel
          // Always do a global push
          if (mpi_size > 1)
          {
            reloc.push_back(
                {cidx, pidx, std::numeric_limits<unsigned int>::max()});
          }
          else
          {
            // Behavior in serial
            // TODO: call particle locate
            std::size_t cell_id
                = mesh->bounding_box_tree()->compute_first_entity_collision(
                    _P->x(cidx, pidx));

            reloc.push_back({cidx, pidx, cell_id});
          }

          dt_rem = 0.0;
        }
        else
        {
          dolfin_error("advect_particles.cpp::do_step",
                       "encountered unknown boundary",
                       "Only internal boundaries implemented yet");
        }
      }
      else
      {
        dolfin_error("advect_particles.cpp::do_step",
                     "found incorrect number of facets (<1 or > 2)", "Unknown");
      }
    }
  } // end_while
}
//-----------------------------------------------------------------------------
advect_particles::~advect_particles() {}
//
//-----------------------------------------------------------------------------
//
//      RUNGE KUTTA 2
//
//-----------------------------------------------------------------------------
//
advect_rk2::advect_rk2(particles& P, FunctionSpace& U, Function& uhi,
                       const std::string type1)
    : advect_particles(P, U, uhi, type1)
{
  update_particle_template();
  init_weights();
}
//-----------------------------------------------------------------------------
advect_rk2::advect_rk2(
    particles& P, FunctionSpace& U, Function& uhi, const std::string type1,
    Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits)
    : advect_particles(P, U, uhi, type1, pbc_limits)
{
  update_particle_template();
  init_weights();
}
//-----------------------------------------------------------------------------
advect_rk2::advect_rk2(particles& P, FunctionSpace& U, Function& uhi,
                       const MeshFunction<std::size_t>& mesh_func)
    : advect_particles(P, U, uhi, mesh_func)

{
  update_particle_template();
  init_weights();
}
//-----------------------------------------------------------------------------
advect_rk2::advect_rk2(particles& P, FunctionSpace& U, Function& uhi,
                       const MeshFunction<std::size_t>& mesh_func,
                       Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits)
    : advect_particles(P, U, uhi, mesh_func, pbc_limits)

{
  update_particle_template();
  init_weights();
}
//-----------------------------------------------------------------------------
void advect_rk2::do_step(double dt)
{
  if (dt <= 0.)
    dolfin_error("advect_particles.cpp::step", "set timestep.",
                 "Timestep should be > 0.");

  const Mesh* mesh = _P->mesh();
  std::size_t gdim = mesh->geometry().dim();

  std::vector<std::vector<double>> coeffs_storage(mesh->num_cells());
  std::size_t num_substeps = 2;

  for (std::size_t step = 0; step < num_substeps; step++)
  {
    // Needed for local reloc
    std::vector<std::array<std::size_t, 3>> reloc;

    for (CellIterator ci(*mesh); !ci.end(); ++ci)
    {
      if (step == 0)
      { // Restrict once per cell, once per timestep
        std::vector<double> coeffs;
        Utils::return_expansion_coeffs(coeffs, *ci, uh);
        coeffs_storage[ci->index()].insert(coeffs_storage[ci->index()].end(),
                                           coeffs.begin(), coeffs.end());
      }

      for (std::size_t i = 0; i < _P->num_cell_particles(ci->index()); i++)
      {
        Eigen::MatrixXd basis_mat(_value_size_loc, _space_dimension);
        Utils::return_basis_matrix(basis_mat.data(), _P->x(ci->index(), i), *ci,
                                   _element);

        // Compute value at point using expansion coeffs and basis matrix, first
        // convert to Eigen matrix
        Eigen::Map<Eigen::VectorXd> exp_coeffs(
            coeffs_storage[ci->index()].data(), _space_dimension);
        Eigen::VectorXd u_p = basis_mat * exp_coeffs;

        Point up(gdim, u_p.data());
        if (step == 0)
          _P->set_property(ci->index(), i, up0_idx, up);
        else
        {
          // Goto next particle, this particle hitted closed bound
          if (_P->property(ci->index(), i, up0_idx)[0]
              == std::numeric_limits<double>::max())
            continue;
          up += _P->property(ci->index(), i, up0_idx);
          up *= 0.5;
        }

        // Reset position to old
        if (step == 1)
          _P->set_property(ci->index(), i, 0,
                           _P->property(ci->index(), i, xp0_idx));

        // Do substep
        do_substep(dt, up, ci->index(), i, step, num_substeps, xp0_idx, up0_idx,
                   reloc);
      }
    }

    // Relocate local and global
    _P->relocate(reloc);
  }
}
//-----------------------------------------------------------------------------
advect_rk2::~advect_rk2() {}
//
//-----------------------------------------------------------------------------
//
//      RUNGE KUTTA 3
//
//-----------------------------------------------------------------------------
//
advect_rk3::advect_rk3(particles& P, FunctionSpace& U, Function& uhi,
                       const std::string type1)
    : advect_particles(P, U, uhi, type1)
{
  update_particle_template();
  init_weights();
}
//-----------------------------------------------------------------------------
advect_rk3::advect_rk3(
    particles& P, FunctionSpace& U, Function& uhi, const std::string type1,
    Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits)
    : advect_particles(P, U, uhi, type1, pbc_limits)
{
  update_particle_template();
  init_weights();
}
//-----------------------------------------------------------------------------
advect_rk3::advect_rk3(particles& P, FunctionSpace& U, Function& uhi,
                       const MeshFunction<std::size_t>& mesh_func)
    : advect_particles(P, U, uhi, mesh_func)

{
  update_particle_template();
  init_weights();
}
//-----------------------------------------------------------------------------
advect_rk3::advect_rk3(particles& P, FunctionSpace& U, Function& uhi,
                       const MeshFunction<std::size_t>& mesh_func,
                       Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits)
    : advect_particles(P, U, uhi, mesh_func, pbc_limits)

{
  update_particle_template();
  init_weights();
}
//-----------------------------------------------------------------------------
void advect_rk3::do_step(double dt)
{
  if (dt < 0.)
    dolfin_error("advect_particles.cpp::step", "set timestep.",
                 "Timestep should be > 0.");

  const Mesh* mesh = _P->mesh();
  const std::size_t gdim = mesh->geometry().dim();
  std::vector<std::vector<double>> coeffs_storage(mesh->num_cells());
  std::size_t num_substeps = 3;

  for (std::size_t step = 0; step < num_substeps; step++)
  {
    // Needed for local reloc
    std::vector<std::array<std::size_t, 3>> reloc;

    for (CellIterator ci(*mesh); !ci.end(); ++ci)
    {
      if (step == 0)
      { // Restrict once per cell, once per timestep
        std::vector<double> coeffs;
        Utils::return_expansion_coeffs(coeffs, *ci, uh);
        coeffs_storage[ci->index()].insert(coeffs_storage[ci->index()].end(),
                                           coeffs.begin(), coeffs.end());
      }

      // Loop over particles
      for (std::size_t i = 0; i < _P->num_cell_particles(ci->index()); i++)
      {
        Eigen::MatrixXd basis_mat(_value_size_loc, _space_dimension);
        Utils::return_basis_matrix(basis_mat.data(), _P->x(ci->index(), i), *ci,
                                   _element);

        // Compute value at point using expansion coeffs and basis matrix, first
        // convert to Eigen matrix
        Eigen::Map<Eigen::VectorXd> exp_coeffs(
            coeffs_storage[ci->index()].data(), _space_dimension);
        Eigen::VectorXd u_p = basis_mat * exp_coeffs;

        Point up(gdim, u_p.data());

        // Then reset position to the old position
        _P->set_property(ci->index(), i, 0,
                         _P->property(ci->index(), i, xp0_idx));

        if (step == 0)
          _P->set_property(ci->index(), i, up0_idx, up * (weights[step]));
        else if (step == 1)
        {
          Point p = _P->property(ci->index(), i, up0_idx);
          if (p[0] == std::numeric_limits<double>::max())
            continue;
          _P->set_property(ci->index(), i, up0_idx, p + up * (weights[step]));
        }
        else if (step == 2)
        {
          Point p = _P->property(ci->index(), i, up0_idx);
          if (p[0] == std::numeric_limits<double>::max())
            continue;
          up *= weights[step];
          up += _P->property(ci->index(), i, up0_idx);
        }

        // Reset position to old
        if (step == 1)
          _P->set_property(ci->index(), i, 0,
                           _P->property(ci->index(), i, xp0_idx));

        // Do substep
        do_substep(dt * dti[step], up, ci->index(), i, step, num_substeps,
                   xp0_idx, up0_idx, reloc);
      }
    }

    // Relocate local and global
    _P->relocate(reloc);
  }
}
//-----------------------------------------------------------------------------
advect_rk3::~advect_rk3() {}
