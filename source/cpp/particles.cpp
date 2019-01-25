// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <limits>

#include <dolfin/fem/FiniteElement.h>

#include "particles.h"
#include "utils.h"

using namespace dolfin;

particles::~particles() {}

particles::particles(
    Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        p_array,
    const std::vector<unsigned int>& p_template, const Mesh& mesh)
    : _mesh(&mesh), _ptemplate(p_template), _mpi_comm(mesh.mpi_comm())
{
  // Note: p_array is 2D [num_particles, property_data]

  // Get geometry dimension of mesh
  _Ndim = mesh.geometry().dim();

  // Set up communication array
  _comm_snd.resize(MPI::size(_mpi_comm));

  _cell2part.resize(mesh.num_cells());

  // Calculate the offset for each particle property and overall size
  std::vector<unsigned int> offset = {0};
  for (const auto& p : p_template)
    offset.push_back(offset.back() + p);
  _plen = offset.back();

  // Loop over particles:
  for (Eigen::Index i = 0; i < p_array.rows(); i++)
  {
    // Position and get hosting cell
    Point xp(_Ndim, p_array.row(i).data());

    unsigned int cell_id
        = _mesh->bounding_box_tree()->compute_first_entity_collision(xp);
    if (cell_id != std::numeric_limits<unsigned int>::max())
    {
      // Initialize particle with position
      particle pnew = {xp};

      for (std::size_t j = 1; j < _ptemplate.size(); ++j)
      {
        Point property(_ptemplate[j], p_array.row(i).data() + offset[j]);
        pnew.push_back(property);
      }

      // TO DO: FLIP type advection requires that particle also
      // carries the old values

      // Push back to particle structure
      _cell2part[cell_id].push_back(pnew);
    }
  }
}
//-----------------------------------------------------------------------------
int particles::add_particle(int c)
{
  particle p(_ptemplate.size());
  _cell2part[c].push_back(p);
  return _cell2part[c].size() - 1;
}
//-----------------------------------------------------------------------------
void particles::interpolate(const Function& phih,
                            const std::size_t property_idx)
{
  std::size_t space_dimension, value_size_loc;
  space_dimension = phih.function_space()->element()->space_dimension();
  value_size_loc = 1;
  for (std::size_t i = 0; i < phih.function_space()->element()->value_rank();
       i++)
    value_size_loc *= phih.function_space()->element()->value_dimension(i);

  if (value_size_loc != _ptemplate[property_idx])
    dolfin_error("particles::interpolate", "get property idx",
                 "Local value size mismatches particle template property");

  for (CellIterator cell(*(_mesh)); !cell.end(); ++cell)
  {
    std::vector<double> coeffs;
    Utils::return_expansion_coeffs(coeffs, *cell, &phih);
    for (std::size_t pidx = 0; pidx < num_cell_particles(cell->index()); pidx++)
    {
      Eigen::MatrixXd basis_mat(value_size_loc, space_dimension);
      Utils::return_basis_matrix(basis_mat.data(), x(cell->index(), pidx),
                                 *cell, phih.function_space()->element());

      Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), space_dimension);
      Eigen::VectorXd phi_p = basis_mat * exp_coeffs;

      // Then update
      Point phi_point(_ptemplate[property_idx], phi_p.data());
      _cell2part[cell->index()][pidx][property_idx] = phi_point;
    }
  }
}

void particles::increment(const Function& phih_new, const Function& phih_old,
                          const std::size_t property_idx)
{
  if (!phih_new.in(*(phih_old.function_space())))
  {
    dolfin_error("particles.cpp::increment", "Compute increment",
                 "Expected Functions to be in the same FunctionSpace");
  }

  std::size_t space_dimension, value_size_loc;
  space_dimension = phih_new.function_space()->element()->space_dimension();

  value_size_loc = 1;
  for (std::size_t i = 0;
       i < phih_new.function_space()->element()->value_rank(); i++)
    value_size_loc *= phih_new.function_space()->element()->value_dimension(i);

  if (value_size_loc != _ptemplate[property_idx])
    dolfin_error("particles::increment", "get property idx",
                 "Local value size mismatches particle template property");

  for (CellIterator cell(*(_mesh)); !cell.end(); ++cell)
  {
    std::vector<double> coeffs_new, coeffs_old, coeffs;
    Utils::return_expansion_coeffs(coeffs_new, *cell, &phih_new);
    Utils::return_expansion_coeffs(coeffs_old, *cell, &phih_old);

    // Just average to get the coefficients
    for (std::size_t i = 0; i < coeffs_new.size(); i++)
      coeffs.push_back(coeffs_new[i] - coeffs_old[i]);

    for (std::size_t pidx = 0; pidx < num_cell_particles(cell->index()); pidx++)
    {
      Eigen::MatrixXd basis_mat(value_size_loc, space_dimension);
      Utils::return_basis_matrix(basis_mat.data(), x(cell->index(), pidx),
                                 *cell, phih_new.function_space()->element());

      Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), space_dimension);
      Eigen::VectorXd delta_phi = basis_mat * exp_coeffs;

      // Then update
      Point delta_phi_p(_ptemplate[property_idx], delta_phi.data());
      _cell2part[cell->index()][pidx][property_idx] += delta_phi_p;
    }
  }
}

void particles::increment(
    const Function& phih_new, const Function& phih_old,
    Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>>
        property_idcs,
    const double theta, const std::size_t step)
{
  if (!phih_new.in(*(phih_old.function_space())))
  {
    dolfin_error("particles.cpp::increment", "Compute increment",
                 "Expected Functions to be in the same FunctionSpace");
  }

  // Check if size =2 and
  if (property_idcs.size() != 2)
    dolfin_error("particles.cpp::increment", "Set property array",
                 "Property indices must come in pairs");
  if (property_idcs[1] <= property_idcs[0])
    dolfin_error("particles.cpp::increment", "Set property array",
                 "Property must be sorted in ascending order");

  // Check if length of slots matches
  if (_ptemplate[property_idcs[0]] != _ptemplate[property_idcs[1]])
    dolfin_error("particles.cpp::increment", "Set property array",
                 "Found none ore incorrect size at particle slot");

  std::size_t space_dimension, value_size_loc;
  space_dimension = phih_new.function_space()->element()->space_dimension();

  value_size_loc = 1;
  for (std::size_t i = 0;
       i < phih_new.function_space()->element()->value_rank(); i++)
    value_size_loc *= phih_new.function_space()->element()->value_dimension(i);

  if (value_size_loc != _ptemplate[property_idcs[0]])
    dolfin_error("particles::increment", "get property idx",
                 "Local value size mismatches particle template property");

  for (CellIterator cell(*(_mesh)); !cell.end(); ++cell)
  {
    std::vector<double> coeffs_new, coeffs_old, coeffs;
    Utils::return_expansion_coeffs(coeffs_new, *cell, &phih_new);
    Utils::return_expansion_coeffs(coeffs_old, *cell, &phih_old);

    // Just average to get the coefficients
    for (std::size_t i = 0; i < coeffs_new.size(); i++)
      coeffs.push_back(coeffs_new[i] - coeffs_old[i]);

    for (std::size_t pidx = 0; pidx < num_cell_particles(cell->index()); pidx++)
    {
      Eigen::MatrixXd basis_mat(value_size_loc, space_dimension);
      Utils::return_basis_matrix(basis_mat.data(), x(cell->index(), pidx),
                                 *cell, phih_new.function_space()->element());

      Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), space_dimension);
      Eigen::VectorXd delta_phi = basis_mat * exp_coeffs;

      Point delta_phi_p(_ptemplate[property_idcs[0]], delta_phi.data());
      // Do the update
      if (step == 1)
      {
        _cell2part[cell->index()][pidx][property_idcs[0]] += delta_phi_p;
      }
      if (step != 1)
      {
        _cell2part[cell->index()][pidx][property_idcs[0]]
            += theta * delta_phi_p
               + (1. - theta)
                     * _cell2part[cell->index()][pidx][property_idcs[1]];
      }
      _cell2part[cell->index()][pidx][property_idcs[1]] = delta_phi_p;
    }
  }
}

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
particles::positions()
{
  // Could just use get_property(0)

  std::size_t n_particles = 0;
  for (const auto& c2p : _cell2part)
    n_particles += c2p.size();

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xp(
      n_particles, _Ndim);

  // Iterate over cells and particles in each cell
  std::size_t row = 0;
  for (const auto& c2p : _cell2part)
  {
    for (const auto& p : c2p)
    {
      for (std::size_t k = 0; k < _Ndim; ++k)
        xp(row, k) = p[0][k];
      ++row;
    }
  }
  assert(row == n_particles);

  return xp;
}

std::vector<double> particles::get_property(const std::size_t idx)
{

  // Test if idx is valid
  if (idx > _ptemplate.size())
    dolfin_error("particles::get_property", "return index",
                 "Requested index exceeds particle template");
  const std::size_t property_dim = _ptemplate[idx];

  std::size_t n_particles = 0;
  for (const auto& c2p : _cell2part)
    n_particles += c2p.size();

  std::vector<double> property_vector;
  property_vector.reserve(n_particles * property_dim);

  // Iterate over cells and particles in each cell
  for (const auto& c2p : _cell2part)
  {
    for (const auto& p : c2p)
    {
      for (std::size_t k = 0; k < property_dim; k++)
        property_vector.push_back(p[idx][k]);
    }
  }
  return property_vector;
}

void particles::push_particle(const double dt, const Point& up,
                              const std::size_t cidx, const std::size_t pidx)
{
  _cell2part[cidx][pidx][0] += up * dt;
}

void particles::particle_communicator_collect(const std::size_t cidx,
                                              const std::size_t pidx)
{
  // Assertion to check if comm_snd has size of num_procs
  dolfin_assert(_comm_snd.size() == MPI::size(_mpi_comm));

  // Get position
  particle ptemp = _cell2part[cidx][pidx];

  const std::vector<unsigned int> procs
      = _mesh->bounding_box_tree()->compute_process_collisions(x(cidx, pidx));

  // Loop over processes
  for (const auto& p : procs)
    _comm_snd[p].push_back(ptemp);
}

void particles::particle_communicator_push()
{
  // Assertion if sender has correct size
  const std::size_t num_processes = MPI::size(_mpi_comm);
  dolfin_assert(_comm_snd.size() == num_processes);

  std::vector<std::vector<double>> comm_snd_vec(num_processes);
  std::vector<double> comm_rcv_vec;

  // Prepare for communication
  // Convert each vector of Points to std::vector<double>
  for (std::size_t p = 0; p < num_processes; p++)
  {
    for (particle part : _comm_snd[p])
    {
      std::vector<double> unpacked = unpack_particle(part);
      comm_snd_vec[p].insert(comm_snd_vec[p].end(), unpacked.begin(),
                             unpacked.end());
    }
    _comm_snd[p].clear(); // Reset array for next time
  }

  // Communicate with all_to_all
  MPI::all_to_all(_mpi_comm, comm_snd_vec, comm_rcv_vec);

  // TODO: thoroughly test this unpacking -> sending -> composing loop

  std::size_t pos_iter = 0;
  while (pos_iter < comm_rcv_vec.size())
  {
    // This is always the particle position
    Point xp(_Ndim, &comm_rcv_vec[pos_iter]);
    unsigned int cell_id
        = _mesh->bounding_box_tree()->compute_first_entity_collision(xp);
    if (cell_id != std::numeric_limits<unsigned int>::max())
    {
      pos_iter += _Ndim; // Add geometric dimension to iterator
      particle pnew = {xp};
      for (std::size_t j = 1; j < _ptemplate.size(); ++j)
      {
        Point property(_ptemplate[j], &comm_rcv_vec[pos_iter]);
        pnew.push_back(property);
        pos_iter += _ptemplate[j]; // Add property dimension to iterator
      }
      // Iterator position must be multiple of _plen
      dolfin_assert(pos_iter % _plen == 0);

      // Push back new particle to hosting cell
      _cell2part[cell_id].push_back(pnew);
    }
    else
    {
      // Jump to following particle in array
      pos_iter += _plen;
    }
  }
}

void particles::relocate()
{
  // Method to relocate particles on moving mesh

  // Update bounding boxes
  _mesh->bounding_box_tree()->build(*(_mesh));

  // Init relocate local
  std::vector<std::array<std::size_t, 3>> reloc;

  // Loop over particles
  for (CellIterator ci(*(_mesh)); !ci.end(); ++ci)
  {
    // Loop over particles in cell
    for (unsigned int i = 0; i < num_cell_particles(ci->index()); i++)
    {
      Point xp = _cell2part[ci->index()][i][0];

      // If cell does not contain particle, then find new cell
      if (!ci->contains(xp))
      {
        // Do entity collision
        std::size_t cell_id
            = _mesh->bounding_box_tree()->compute_first_entity_collision(xp);

        reloc.push_back({ci->index(), i, cell_id});
      }
    }
  }

  relocate(reloc);
}

std::vector<double> particles::unpack_particle(const particle part)
{
  // Unpack particle into std::vector<double>
  std::vector<double> part_unpacked;
  for (std::size_t i = 0; i < _ptemplate.size(); i++)
    part_unpacked.insert(part_unpacked.end(), part[i].coordinates(),
                         part[i].coordinates() + _ptemplate[i]);
  return part_unpacked;
}

void particles::get_particle_contributions(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& q,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& f, const Cell& dolfin_cell,
    std::shared_ptr<const FiniteElement> element,
    const std::size_t space_dimension, const std::size_t value_size_loc,
    const std::size_t property_idx)
{
  // TODO: some checks if element type matches property index
  if (value_size_loc != _ptemplate[property_idx])
    dolfin_error("particles::get_particle_contributions", "get property idx",
                 "Local value size mismatches particle template property");

  // Get cell index and num particles
  std::size_t cidx = dolfin_cell.index();
  std::size_t Npc = num_cell_particles(cidx);

  // Get and set cell data
  std::vector<double> vertex_coordinates;
  dolfin_cell.get_vertex_coordinates(vertex_coordinates);
  ufc::cell ufc_cell;
  dolfin_cell.get_cell_data(ufc_cell);

  // Resize return values
  q.resize(space_dimension, Npc * value_size_loc);
  f.resize(Npc * value_size_loc);

  if (Npc > 0)
  {
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        basis_matrix(space_dimension, value_size_loc);
    for (std::size_t pidx = 0; pidx < Npc; pidx++)
    {
      std::size_t lb = pidx * value_size_loc;

      element->evaluate_basis_all(
          basis_matrix.data(), x(cidx, pidx).coordinates(),
          vertex_coordinates.data(), ufc_cell.orientation);

      // Place in matrix
      q.block(0, lb, space_dimension, value_size_loc) = basis_matrix;

      // Place in vector
      for (std::size_t m = 0; m < value_size_loc; ++m)
        f(m + lb) = _cell2part[cidx][pidx][property_idx][m];
    }
  }
  else
  {
    // TODO: make function recognize FunctionSpace.ufl_element().family()!

    // Encountered empty cell
    for (std::size_t it = 0; it < vertex_coordinates.size(); it += _Ndim)
    {
      std::cout << "Coordinates vertex " << it << std::endl;
      for (std::size_t jt = 0; jt < _Ndim; ++jt)
      {
        std::cout << vertex_coordinates[it + jt] << std::endl;
      }
    }
    dolfin_error("pdestaticcondensations.cpp::project", "perform projection",
                 "Cells without particle not yet handled, empty cell (%d)",
                 cidx);
  }
}

void particles::relocate(std::vector<std::array<std::size_t, 3>>& reloc)
{
  const std::size_t mpi_size = MPI::size(_mpi_comm);

  // Relocate local and global
  for (const auto& r : reloc)
  {
    const std::size_t& cidx = r[0];
    const std::size_t& pidx = r[1];
    const std::size_t& cidx_recv = r[2];

    if (cidx_recv == std::numeric_limits<unsigned int>::max())
    {
      if (mpi_size > 1)
        particle_communicator_collect(cidx, pidx);
    }
    else
    {
      particle p = _cell2part[cidx][pidx];
      _cell2part[cidx_recv].push_back(p);
    }
  }

  // Sort into reverse order
  std::sort(reloc.rbegin(), reloc.rend());
  for (const auto& r : reloc)
  {
    const std::size_t& cidx = r[0];
    const std::size_t& pidx = r[1];
    delete_particle(cidx, pidx);
  }

  // Relocate global
  if (mpi_size > 1)
    particle_communicator_push();
}
