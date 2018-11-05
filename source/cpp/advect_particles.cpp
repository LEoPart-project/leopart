// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com

#include "advect_particles.h"
using namespace dolfin;

//-----------------------------------------------------------------------------
advect_particles::advect_particles(particles& P, FunctionSpace& U, Function& uhi,
                                   const BoundaryMesh& bmesh, const std::string type1,
                                   const std::string update_particle)
    : _P(&P), uh(&uhi), _element(U.element()), update_particle(update_particle)
{
    /*
     * Following types are distinguished:
     * "open"       --> open boundary
     * "periodic"   --> periodic bc (additional info on extent required)
     * "closed"     --> closed boundary
    */
    set_bfacets(bmesh, type1);

    // If run in parallel, then get interior facet indices
    if (MPI::size(_P->mesh()->mpi_comm()) > 1)
      int_facets = interior_facets();

    // Set facet and cell2facet info
    cell2facet.resize(_P->mesh()->num_cells());
    set_facets_info();

    // Set some other useful info
    _space_dimension = _element->space_dimension();
    _value_size_loc = 1;
    for (std::size_t i = 0; i < _element->value_rank(); i++)
       _value_size_loc *= _element->value_dimension(i);

    // Check input of particle update
    if (this->update_particle != "none"
        && this->update_particle != "vector"
        && this->update_particle != "scalar"
        && this->update_particle != "both")
        dolfin_error("advect_particles.cpp::advect_particles",
                     "could not set particle property updater",
                     "Provide any of: none, scalar, vector, both");
}
//-----------------------------------------------------------------------------
// Using delegate constructors here
advect_particles::advect_particles(particles& P, FunctionSpace& U, Function& uhi,
                                   const BoundaryMesh& bmesh, const std::string type1,
                                   Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits,
                                   const std::string update_particle)
    : advect_particles::advect_particles(P, U, uhi, bmesh, type1, update_particle)
{
  std::size_t gdim = _P->mesh()->geometry().dim();

  // Then the only thing to do: check if type1 was "periodic"
  if (type1 == "periodic")
  {
    // TODO: Perform a check if it has the right size, always has to come in pairs
    // TODO: do provided values make sense?
    if ((pbc_limits.size() % (gdim * 4)) != 0)
    {
      dolfin_error("advect_particles.cpp::advect_particles",
                   "construct periodic boundary information",
                   "Incorrect shape of pbc_limits provided?");
    }


    std::size_t num_rows = pbc_limits.size()/(gdim * 2);
    for(std::size_t i = 0; i < num_rows ; i++ )
    {
      std::vector<double> pbc_helper(gdim * 2 );
      for(std::size_t j = 0; j < gdim * 2; j++)
        pbc_helper[j] = pbc_limits[i * gdim * 2 + j];

      pbc_lims.push_back( pbc_helper );
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
advect_particles::advect_particles(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh,
                                   const std::string type1, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices1,
                                   const std::string type2, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices2,
                                   const std::string update_particle)
    : _P(&P), uh(&uhi), _element( U.element() ), update_particle(update_particle)
{
    if(type1 == type2){
        dolfin_error("advect_particles.cpp::advect_particles","could not initialize advect_particles",
                     "Are boundary 1 and boundary 2 of the same type?");
    }

    set_bfacets(bmesh, type1, indices1);
    set_bfacets(bmesh, type2, indices2);
    if (MPI::size(_P->mesh()->mpi_comm()) > 1)
      int_facets = interior_facets();

    // Length should amount to size of boundary mesh, works in 3D?
    if((obc_facets.size() + cbc_facets.size() + pbc_facets.size()) != bmesh.num_cells())
    {
      std::cout<<"Boundary mesh num cells "<<bmesh.num_cells()<<std::endl;
      std::cout<<"Size open "<<obc_facets.size()<<std::endl;
      std::cout<<"Size closed "<<cbc_facets.size()<<std::endl;
      std::cout<<"Size period "<<pbc_facets.size()<<std::endl;
      dolfin_error("advect_particles.cpp::advect_particles", "set boundary parts",
                   "Size of different boundary parts does not add up to boundary mesh size");
    }

    // Set facet and cell2facet info
    cell2facet.resize(_P->mesh()->num_cells());
    set_facets_info();

    // Set some other useful info
    _space_dimension = _element->space_dimension();
    _value_size_loc = 1;
    for (std::size_t i = 0; i < _element->value_rank(); i++)
       _value_size_loc *= _element->value_dimension(i);

    // Check input of particle update
    if (this->update_particle != "none" && this->update_particle != "vector" &&
        this->update_particle != "scalar" && this->update_particle != "both")
        dolfin_error("advect_particles.cpp::advect_particles",
                     "could not set particle property updater",
                     "Provide any of: none, scalar, vector, both");
}
//-----------------------------------------------------------------------------
advect_particles::advect_particles(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                                   Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices1,
                                   const std::string type2,
                                   Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices2,
                                   Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits,
                                   const std::string update_particle)
    : advect_particles::advect_particles(P, U, uhi, bmesh, type1, indices1, type2, indices2, update_particle)
{
  std::size_t gdim = _P->mesh()->geometry().dim();

  if (type1 == "periodic" || type2 == "periodic")
  {
    if ((pbc_limits.size() % (gdim * 4)) != 0)
    {
      dolfin_error("advect_particles.cpp::advect_particles",
                   "construct periodic boundary information",
                   "Incorrect shape of pbc_limits provided?");
    }

    std::size_t num_rows = pbc_limits.size()/( gdim * 2);
    for (std::size_t i = 0; i < num_rows ; i++)
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
void advect_particles::set_facets_info()
{
  //
  // In 2D, we have the following dimensions
  //      0       Vertices
  //      1       Facets
  //      2       Cells
  //

  std::size_t _cdim = _P->mesh()->topology().dim();
  std::size_t gdim = _P->mesh()->geometry().dim();
  std::size_t _fdim = _cdim - 1;
  std::size_t _vdim = _fdim - 1;

  for (FacetIterator fi(*(_P->mesh())); !fi.end(); ++fi )
  {
    //std::cout<<"Facet Index "<<fi->index()<<std::endl;

    // Get and store facet normal and facet midpoint
    Point facet_n  = fi->normal();
    Point facet_mp = fi->midpoint();
    std::vector<double> facet_n_coords(facet_n.coordinates(), facet_n.coordinates() + gdim);
    std::vector<double> facet_mp_coords(facet_mp.coordinates(), facet_mp.coordinates() + gdim);

    // Initialize facet vertex coordinates (assume simplicial mesh)
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> fv_coords(gdim, gdim);

    // Initialize cell connectivity vector and normal direction
    std::vector<std::size_t> cellfcell; // A facet allways connects 2 elements
    std::vector<bool> outward_normal;

    // Vertex coordinate vector for simplical elements
    std::vector<double> cvertex_coords((gdim + 1) * gdim);
    std::size_t k = 0;
    for (VertexIterator vi(*fi); !vi.end(); ++vi)
    {
      for(std::size_t j = 0; j < gdim; j++)
        fv_coords(k, j) = *(vi->x() + j);
      ++k;
    }

    Eigen::Array<double, 1, Eigen::Dynamic> pv_coords(gdim);
    for (CellIterator ci(*fi); !ci.end(); ++ci)
    {
      //std::cout<<"Neighbor cells"<<ci->index()<<std::endl;
      ci->get_vertex_coordinates(cvertex_coords);

      // Now check if we can find any mismatching vertices
      bool outward_pointing = true;       // By default, we assume outward pointing normal
      for(std::size_t l = 0; l < (gdim + 1) * gdim; l += gdim)
      {
        Eigen::Map<Eigen::Array<double, 1, Eigen::Dynamic>>
          cv_coords(cvertex_coords.data() + l, gdim);
        pv_coords = cv_coords - fv_coords.row(0);

        double l2diff = std::inner_product(facet_n_coords.begin(),
                                           facet_n_coords.end(), pv_coords.data(), 0.0);
        if (l2diff > 1E-10 ){
          outward_pointing = false;
          break;
        }
      }

      // Store relevant data
      cellfcell.push_back(ci->index());
      outward_normal.push_back(outward_pointing);
      cell2facet[ci->index()].push_back(fi->index());
    }

    // Perform some safety checks
    if (cellfcell.size() == 1)
    {
      // Then the facet index must be in one of boundary facet lists
      if ((std::find(int_facets.begin(), int_facets.end(), fi->index()) != int_facets.end()) &&
          (std::find(obc_facets.begin(), obc_facets.end(), fi->index()) != obc_facets.end()) &&
          (std::find(cbc_facets.begin(), cbc_facets.end(), fi->index()) != cbc_facets.end()) &&
          (std::find(pbc_facets.begin(), pbc_facets.end(), fi->index()) != pbc_facets.end())   )
      {
        dolfin_error("advect_particles.cpp::set_facets_info",
                     "get correct facet 2 cell connectivity.",
                     "Detected only one cell neighbour to facet, but cannot find facet in boundary lists.");
      }
    }
    else if(cellfcell.size() == 2)
    {
      if (cellfcell[0] == cellfcell[1])
        dolfin_error("advect_particles.cpp::set_facets_info",
                     "get correct facet 2 cell connectivity.",
                     "Neighboring cells ");
      if (outward_normal[0] == outward_normal[1])
      {
        dolfin_error("advect_particles.cpp::set_facets_info",
                     "get correct facet normal direction",
                     "The normal cannot be of same direction for neighboring cells");
      }

    }
    else
    {
      dolfin_error("advect_particles.cpp::set_facets_info",
                   "get connecting cells",
                   "Each facet should neighbor at max two cells.");
    }

    // Store info in facets_info variable
    facet_info finf({*fi, facet_mp, facet_n, cellfcell, outward_normal});
    facets_info.push_back(finf);
  } // End facet iterator

}
//-----------------------------------------------------------------------------
void advect_particles::set_bfacets(const BoundaryMesh& bmesh, const std::string btype)
{
  if (btype == "closed")
    cbc_facets = boundary_facets(bmesh);
  else if(btype == "open")
    obc_facets = boundary_facets(bmesh);
  else if(btype == "periodic")
    pbc_facets = boundary_facets(bmesh);
  else
  {
    dolfin_error("advect_particles.cpp::set_bfacets",
                 "Unknown boundary type",
                 "Set boundary type correct");
  }
}
//-----------------------------------------------------------------------------
void advect_particles::set_bfacets(const BoundaryMesh& bmesh, const std::string btype, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> bidcs)
{
  if(btype == "closed")
    cbc_facets = boundary_facets(bmesh, bidcs);
  else if(btype == "open")
    obc_facets = boundary_facets(bmesh, bidcs);
  else if(btype == "periodic")
    pbc_facets = boundary_facets(bmesh, bidcs);
  else
  {
    dolfin_error("advect_particles.cpp::set_bfacets",
                 "Unknown boundary type",
                 "Set boundary type correct");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> advect_particles::boundary_facets(const BoundaryMesh& bmesh)
{
  std::size_t d = (_P->mesh()->geometry().dim()) - 1;
  MeshFunction<std::size_t>  boundary_facets = bmesh.entity_map(d);
  std::size_t* val = boundary_facets.values();
  std::vector<std::size_t> bfacet_idcs;

  for (std::size_t i = 0; i<boundary_facets.size(); i++)
  {
    bfacet_idcs.push_back(*(val + i));
    // Make sure that diff equals 0
    Cell fbm(bmesh,i);
    Facet fm(*(_P->mesh()), *(val+i));
    Point diff = fm.midpoint() - fbm.midpoint();
    if (diff.norm() > 1E-10)
      dolfin_error("advect_particles.cpp::boundary_facets 1", "finding facets matching boundary mesh facets",
                         "Unknown");
  }
  return bfacet_idcs;
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> advect_particles::boundary_facets(const BoundaryMesh& bmesh, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> bidcs)
{
    // This method is not yet tested!
    std::size_t d = _P->mesh()->geometry().dim() - 1;
    MeshFunction<std::size_t>  boundary_facets = bmesh.entity_map(d);
    std::size_t* val = boundary_facets.values();
    std::vector<std::size_t> bfacet_idcs;

    for(std::size_t i =0; i<bidcs.size(); i++){
        // Return the facet index on the parent mesh
        bfacet_idcs.push_back( *(val+bidcs[i]) );

        // Debugging only, check if diff equals 0
        Cell fbm(bmesh, bidcs[i]);
        Facet fm(*(_P->mesh()),*(val+bidcs[i]));

        Point diff = fm.midpoint() - fbm.midpoint();
        if(diff.norm() > 1E-10)
            dolfin_error("advect_particles.cpp::boundary_facets 2", "finding facets matching boundary mesh facets",
                         "Unknown");
    }
    return bfacet_idcs;
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> advect_particles::interior_facets()
{
  std::vector<std::size_t> interior_fids;
  std::size_t D = _P->mesh()->topology().dim();
  for (FacetIterator f(*(_P->mesh())); !f.end(); ++f)
    if (f->num_entities(D) == 1
        and f->num_global_entities(D) == 2)
      interior_fids.push_back(f->index());

  return interior_fids;
}
//-----------------------------------------------------------------------------
void advect_particles::do_step(double dt)
{
  const MPI_Comm mpi_comm = _P->mesh()->mpi_comm();
  const std::size_t gdim = _P->mesh()->geometry().dim();

  std::size_t num_processes = MPI::size(mpi_comm);

    // Needed for local reloc
    std::vector<std::size_t> reloc_local_c;
    std::vector<particle>    reloc_local_p;

    for (CellIterator ci(*(_P->mesh())); !ci.end(); ++ci)
    {
        std::vector<double> coeffs;
        // Restrict once per cell, once per timestep
        Utils::return_expansion_coeffs(coeffs, *ci, uh);

        // Loop over particles in cell
        for (int i = 0; i < _P->num_cell_particles(ci->index()); i++)
        {
            // FIXME: It might be better to use 'pointer iterator here' as we need to erase from cell2part vector
            // now we decrement iterator int when needed

            std::vector<double> basis_matrix(_space_dimension * _value_size_loc);

            Utils::return_basis_matrix(basis_matrix, _P->x(ci->index(), i), *ci, _element);

            // Compute value at point using expansion coeffs and basis matrix, first convert to Eigen matrix
            Eigen::Map<Eigen::MatrixXd> basis_mat(basis_matrix.data(), _value_size_loc, _space_dimension);
            Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), _space_dimension);
            Eigen::VectorXd u_p =  basis_mat * exp_coeffs;

            // Convert velocity to point
            Point up(gdim, u_p.data());

            std::size_t cidx_recv = ci->index();
            double dt_rem = dt;

            while(dt_rem > 1E-15)
            {
              // Returns facet which is intersected and the time it takes to do so
              std::tuple<std::size_t, double> intersect_info
                = time2intersect(cidx_recv, dt_rem, _P->x(ci->index(), i), up);
              const std::size_t target_facet = std::get<0>(intersect_info);
              const double dt_int = std::get<1>(intersect_info);

              if (target_facet == std::numeric_limits<std::size_t>::max())
              {
                // Then remain within cell, finish time step
                _P->push_particle(dt_rem, up, ci->index(), i);
                dt_rem = 0.0;
                // TODO: if step == last tstep: update particle position old to most recent value
                // If cidx_recv != ci->index(), particles crossed facet and hence need to be relocated
                if (cidx_recv != ci->index())
                {
                  reloc_local_c.push_back(cidx_recv);
                  reloc_local_p.push_back(_P->get_particle(ci->index(), i));
                  _P->delete_particle(ci->index(), i);
                  i--; //Decrement iterator
                }
              }
              else
              {
                const Facet f = facets_info[target_facet].facet;
                const std::size_t D = _P->mesh()->topology().dim();
                const unsigned int* facet_cells = f.entities(D);

                // Two options: if internal (==2) else if boundary
                if (f.num_entities(D) == 2)
                {
                  // Then we cross facet which has a neighboring cell
                  _P->push_particle(dt_int, up, ci->index(), i);

                  cidx_recv = (facet_cells[0] == cidx_recv) ? facet_cells[1] : facet_cells[0];

                  // Update remaining time
                  dt_rem -= dt_int;
                  if (dt_rem < 1E-15)
                  {
                    // Then terminate
                    dt_rem = 0.0;
                    if (cidx_recv != ci->index())
                    {
                      reloc_local_c.push_back(cidx_recv);
                      reloc_local_p.push_back(_P->get_particle(ci->index(),i));
                      _P->delete_particle(ci->index(), i);
                      i--; // Decrement iterator
                    }
                  }
                }
                else if (f.num_entities(D) == 1)
                {
                  // Then we hit a boundary, but which type?
                  if (std::find(int_facets.begin(), int_facets.end(),  target_facet) != int_facets.end())
                  {
                    // Then it is an internal boundary
                    // Do a full push
                    _P->push_particle(dt_rem, up, ci->index(), i);
                    dt_rem *= 0.;

                    if (pbc_active)
                      pbc_limits_violation(ci->index(),i); // Check on sequence crossing internal bc -> crossing periodic bc
                    // TODO: do same for closed bcs to handle (unlikely event): internal bc-> closed bc

                    // Go to the particle communicator
                    _P->particle_communicator_collect(ci->index(), i);
                    i--;
                  }
                  else if(std::find(obc_facets.begin(), obc_facets.end(),  target_facet) != obc_facets.end())
                  {
                    // Particle leaves the domain. Simply erase!
                    // FIXME: additional check that particle indeed leaves domain (u\cdotn > 0)
                    apply_open_bc(ci->index(), i);
                    dt_rem *= 0.;
                    i--;
                  }
                  else if(std::find(cbc_facets.begin(), cbc_facets.end(),  target_facet) != cbc_facets.end())
                  {
                    // Closed BC
                    apply_closed_bc(dt_int, up, ci->index(), i, target_facet );
                    dt_rem -= dt_int;
                  }
                  else if(std::find(pbc_facets.begin(), pbc_facets.end(),  target_facet) != pbc_facets.end())
                  {
                    // Then periodic bc
                    apply_periodic_bc(dt_rem, up, ci->index(), i, target_facet );
                    if (num_processes > 1) //Behavior in parallel
                      _P->particle_communicator_collect(ci->index(), i);
                    else
                    {
                      // Behavior in serial
                      std::size_t cell_id = _P->mesh()->bounding_box_tree()->compute_first_entity_collision( _P->x(ci->index(), i));
                      reloc_local_c.push_back(cell_id);
                      reloc_local_p.push_back(_P->get_particle(ci->index(), i));
                      _P->delete_particle(ci->index(), i);
                    }
                    dt_rem *= 0.;
                    i--;
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
            } // end while
        } // end for
    } // end for

    // Relocate local
    for (std::size_t i = 0; i < reloc_local_c.size(); ++i)
    {
      if (reloc_local_c[i] != std::numeric_limits<unsigned int>::max())
        _P->add_particle(reloc_local_c[i], reloc_local_p[i]);
      else
      {
        dolfin_error("advection.cpp::do_step",
                     "find a hosting cell on local process", "Unknown");
      }
    }

    // Debug only
    /*
    for (std::size_t p = 0; p < num_processes; p++){
        std::cout<<"Size of comm_snd at process "<<p<<" "<<comm_snd[p].size()<<std::endl;
    }
    */

    // Relocate global
    if (num_processes > 1)
      _P->particle_communicator_push();
}
//-----------------------------------------------------------------------------
std::tuple<std::size_t, double> advect_particles::time2intersect(std::size_t cidx, double dt, const Point xp, const Point up){
    // Time to facet intersection
    double dt_int = std::numeric_limits<double>::max();
    std::size_t target_facet = std::numeric_limits<std::size_t>::max();

    for (std::size_t fidx: cell2facet[cidx])
    {
      Point normal = facets_info[fidx].normal;

        // Make sure normal is outward pointing
        std::vector<std::size_t>::iterator it;
        it = std::find(facets_info[fidx].cells.begin(), facets_info[fidx].cells.end(), cidx);
        std::size_t it_idx = std::distance(facets_info[fidx].cells.begin(), it);
        if (it_idx == facets_info[fidx].cells.size())
        {
            dolfin_error("advect_particles.cpp::time2intersect","did not find matching facet cell connectivity",
                                      "Unknown");
        }
        else
        {
            bool outward_normal = facets_info[fidx].outward[it_idx];
            if (!outward_normal) normal *= -1.;
        }

        // Compute distance to point. For procedure, see Haworth (2010). Though it is slightly modified
        double h = facets_info[fidx].facet.distance(xp);

        //double dtintd = std::max(0., h / (up.dot(normal)) ); //See Haworth
        double denom  = up.dot(normal);
        if(denom > 0. && denom < 1e-8) denom *= -1.; // If up orth to normal --> blows up timestep

        double dtintd = h / denom;
        // TODO: is this robust for: 1) very small h? 2) infinite number?
        if( ( dtintd < dt_int && dtintd > 0. && h > 1E-10) || ( h<1E-10 && denom > 0. ) ){
            dt_int = dtintd;
            // Then hit a face or located exactly at a face with non-zero velocity in outward
            // normal direction
            if(dt_int <= dt){
                target_facet = fidx;
            }
        }
    }
    // Store and return intersect info in tuple
    std::tuple<std::size_t, double> intersect_info(target_facet, dt_int);
    return intersect_info;
}
//-----------------------------------------------------------------------------
void advect_particles::apply_open_bc(std::size_t cidx, std::size_t pidx){
  _P->delete_particle(cidx, pidx);
}
//-----------------------------------------------------------------------------
void advect_particles::apply_closed_bc(double dt, Point& up, std::size_t cidx, std::size_t pidx, std::size_t fidx ){
    // First push particle
    _P->push_particle(dt, up, cidx, pidx);
    // Mirror velocity
    Point normal = facets_info[fidx].normal;
    up -= 2*(up.dot(normal))*normal;
}
//-----------------------------------------------------------------------------
void advect_particles::apply_periodic_bc(double dt, Point& up, std::size_t cidx,  std::size_t pidx, std::size_t fidx){
  const std::size_t gdim = _P->mesh()->geometry().dim();
    Point midpoint = facets_info[fidx].midpoint;
    std::size_t row_match = std::numeric_limits<std::size_t>::max();
    std::size_t row_friend;
    std::size_t component;
    bool hit = false;
    for(std::size_t i = 0; i < pbc_lims.size(); i++){
        for(std::size_t j = 0; j < gdim; j++){
            if( std::abs(midpoint[j] - pbc_lims[i][ j*2 ]) < 1E-10
                    &&
                std::abs(midpoint[j] - pbc_lims[i][ j*2  + 1 ]) < 1E-10 ){
                // Then we most likely found a match, but check if midpoint coordinates are in between the limits
                // for the other coordinate directions
                hit = true;
                for(std::size_t k = 0; k < gdim; k++){
                    if( k == j ) continue;
                    // New formulation
                    if( midpoint[k] <= pbc_lims[i][ k*2 ]  || midpoint[k] >= pbc_lims[i][ k*2 + 1 ] )
                        hit = false;
                }
                if(hit){
                    row_match = i ;
                    component = j;
                    goto break_me;
                }
            }
        }
    }

    break_me:
    // Throw an error if rowmatch not set at this point
    if( row_match == std::numeric_limits<std::size_t>::max() )
        dolfin_error("advect_particles.cpp::apply_periodic_bc", "find matching periodic boundary info", "Unknown");
    // Column and matchin column come in pairs
    if( row_match % 2 == 0 ){
        // Find the uneven friend
        row_friend = row_match + 1;
    }else{
        // Find the even friend
        row_friend = row_match - 1;
    }

    // For multistep/multistage (!?) schemes, you may need to copy the old position before doing the actual push
    _P->push_particle(dt, up, cidx, pidx);

    // Point formulation
    Point x = _P->x(cidx, pidx);
    x[component] += pbc_lims[row_friend][component*2] - pbc_lims[row_match][component*2];

    // Corners can be tricky, therefore include this test
    for(std::size_t i = 0; i < gdim; i++)
    {
      if (i == component) continue; // Skip this
      if (x[i] < pbc_lims[row_match][i*2])
      {
        // Then we push the particle to the other end of domain
        x[i] += (pbc_lims[row_friend][i*2 + 1] - pbc_lims[row_match][i*2]);
      }
      else if (x[i] > pbc_lims[row_match][i*2 + 1])
      {
        x[i] -= (pbc_lims[row_match][i*2 + 1] - pbc_lims[row_friend][i*2]);
      }
    }
    _P->set_property(cidx, pidx, 0, x);
}
//-----------------------------------------------------------------------------
void advect_particles::pbc_limits_violation(std::size_t cidx, std::size_t pidx){
    // This method guarantees that particles can cross internal bc -> periodic bc in one
    // time step without being deleted.
    // FIXME: more efficient implementation??
    // FIXME: can give troubles when domain decomposition results in one cell in domain corner
    // Check if periodic bcs are violated somewhere, if so, modify particle position
  std::size_t gdim = _P->mesh()->geometry().dim();

  Point x = _P->x(cidx, pidx);

  for (std::size_t i = 0; i < pbc_lims.size()/2; i++)
  {
    for (std::size_t j = 0; j< gdim; j++)
    {
      if (std::abs(pbc_lims[2*i][2*j] - pbc_lims[2*i][2*j+1]) < 1E-13 )
      {
        if(x[j] > pbc_lims[2*i][2*j] &&
           x[j] > pbc_lims[2*i+1][2*j])
        {
          x[j] -=
            (std::max(pbc_lims[2*i][2*j], pbc_lims[2*i+1][2*j]) -
             std::min(pbc_lims[2*i][2*j], pbc_lims[2*i+1][2*j]) );
          // Check whether the other bounds are violated, to handle corners
          // FIXME: cannot handle cases where domain of friend in one direction is different from
          // match, reason: looping over periodic bc pairs
          for (std::size_t k = 0; k < gdim; k++)
          {
            if (k == j) continue;
            if (x[k] < pbc_lims[2*i][2*k])
            {
              x[k] += (pbc_lims[2*i + 1][2*k + 1]  - pbc_lims[2*i][2*k]);
            }
            else if (x[k] > pbc_lims[2*i][2*k + 1])
            {
              x[k]  -= (pbc_lims[2*i][2*k + 1] - pbc_lims[2*i + 1][2*k]);
            }
          }
        }
        else if (x[j] < pbc_lims[2*i][2*j] && x[j] < pbc_lims[2*i+1][2*j])
        {
          x[j] +=
            (std::max(pbc_lims[2*i][2*j], pbc_lims[2*i+1][2*j]) -
             std::min(pbc_lims[2*i][2*j], pbc_lims[2*i+1][2*j]) );
          // Check wheter the other bounds are violated, to handle corners
          for(std::size_t k = 0; k < gdim; k++)
          {
            if (k == j) continue;
            if (_P->x(cidx, pidx)[k] < pbc_lims[2*i][2*k])
            {
              x[k] += (pbc_lims[2*i + 1][2*k + 1]  - pbc_lims[2*i][2*k]);
            }
            else if (x[k] > pbc_lims[2*i][2*k + 1])
            {
              x[k] -= (pbc_lims[2*i][2*k + 1] - pbc_lims[2*i + 1][2*k]);
            }
          }
        } // else do nothing
      }
    }
  }
  _P->set_property(cidx, pidx, 0, x);
}
//-----------------------------------------------------------------------------
void advect_particles::do_substep(double dt, Point& up, const std::size_t cidx, std::size_t* pidx,
                                  const std::size_t step, const std::size_t num_steps,
                                  const std::size_t xp0_idx, const std::size_t up0_idx,
                                  std::vector<std::size_t>& reloc_local_c, std::vector<particle>& reloc_local_p){
    double dt_rem = dt;
    //std::size_t cidx_recv = cidx;

    const std::size_t mpi_size = MPI::size(_P->mesh()->mpi_comm());
    const std::size_t gdim = _P->mesh()->geometry().dim();
    std::size_t cidx_recv = std::numeric_limits<std::size_t>::max();

    if (step == 0)
      cidx_recv = cidx;
    else
    {
      // The reason for doing this step is:
      // for the multistep (RK) schemes, the carried old position may not be the same as the cell
      // where the particle lives newest position is always carried
      // TODO: Can we think of smarter implementation?
      cidx_recv = _P->mesh()->bounding_box_tree()->compute_first_entity_collision( _P->x(cidx, *pidx));

      // One alternative might be:
      // Cell cell(*(_P->_mesh), cidx);
      // bool contain = cell.contains(_P->_cell2part[cidx][*pidx][0])
      // If true  cidx_recv = cidx; and continue
      // if not: do entity collision

      // FIXME: this approach is robust for the internal points multistep schemes, but what about multistage schemes and
      // near closed/periodic bc's?
      if (cidx_recv == std::numeric_limits<unsigned int>::max())
      {
        _P->push_particle(dt_rem, up, cidx, *pidx);
        if (pbc_active)
          pbc_limits_violation(cidx,*pidx);

        if(step == (num_steps - 1))
        {
          // Copy current position to old position
          // so something like
          _P->set_property(cidx, *pidx, xp0_idx, _P->x(cidx, *pidx));
        }
        // Apparently, this always lead to a communicate, but why?
        _P->particle_communicator_collect(cidx, *pidx);
        (*pidx)--;
        return; // Stop right here
      }
    }

    bool hit_cbc = false; // Hit closed boundary condition (?!)
    while(dt_rem > 1E-15)
    {
      // Returns facet which is intersected and the time it takes to do so
      std::tuple<std::size_t, double> intersect_info
        = time2intersect(cidx_recv, dt_rem, _P->x(cidx, *pidx), up);
      const std::size_t target_facet = std::get<0>(intersect_info);
      const double dt_int = std::get<1>(intersect_info);

      if (target_facet == std::numeric_limits<std::size_t>::max())
      {
          // Then remain within cell, finish time step
          _P->push_particle(dt_rem, up, cidx, *pidx);
          dt_rem *= 0.;

            if (step == (num_steps - 1))
              // Copy current position to old position
              _P->set_property(cidx, *pidx, xp0_idx, _P->x(cidx, *pidx));

            // If cidx_recv != ci->index(), particles crossed facet and hence need to be relocated
            if (cidx_recv != cidx)
            {
              // Then relocate local
              reloc_local_c.push_back(cidx_recv);
              reloc_local_p.push_back(_P->get_particle(cidx, *pidx));
              _P->delete_particle(cidx, *pidx);
              (*pidx)--; //Decrement iterator
            }
        }
        else
        {
          // Two options: if internal (==2) else if boundary
          if (facets_info[target_facet].cells.size() == 2 )
          {
            // Then we cross facet which has a neighboring cell
            _P->push_particle(dt_int, up, cidx, *pidx);

            // Update index of receiving cell
            for (std::size_t cidx_iter: facets_info[target_facet].cells)
            {
              if (cidx_iter != cidx_recv)
              {
                cidx_recv = cidx_iter;
                break;
              }
            }

            // Update remaining time
            dt_rem -= dt_int;
            if (dt_rem < 1E-15)
            {
              // Then terminate
              dt_rem *= 0.;
              // Copy current position to old position
              if (step == (num_steps - 1))
                _P->set_property(cidx, *pidx, xp0_idx, _P->x(cidx, *pidx));

              if(cidx_recv != cidx ){
                reloc_local_c.push_back(cidx_recv);
                reloc_local_p.push_back(_P->get_particle(cidx, *pidx));
                _P->delete_particle(cidx, *pidx);
                (*pidx)--; //Decrement iterator
              }
            }
          }
          else if (facets_info[target_facet].cells.size() == 1)
          {
            // Then we hit a boundary, but which type?
            if(std::find(int_facets.begin(), int_facets.end(), target_facet) != int_facets.end())
            {
              _P->push_particle(dt_rem, up, cidx, *pidx);
              dt_rem *= 0.;

              if(pbc_active) pbc_limits_violation(cidx,*pidx); // Updates particle position if pbc_limits is violated
              // Copy current position to old position
              if(step == (num_steps - 1) || hit_cbc)
                _P->set_property(cidx, *pidx, xp0_idx, _P->x(cidx, *pidx));

              _P->particle_communicator_collect(cidx, *pidx);
              (*pidx)--;
              return; // Stop right here
            }
            else if(std::find(obc_facets.begin(), obc_facets.end(), target_facet) != obc_facets.end())
            {
              // Particle leaves the domain. Simply erase!
              apply_open_bc(cidx, *pidx);
              dt_rem *= 0.;
              (*pidx)--;
            }
            else if(std::find(cbc_facets.begin(), cbc_facets.end(), target_facet) != cbc_facets.end())
            {
              apply_closed_bc(dt_int, up, cidx, *pidx, target_facet);
              dt_rem -= dt_int;

              // TODO: CHECK THIS
              dt_rem += (1. - dti[step]) * (dt/dti[step]); // Make timestep complete
              // If we hit a closed bc, modify following, probably is first order:

              // TODO: UPDATE AS PARTICLE!
              std::vector<double> dummy_vel(gdim, std::numeric_limits<double>::max());
              _P->set_property(cidx, *pidx, up0_idx, Point(gdim, dummy_vel.data()));

              hit_cbc = true;
            }
            else if(std::find(pbc_facets.begin(), pbc_facets.end(), target_facet) != pbc_facets.end())
            {
              //TODO: add support for periodic bcs
              apply_periodic_bc(dt_rem, up, cidx, *pidx, target_facet);

              // Copy current position to old position
              if(step == (num_steps - 1) )
                _P->set_property(cidx, *pidx, xp0_idx, _P->x(cidx, *pidx));

              if (mpi_size > 1)
              {
                // Behavior in parallel
                // Always do a global push
                _P->particle_communicator_collect(cidx, *pidx);
              }
              else
              {
                // Behavior in serial
                // TODO: call particle locate
                std::size_t cell_id = _P->mesh()->bounding_box_tree()->compute_first_entity_collision(_P->x(cidx, *pidx));

                reloc_local_c.push_back(cell_id);
                reloc_local_p.push_back(_P->get_particle(cidx, *pidx));
                _P->delete_particle(cidx, *pidx);
              }

              dt_rem *= 0.;
              (*pidx)--;
            }
            else
            {
              dolfin_error("advect_particles.cpp::do_step","encountered unknown boundary", "Only internal boundaries implemented yet");
            }
          }
          else
          {
            dolfin_error("advect_particles.cpp::do_step","found incorrect number of facets (<1 or > 2)", "Unknown");
          }
        }
    }// end_while
}
//-----------------------------------------------------------------------------
advect_particles::~advect_particles(){}
//
//-----------------------------------------------------------------------------
//
//      RUNGE KUTTA 2
//
//-----------------------------------------------------------------------------
//
advect_rk2::advect_rk2(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                       const std::string update_particle)
        : advect_particles(P, U, uhi, bmesh, type1, update_particle)
{
    update_particle_template();
    init_weights();
}
//-----------------------------------------------------------------------------
advect_rk2::advect_rk2(particles& P, FunctionSpace& U, Function& uhi,
                       const BoundaryMesh& bmesh, const std::string type1, Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits,
                        const std::string update_particle)
        : advect_particles(P, U, uhi, bmesh, type1, pbc_limits, update_particle)
{
    update_particle_template();
    init_weights();
}
//-----------------------------------------------------------------------------
advect_rk2::advect_rk2(particles& P, FunctionSpace& U, Function& uhi,
                       const BoundaryMesh& bmesh, const std::string type1, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices1,
                        const std::string type2, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices2,
                        const std::string update_particle)
        : advect_particles(P, U, uhi, bmesh, type1, indices1, type2, indices2, update_particle)
{
    update_particle_template();
    init_weights();
}
//-----------------------------------------------------------------------------
advect_rk2::advect_rk2(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh,
                       const std::string type1, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices1,
                       const std::string type2, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices2,
                       Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits,
                       const std::string update_particle)
        : advect_particles(P, U, uhi, bmesh, type1, indices1, type2, indices2, pbc_limits, update_particle)
{
    update_particle_template();
    init_weights();
}
//-----------------------------------------------------------------------------
void advect_rk2::do_step(double dt){
    if(dt <= 0.)
        dolfin_error("advect_particles.cpp::step", "set timestep.", "Timestep should be > 0.");

    const MPI_Comm mpi_comm = _P->mesh()->mpi_comm();
    std::size_t gdim = _P->mesh()->geometry().dim();

    std::size_t num_processes = MPI::size(mpi_comm);

    std::vector<std::vector<double> > coeffs_storage(_P->mesh()->num_cells());
    std::size_t num_substeps = 2;

    for (std::size_t step = 0; step < num_substeps; step++)
    {
        // Needed for local reloc
        std::vector<std::size_t> reloc_local_c;
        std::vector<particle>    reloc_local_p;

        for (CellIterator ci(*(_P->mesh())); !ci.end(); ++ci)
        {
            if(step == 0 )
            { // Restrict once per cell, once per timestep
                std::vector<double> coeffs;
                Utils::return_expansion_coeffs(coeffs, *ci, uh);
                coeffs_storage[ci->index()].insert(coeffs_storage[ci->index()].end(), coeffs.begin(), coeffs.end());
            }

            for(std::size_t i = 0; i < _P->num_cell_particles(ci->index()) ; i++){
                std::vector<double> basis_matrix(_space_dimension * _value_size_loc);
                Utils::return_basis_matrix(basis_matrix, _P->x(ci->index(), i), *ci, _element);

                // Compute value at point using expansion coeffs and basis matrix, first convert to Eigen matrix
                Eigen::Map<Eigen::MatrixXd> basis_mat(basis_matrix.data(), _value_size_loc, _space_dimension);
                Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs_storage[ci->index()].data(), _space_dimension);
                Eigen::VectorXd u_p =  basis_mat * exp_coeffs ;

                Point up(gdim, u_p.data() );
                if (step == 0)
                  _P->set_property(ci->index(), i, up0_idx, up);
                else
                {
                  // Goto next particle, this particle hitted closed bound
                  if(_P->property(ci->index(), i, up0_idx)[0] == std::numeric_limits<double>::max()) continue;
                  up += _P->property(ci->index(), i, up0_idx);
                  up *= 0.5;
                }

                // Reset position to old
                if(step == 1)
                  _P->set_property(ci->index(), i, 0,
                                   _P->property(ci->index(), i, xp0_idx));

                // Do substep
                do_substep(dt, up, ci->index(), &i, step, num_substeps,
                           xp0_idx, up0_idx, reloc_local_c, reloc_local_p);
            }
        }

        // Local relocate
        for (std::size_t i = 0; i < reloc_local_c.size(); i++)
        {
          if (reloc_local_c[i] != std::numeric_limits<unsigned int>::max())
            _P->add_particle(reloc_local_c[i], reloc_local_p[i]);
          else
          {
            dolfin_error("advection_rk2.cpp::do_step",
                         "find a hosting cell on local process", "Unknown");
          }
        }

        // Global relocate
        if (num_processes > 1)
          _P->particle_communicator_push();
    }
}
//-----------------------------------------------------------------------------
advect_rk2::~advect_rk2(){}
//
//-----------------------------------------------------------------------------
//
//      RUNGE KUTTA 3
//
//-----------------------------------------------------------------------------
//
advect_rk3::advect_rk3(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                       const std::string update_particle)
        : advect_particles(P, U, uhi, bmesh, type1, update_particle)
{
    update_particle_template();
    init_weights();
}
//-----------------------------------------------------------------------------
advect_rk3::advect_rk3(particles& P, FunctionSpace& U, Function& uhi,
                       const BoundaryMesh& bmesh, const std::string type1, Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits,
                        const std::string update_particle)
        : advect_particles(P, U, uhi, bmesh, type1, pbc_limits, update_particle)
{
    update_particle_template();
    init_weights();
}
//-----------------------------------------------------------------------------
advect_rk3::advect_rk3(particles& P, FunctionSpace& U, Function& uhi,
                       const BoundaryMesh& bmesh, const std::string type1, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices1,
                        const std::string type2, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices2,
                        const std::string update_particle)
        : advect_particles(P, U, uhi, bmesh, type1, indices1, type2, indices2, update_particle)
{
    update_particle_template();
    init_weights();
}
//-----------------------------------------------------------------------------
advect_rk3::advect_rk3(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                       Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices1,
                       const std::string type2, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices2,
                       Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits,
                       const std::string update_particle)
        : advect_particles(P, U, uhi, bmesh, type1, indices1, type2, indices2, pbc_limits, update_particle )
{
    update_particle_template();
    init_weights();
}
//-----------------------------------------------------------------------------
void advect_rk3::do_step(double dt){
    if(dt < 0.)
        dolfin_error("advect_particles.cpp::step", "set timestep.", "Timestep should be > 0.");

    const MPI_Comm mpi_comm = _P->mesh()->mpi_comm();
    const std::size_t gdim = _P->mesh()->geometry().dim();
    std::size_t num_processes = MPI::size(mpi_comm);

    std::vector<std::vector<double> > coeffs_storage(_P->mesh()->num_cells());
    std::size_t num_substeps = 3;

    for(std::size_t step = 0; step < num_substeps; step++){
        // Needed for local reloc
        std::vector<std::size_t> reloc_local_c;
        std::vector<particle>    reloc_local_p;

        for( CellIterator ci(*(_P->mesh())); !ci.end(); ++ci){
            if(step == 0 ){ // Restrict once per cell, once per timestep
                std::vector<double> coeffs;
                Utils::return_expansion_coeffs(coeffs, *ci, uh);
                coeffs_storage[ci->index()].insert(coeffs_storage[ci->index()].end(), coeffs.begin(), coeffs.end());
            }

            // Loop over particles
            for(std::size_t i = 0; i < _P->num_cell_particles(ci->index()) ; i++){
                std::vector<double> basis_matrix(_space_dimension * _value_size_loc);

                Utils::return_basis_matrix(basis_matrix, _P->x(ci->index(), i), *ci, _element);

                // Compute value at point using expansion coeffs and basis matrix, first convert to Eigen matrix
                Eigen::Map<Eigen::MatrixXd> basis_mat(basis_matrix.data(), _value_size_loc, _space_dimension);
                Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs_storage[ci->index()].data(), _space_dimension);
                Eigen::VectorXd u_p =  basis_mat * exp_coeffs ;

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
                  if(p[0] == std::numeric_limits<double>::max())
                    continue;
                  up *= weights[step];
                  up += _P->property(ci->index(), i, up0_idx);
                }

                // Reset position to old
                if(step == 1)
                  _P->set_property(ci->index(), i, 0,
                                   _P->property(ci->index(), i, xp0_idx));

                // Do substep
                do_substep(dt*dti[step], up, ci->index(), &i, step, num_substeps,
                           xp0_idx, up0_idx, reloc_local_c, reloc_local_p);
            }
        }

        // Local relocate
        for(std::size_t i = 0; i<reloc_local_c.size(); i++)
        {
          if(reloc_local_c[i] != std::numeric_limits<unsigned int>::max())
            _P->add_particle(reloc_local_c[i], reloc_local_p[i]);
          else
          {
            dolfin_error("advection_rk2.cpp::do_step",
                         "find a hosting cell on local process", "Unknown");
          }
        }

        // Global relocate
        if(num_processes > 1) _P->particle_communicator_push();
    }
}
//-----------------------------------------------------------------------------
advect_rk3::~advect_rk3(){}
