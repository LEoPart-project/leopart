// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com

#include "advect_particles.h"
using namespace dolfin;

//-----------------------------------------------------------------------------
advect_particles::advect_particles( particles& P, FunctionSpace& U, Function& uhi,
                                    const BoundaryMesh& bmesh, const std::string type1,
                                    const std::string update_particle)
    : _P(&P), uh(&uhi), _element( U.element() ), update_particle(update_particle)
{
    /*
     * Following types are distinghuished:
     * "open"       --> open boundary
     * "periodic"   --> periodic bc (additional info on extent required)
     * "closed"     --> closed boundary
    */
    set_bfacets(bmesh, type1);

    // If run in parallel, then get interior facet indices
    if(MPI::size(_P->mesh()->mpi_comm()) > 1) int_facets = interior_facets();

    // Set facet and cell2facet info
    cell2facet.resize(_P->mesh()->num_cells());
    set_facets_info();

    // Set some other useful info
    _space_dimension = _element->space_dimension();
    _value_size_loc = 1;
    for (std::size_t i = 0; i < _element->value_rank(); i++)
       _value_size_loc *= _element->value_dimension(i);

    // Check input of particle update
    if( this->update_particle != "none" && this->update_particle != "vector" &&
        this->update_particle != "scalar" && this->update_particle != "both")
        dolfin_error("advect_particles.cpp::advect_particles","could not set particle property updater","Provide any of: none, scalar, vector, both");
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
    if(type1 == "periodic"){

        // TODO: Perform a check if it has the right size, always has to come in pairs
        // TODO: do provided values make sense?
        if( (pbc_limits.size() % ( gdim * 4) ) != 0 )
            dolfin_error("advect_particles.cpp::advect_particles","construct periodic boundary information", "Incorrect shape of pbc_limits provided?");

        std::size_t num_rows = pbc_limits.size()/(gdim * 2);
        for(std::size_t i = 0; i < num_rows ; i++ ){
            std::vector<double> pbc_helper(gdim * 2 );
            for(std::size_t j = 0; j < gdim * 2; j++){
                pbc_helper[j] = pbc_limits[i * gdim * 2 + j];
            }
            pbc_lims.push_back( pbc_helper );
        }
        pbc_active = true;
    }else{
        dolfin_error("advect_particles.cpp::advect_particles","could not set pbc_lims","Did you provide limits for a non-periodic BC?");
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
    if (MPI::size(_P->mesh()->mpi_comm()) > 1) int_facets = interior_facets();
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
    if( this->update_particle != "none" && this->update_particle != "vector" &&
        this->update_particle != "scalar" && this->update_particle != "both")
        dolfin_error("advect_particles.cpp::advect_particles","could not set particle property updater","Provide any of: none, scalar, vector, both");
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
    if(type1 == "periodic" || type2 == "periodic"){
        if( (pbc_limits.size() % ( gdim * 4) ) != 0 )
            dolfin_error("advect_particles.cpp::advect_particles","construct periodic boundary information", "Incorrect shape of pbc_limits provided?");
        std::size_t num_rows = pbc_limits.size()/( gdim * 2);
        for(std::size_t i = 0; i < num_rows ; i++ ){
            std::vector<double> pbc_helper( gdim * 2 );
            for(std::size_t j = 0; j < gdim * 2; j++){
                pbc_helper[j] = pbc_limits[i * gdim * 2 + j];
            }
            pbc_lims.push_back( pbc_helper );
        }
        pbc_active = true;
    }else{
        dolfin_error("advect_particles.cpp::advect_particles","could not set pbc_lims","Did you provide limits for a non-periodic BC?");
    }
}
//-----------------------------------------------------------------------------
void advect_particles::set_facets_info(){
    /*
     * In 2D, we have the following dimensions
     *      0       Vertices
     *      1       Facets
     *      2       Cells
    */

  std::size_t _cdim = _P->mesh()->topology().dim();
  std::size_t gdim = _P->mesh()->geometry().dim();
    std::size_t _fdim = _cdim - 1;
    std::size_t _vdim = _fdim - 1;

    for ( FacetIterator fi(*(_P->mesh())); !fi.end(); ++fi )
    {
      Facet f(*(_P->mesh()), fi->index());
        //std::cout<<"Facet Index "<<fi->index()<<std::endl;

        // Get and store facet normal and facet midpoint
        Point facet_n  = f.normal();
        Point facet_mp = f.midpoint();

        double* facet_n_ptr = facet_n.coordinates();
        double* facet_mp_ptr = facet_mp.coordinates();
        std::vector<double> facet_n_coords(gdim);
        std::vector<double> facet_mp_coords(gdim);
        for(std::size_t m = 0; m < gdim; m++){
            facet_n_coords[m] = *(facet_n_ptr  + m);
            facet_mp_coords[m]= *(facet_mp_ptr + m);
        }

        // Initialize facet vertex coordinates (assume symplical mesh)
        std::vector<std::vector<double>> fvertex_coords(gdim);

        // Initialize cell connectivity vector and normal direction
        std::vector<std::size_t> cellfcell; // A facet allways connects 2 elements
        std::vector<bool> outward_normal;
        // Vertex coordinate vector for simplical elements
        std::vector<double> cvertex_coords( (gdim + 1) * gdim );
        std::size_t k = 0;
        for ( VertexIterator vi(f); !vi.end(); ++vi)
        {
          Vertex v(*(_P->mesh()), vi->index());
            for(std::size_t j = 0; j < gdim; j++)
                fvertex_coords[k].push_back(*( v.x() + j ));
            k++;
        } // End vertex iterator
        for (CellIterator ci(f); !ci.end(); ++ci)
        {
          //std::cout<<"Neighbor cells"<<ci->index()<<std::endl;
          Cell c(*(_P->mesh()), ci->index());
          c.get_vertex_coordinates(cvertex_coords);

          // Now check if we can find any mismatching vertices
          bool outward_pointing = true;       // By default, we assume outward pointing normal
          for(std::size_t l = 0; l < (gdim + 1) * gdim; l+= gdim)
          {
              std::vector<double> diff;
              // There must be a better way for subtracting vectors?
              std::vector<double> dummy(gdim);
              for(std::size_t m = 0; m < gdim; m++) dummy[m] = cvertex_coords[l+m];
              std::vector<double> pv = subtract(dummy, fvertex_coords[0]); //Allways just take first facet vertex
              double l2diff= std::inner_product(facet_n_coords.begin(), facet_n_coords.end(), pv.begin(), 0.0);
              if(l2diff > 1E-10 ){
                  outward_pointing = false;
                  break;
              }
          }

          // Store relevant data
          cellfcell.push_back(ci->index());
          outward_normal.push_back(outward_pointing);
          cell2facet[ci->index()].push_back(fi->index());
        } // End cell iterator

        // Perform some safety checks
        if(cellfcell.size() == 1){
          // Then the facet index must be in one of boundary facet lists
            if((std::find(int_facets.begin(), int_facets.end(), fi->index()) != int_facets.end()) &&
               (std::find(obc_facets.begin(), obc_facets.end(), fi->index()) != obc_facets.end()) &&
               (std::find(cbc_facets.begin(), cbc_facets.end(), fi->index()) != cbc_facets.end()) &&
               (std::find(pbc_facets.begin(), pbc_facets.end(), fi->index()) != pbc_facets.end())   ){
                dolfin_error("advect_particles.cpp::set_facets_info", "get correct facet 2 cell connectivity.",
                             "Detected only one cell neighbour to facet, but cannot find facet in boundary lists.");
            }
        }else if(cellfcell.size() == 2){
            if(cellfcell[0] == cellfcell[1])
                dolfin_error("advect_particles.cpp::set_facets_info", "get correct facet 2 cell connectivity.",
                             "Neighboring cells ");
            if(outward_normal[0] == outward_normal[1])
                dolfin_error("advect_particles.cpp::set_facets_info","get correct facet normal direction",
                             "The normal cannot be of same direction for neighboring cells");
        }else{
            dolfin_error("advect_particles.cpp::set_facets_info","get connecting cells",
                         "Each facet should neighbor at max two cells.");
        }
        // Store info in facets_info variable
        facet_info finf(f, facet_mp, facet_n, cellfcell, outward_normal);
        facets_info.push_back(finf);
    } // End facet iterator

    // Some optional checks for cell2facet and facets_info
    /*
    for(std::size_t i = 0; i<facets_info.size(); i++){
        std::cout<<"Facet "<<i<<" is connected to cell(s):"<<std::endl;
        for(std::size_t x : std::get<3>(facets_info[i])) std::cout<<x<<std::endl;
    }
    for(std::size_t i = 0; i < cell2facet.size(); i++){
        std::cout<<"Cell "<<i<<" is connected to facets:"<<std::endl;
        for(std::size_t x: cell2facet[i]) std::cout<<x<<std::endl;
    }
    */
}
//-----------------------------------------------------------------------------
void advect_particles::set_bfacets(const BoundaryMesh& bmesh, const std::string btype){
    if(btype == "closed"){
        cbc_facets = boundary_facets(bmesh);
    }else if(btype == "open"){
        obc_facets = boundary_facets(bmesh);
    }else if(btype == "periodic"){
        pbc_facets = boundary_facets(bmesh);
    }else{
        dolfin_error("advect_particles.cpp::set_bfacets", "Unknown boundary type", "Set boundary type correct");
    }
}
//-----------------------------------------------------------------------------
void advect_particles::set_bfacets(const BoundaryMesh& bmesh, const std::string btype, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> bidcs){
    if(btype == "closed"){
        cbc_facets = boundary_facets(bmesh, bidcs);
    }else if(btype == "open"){
        obc_facets = boundary_facets(bmesh, bidcs);
    }else if(btype == "periodic"){
        pbc_facets = boundary_facets(bmesh, bidcs);
    }else{
        dolfin_error("advect_particles.cpp::set_bfacets", "Unknown boundary type", "Set boundary type correct");
    }
}
//-----------------------------------------------------------------------------
std::vector<double> advect_particles::subtract(std::vector<double>& u, std::vector<double>& v){
    if(u.size() != v.size())
        dolfin_error("advect_particles.cpp::subtract","subtract vectors of different size",
                     "Provide two equally shape vectors");
    std::vector<double> subtract(u.size());
    for(std::size_t j=0; j < u.size(); j++)
    {
        subtract[j] = u[j] - v[j];
    }
    return subtract;
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> advect_particles::boundary_facets(const BoundaryMesh& bmesh){
  std::size_t d = (_P->mesh()->geometry().dim())-1;
    MeshFunction<std::size_t>  boundary_facets = bmesh.entity_map(d);
    std::size_t* val = boundary_facets.values();
    std::vector<std::size_t> bfacet_idcs;

    for(std::size_t i =0; i<boundary_facets.size(); i++){
        bfacet_idcs.push_back( *(val+i) );
        // Make sure that diff equals 0
        Cell fbm(bmesh,i);
        Facet fm(*(_P->mesh()),*(val+i));
        Point diff = fm.midpoint() - fbm.midpoint();
        if(diff.norm() > 1E-10)
            dolfin_error("advect_particles.cpp::boundary_facets 1", "finding facets matching boundary mesh facets",
                         "Unknown");
    }
    return bfacet_idcs;
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> advect_particles::boundary_facets(const BoundaryMesh& bmesh, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> bidcs){
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
std::vector<std::size_t> advect_particles::interior_facets(){
  std::size_t d = _P->mesh()->geometry().dim() - 1;
    BoundaryMesh bmesh(*(_P->mesh()),"interior");
    std::vector<std::size_t> bfacet_idcs = boundary_facets(bmesh);
    return bfacet_idcs;
}
//-----------------------------------------------------------------------------
/* FIXME: ignore moving mesh support
void advect_particles::relocate_particles(){
    // FIXME

    // Naive implementation for particle relocation on
    // moving mesh (with parallel support)
    init_helper_matrices();
    _P->update_bounding_boxes();

    // Needed for local relocation
    std::vector<std::size_t> reloc_local_c;
    std::vector<particle>    reloc_local_p;
    std::vector<vec2d>       reloc_local_xp0;
    std::vector<Point>       reloc_local_up0;

    // Needed for parallel implementation
    const MPI_Comm mpi_comm = _P->_mesh->mpi_comm();
    std::size_t num_processes = MPI::size(mpi_comm);
    std::vector<std::vector<double>> xp_push(num_processes);
    std::vector<std::vector<double>> up_push(num_processes);
    std::vector<std::vector<double>> rhop_push(num_processes);

    std::vector<std::vector<double>> xp0_push(num_processes);
    std::vector<std::vector<double>> up0_push(num_processes);

    for(std::size_t i =0; i < _P->_cell2part.size() ; i++){
        for(std::size_t j=0; j<_P->_cell2part[i].size(); j++){
            vec2d xp = std::get<0>(_P->_cell2part[i][j]);
            Point xp_dolfin(xp[0], xp[1]);

            // FIXME: first check if old cell still contains particle
            // if so 'cycle'
            // if not check neighbors of cell?
            // if not, then do a full locate

            std::size_t cidx_recv = _P->locate_particle(xp_dolfin);
            if( i != cidx_recv){
                // Parallel behavior
                if (cidx_recv != std::numeric_limits<unsigned int>::max() ){
                    reloc_local_c.push_back(cidx_recv);
                    particle temp_particle = _P->_cell2part[i][j];
                    reloc_local_p.push_back(temp_particle);
                    reloc_local_xp0.push_back(xp_o[i][j]);
                    reloc_local_up0.push_back(up_o[i][j]);
                    _P->_cell2part[i].erase(_P->_cell2part[i].begin() + j);
                    xp_o[i].erase(xp_o[i].begin() + j );
                    up_o[i].erase(up_o[i].begin() + j );
                    relocate_local_collector(i,j);
                }else if(num_processes > 1){
                    //std::cout<<"Landed in MPI collector"<<std::endl;
                    // FIXME
                    // goto the particle collector.
                    goto_particle_collector(xp_push, up_push, rhop_push, i, j,
                                               xp0_push, up0_push);

                    //particle_communicator_collector(xp_push, up_push, rhop_push, i, j);
                    // particle_erase is built-in
                }else{
                    std::cout<<"Ariving here is dangerous"<<std::endl;
                    // FIXME
                    _P->_cell2part[i].erase(_P->_cell2part[i].begin() + j);
                    // TODO: also erase dup0, drhop0, now it might give troubles in serial!
                    //
                }
                // Decrement particle iterator (when erased)
                j--;
            }
        }
    }

    // Relocate local
    relocate_local_pusher(reloc_local_c, reloc_local_p, reloc_local_xp0, reloc_local_up0);

    // Relocate among processes: note, if not found, particle is silently destroyed
    // MPI relocate
    if(num_processes > 1) goto_particle_pusher(xp_push, up_push, rhop_push, xp0_push, up0_push);


    // Also, reset and set cell2facet and
    cell2facet.clear();
    facets_info.clear();
    cell2facet.resize(_P->_mesh->num_cells());
    set_facets_info();
}
*/
//-----------------------------------------------------------------------------
void advect_particles::do_step(double dt){
  const MPI_Comm mpi_comm = _P->mesh()->mpi_comm();
  const std::size_t gdim = _P->mesh()->geometry().dim();

    std::size_t num_processes = MPI::size(mpi_comm);

    // Needed for local reloc
    std::vector<std::size_t> reloc_local_c;
    std::vector<particle>    reloc_local_p;

    std::vector<std::vector<particle>> comm_snd(num_processes);

    for( CellIterator ci(*(_P->mesh())); !ci.end(); ++ci){
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
            Eigen::VectorXd u_p =  basis_mat * exp_coeffs ;

            // Convert velocity to point
            Point up(gdim, u_p.data() );

            std::size_t cidx_recv = ci->index();
            double dt_rem = dt;

            while(dt_rem > 1E-15){
                // Returns facet which is intersected and the time it takes to do so
                std::tuple<std::size_t, double> intersect_info = time2intersect(cidx_recv, dt_rem,
                                                                                _P->x(ci->index(), i), up);
                if(std::get<0>(intersect_info) == std::numeric_limits<std::size_t>::max() ){
                    // Then remain within cell, finish time step
                    _P->push_particle(dt_rem, up, ci->index(), i);
                    dt_rem *= 0.;
                    // TODO: if step == last tstep: update particle position old to most recent value

                    // If cidx_recv != ci->index(), particles crossed facet and hence need to be relocated
                    if(cidx_recv != ci->index() ){
                        reloc_local_c.push_back(cidx_recv);
                        // FIXME: remove
                        //particle p_temp = _P->_cell2part[ci->index()][i];
                        reloc_local_p.push_back(_P->_cell2part[ci->index()][i]);
                        _P->_cell2part[ci->index()].erase(_P->_cell2part[ci->index()].begin() + i);
                        i--; //Decrement iterator
                    }
                }else{
                    // Two options: if internal (==2) else if boundary
                    if( std::get<3>(facets_info[std::get<0>(intersect_info)]).size() == 2 ){
                        // Then we cross facet which has a neighboring cell
                        _P->push_particle(std::get<1>(intersect_info), up, ci->index(), i);

                        // Update index of receiving cell
                        for(std::size_t cidx_iter :
                            std::get<3>(facets_info[std::get<0>(intersect_info)])){
                            if( cidx_iter != cidx_recv ){
                                cidx_recv = cidx_iter;
                                break;
                            }
                        }
                        // Update remaining time
                        dt_rem -= std::get<1>(intersect_info);
                        if(dt_rem < 1E-15){
                            // Then terminate
                            dt_rem *= 0.;
                            if(cidx_recv != ci->index() ){
                                reloc_local_c.push_back(cidx_recv);
                                reloc_local_p.push_back(_P->_cell2part[ci->index()][i]);
                                _P->_cell2part[ci->index()].erase(_P->_cell2part[ci->index()].begin() + i);
                                i--; // Decrement iterator
                            }
                        }
                    }else if(std::get<3>(facets_info[std::get<0>(intersect_info)]).size() == 1){
                        // Then we hit a boundary, but which type?
                        if(std::find(int_facets.begin(), int_facets.end(),  std::get<0>(intersect_info)) != int_facets.end()){
                            // Then it is an internal boundary
                            // Do a full push
                            _P->push_particle(dt_rem, up, ci->index(), i);
                            dt_rem *= 0.;

                            if(pbc_active) pbc_limits_violation(ci->index(),i); // Check on sequence crossing internal bc -> crossing periodic bc
                            // TODO: do same for closed bcs to handle (unlikely event): internal bc-> closed bc

                            // Go to the particle communicator
                            _P->particle_communicator_collect(comm_snd, ci->index(), i);
                            i--;
                        }else if(std::find(obc_facets.begin(), obc_facets.end(),  std::get<0>(intersect_info)) != obc_facets.end()){
                            // Particle leaves the domain. Simply erase!
                            // FIXME: additional check that particle indeed leaves domain (u\cdotn > 0)
                            apply_open_bc(ci->index(), i);
                            dt_rem *= 0.;
                            i--;
                        }else if(std::find(cbc_facets.begin(), cbc_facets.end(),  std::get<0>(intersect_info)) != cbc_facets.end()){
                            // Closed BC
                            apply_closed_bc(std::get<1>(intersect_info), up, ci->index(), i, std::get<0>(intersect_info) );
                            dt_rem -= std::get<1>(intersect_info);
                        }else if(std::find(pbc_facets.begin(), pbc_facets.end(),  std::get<0>(intersect_info)) != pbc_facets.end()){
                            // Then periodic bc
                            apply_periodic_bc(dt_rem, up, ci->index(), i, std::get<0>(intersect_info) );
                            if (num_processes > 1){ //Behavior in parallel
                                _P->particle_communicator_collect(comm_snd, ci->index(), i);
                            }else{
                                // Behavior in serial
                              std::size_t cell_id = _P->mesh()->bounding_box_tree()->compute_first_entity_collision( _P->x(ci->index(), i));
                                reloc_local_c.push_back(cell_id);
                                reloc_local_p.push_back(_P->_cell2part[ci->index()][i]);
                                _P->_cell2part[ci->index()].erase(_P->_cell2part[ci->index()].begin() + i);
                            }
                            dt_rem *= 0.;
                            i--;
                        }else{
                            dolfin_error("advect_particles.cpp::do_step","encountered unknown boundary", "Only internal boundaries implemented yet");
                        }
                    }else{
                        dolfin_error("advect_particles.cpp::do_step","found incorrect number of facets (<1 or > 2)", "Unknown");
                    }
                } // end else
            } // end while
        } // end for
    } // end for

    // Relocate local
    for(std::size_t i = 0; i<reloc_local_c.size(); i++){
        if(reloc_local_c[i] != std::numeric_limits<unsigned int>::max() ){
            _P->_cell2part[reloc_local_c[i]].push_back( reloc_local_p[i] );
        }else{
            dolfin_error("advection.cpp::do_step", "find a hosting cell on local process", "Unknown");
        }
    }

    // Debug only
    /*
    for (std::size_t p = 0; p < num_processes; p++){
        std::cout<<"Size of comm_snd at process "<<p<<" "<<comm_snd[p].size()<<std::endl;
    }
    */

    // Relocate global
    if(num_processes > 1) _P->particle_communicator_push(comm_snd);
}
//-----------------------------------------------------------------------------
/*
void advect_particles::update_particle_property_pic(const Function& phih, const std::size_t idx)
{
    // Update particle properties using PIC-method

    // Update particle property given by idx
    if(idx != 1 && idx != 2){
        warning("advect_particles.cpp::update_particle_property, property index must be 1 or 2");
        return;
    }

    std::size_t space_dimension_phih, value_size_loc_phih;
    space_dimension_phih = phih.function_space()->element()->space_dimension();
    value_size_loc_phih  = 1;
    for (std::size_t i = 0; i < phih.function_space()->element()->value_rank(); i++)
      value_size_loc_phih *= phih.function_space()->element()->value_dimension(i);

    if(phih.function_space()->element()->value_rank() != 0 && idx == 2 ){
        dolfin_error("advect_particles.cpp::update_particle_property", "update particle property",
                     "Mismatch between function space and particle property to update: expecting scalar");
    }else if (phih.function_space()->element()->value_rank() != 1 && idx == 1){
        dolfin_error("advect_particles.cpp::update_particle_property", "update particle property",
                     "Mismatch between function space and particle property to update: expecting vector");
    }

    for( CellIterator cell(*(_P->_mesh)); !cell.end(); ++cell){
        std::vector<double> coeffs;
        return_expansion_coeffs(coeffs, *cell, phih);
        for(int i = 0; i < _P->_cell2part[cell->index()].size() ; i++)
        {

            std::vector<double> basis_matrix(space_dimension_phih * value_size_loc_phih);
            return_basis_matrix(basis_matrix, std::get<0>(_P->_cell2part[cell->index()][i]),*cell,
                                phih.function_space()->element());

            Eigen::Map<Eigen::MatrixXd> basis_mat(basis_matrix.data(), value_size_loc_phih, space_dimension_phih);
            Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), space_dimension_phih);
            Eigen::VectorXd phi_p =  basis_mat * exp_coeffs ;

            // So far is generic, now distinguish betwen methods
            if(idx == 1){
                vec2d phip_vec(phi_p[0], phi_p[1]);
                std::get<1>(_P->_cell2part[cell->index()][i])  = phip_vec;
            }else if(idx == 2){
                std::get<2>(_P->_cell2part[cell->index()][i]) = phi_p[0];
            }
        }
    }
}
*/
/*
void advect_particles::update_particle_property_flip(const Function& phih_new, const Function& phih_old,
                                                     const std::size_t idx, const double theta, const std::size_t stepnum)
{
    // Update particle properties using FLIP-method

    // FIXME: check if function spaces are equal
    //if (phih_new.function_space() != phih_old.function_space() )
    //    dolfin_error("advect_particles.cpp::update_particle_property", "update particle property",
    //                 "phih_new and phih_old must be from same function space");

    // Update particle property given by 1 <= idx <=2
    if(idx != 1 && idx != 2){
        warning("advect_particles.cpp::update_particle_property, property index must be 1 or 2");
        return;
    }

    // Check if particle properties are tracked correctly
    if(idx == 1 && (update_particle != "vector" && update_particle != "both") ){
         dolfin_error("advect_particles.cpp::update_particle_property", "update particle property",
                       "Particle property not tracked.");
    }else if(idx == 2 && (update_particle != "scalar" && update_particle != "both") ){
        dolfin_error("advect_particles.cpp::update_particle_property", "update particle property",
                      "Particle property not tracked.");
    }

    // Check if function space (new) matches with particle update
    if(phih_new.function_space()->element()->value_rank() != 0 && idx == 2 ){
        dolfin_error("advect_particles.cpp::update_particle_property", "update particle property",
                     "Mismatch between function space and particle property to update: expecting scalar");
    }else if (phih_new.function_space()->element()->value_rank() != 1 && idx == 1){
        dolfin_error("advect_particles.cpp::update_particle_property", "update particle property",
                     "Mismatch between function space and particle property to update: expecting vector");
    }


    std::size_t space_dimension_phih, value_size_loc_phih;
    space_dimension_phih = phih_new.function_space()->element()->space_dimension();
    value_size_loc_phih  = 1;
    for (std::size_t i = 0; i < phih_new.function_space()->element()->value_rank(); i++)
      value_size_loc_phih *= phih_new.function_space()->element()->value_dimension(i);

    // Update particle property
    for( CellIterator cell(*(_P->_mesh)); !cell.end(); ++cell){
        // Get expansion coefficients
        std::vector<double> coeffs_new, coeffs_old;
        return_expansion_coeffs(coeffs_new, *cell, phih_new);
        return_expansion_coeffs(coeffs_old, *cell, phih_old);

        for(int i = 0; i < _P->_cell2part[cell->index()].size() ; i++)
        {
            std::vector<double> basis_matrix(space_dimension_phih * value_size_loc_phih);
            return_basis_matrix(basis_matrix, std::get<0>(_P->_cell2part[cell->index()][i]),*cell,
                                phih_new.function_space()->element());

            Eigen::Map<Eigen::MatrixXd> basis_mat(basis_matrix.data(), value_size_loc_phih, space_dimension_phih);
            Eigen::Map<Eigen::VectorXd> exp_coeffs_new(coeffs_new.data(), space_dimension_phih);
            Eigen::Map<Eigen::VectorXd> exp_coeffs_old(coeffs_old.data(), space_dimension_phih);
            Eigen::VectorXd delta_phi_p =  basis_mat * (exp_coeffs_new - exp_coeffs_old); //exp_coeffs ;

            // So far generic, now distinguish betwen vector or scalar
            if(idx == 1){
                // Vector valued case
                vec2d delta_phip_vec(delta_phi_p[0], delta_phi_p[1]);
                if(stepnum > 0){
                    std::get<1>(_P->_cell2part[cell->index()][i]) += theta * delta_phip_vec + (1. - theta) * dup[cell->index()][i];
                }else if (stepnum == 0){
                    std::get<1>(_P->_cell2part[cell->index()][i]) += delta_phip_vec;
                }else{
                    dolfin_error("advect_particles::update_particle_property","find correct stepnumber", "Stepnumber must be >=0");
                }
                // Finally overwrite particle value
                dup[cell->index()][i] = delta_phip_vec;
            }else if(idx == 2){
                // Scalar valued case
                if(stepnum > 0){
                    std::get<2>(_P->_cell2part[cell->index()][i]) += theta * delta_phi_p[0] + (1. - theta) * drhop[cell->index()][i];
                }else if(stepnum == 0){
                    std::get<2>(_P->_cell2part[cell->index()][i]) += delta_phi_p[0];
                }else{
                    dolfin_error("advect_particles::update_particle_property","find correct stepnumber", "Stepnumber must be >=0");
                }
                drhop[cell->index()][i] = delta_phi_p[0];
            }
        }
    }
}

/*
 * NOTE: following particle updater limited to FLIP with two steps only!
 */

/*
void advect_particles::update_particle_property(const Function& phih, const std::size_t stepnum){
    std::size_t space_dimension_phih, value_size_loc_phih;
    space_dimension_phih = phih.function_space()->element()->space_dimension();
    value_size_loc_phih  = 1;
    for (std::size_t i = 0; i < phih.function_space()->element()->value_rank(); i++)
      value_size_loc_phih *= phih.function_space()->element()->value_dimension(i);

    if(phih.function_space()->element()->value_rank() != 0 && update_particle == "scalar" ){
        dolfin_error("advect_particles.cpp::update_particle_property", "update particle property",
                     "Mismatch between function space and particle property to update");
    }else if (phih.function_space()->element()->value_rank() != 1 && update_particle == "vector"){
        dolfin_error("advect_particles.cpp::update_particle_property", "update particle property",
                     "Mismatch between function space and particle property to update");
    }

    if(update_particle == "scalar" || update_particle == "both"){
        warning("advect_particles::update_particle_property, cannot update scalar or scalar and vector quantities right now");
        //return;
    }

    for( CellIterator cell(*(_P->_mesh)); !cell.end(); ++cell){
        std::vector<double> coeffs;
        return_expansion_coeffs(coeffs, *cell, phih);
        for(int i = 0; i < _P->_cell2part[cell->index()].size() ; i++)
        {
            std::vector<double> basis_matrix(space_dimension_phih * value_size_loc_phih);
            // Replaced by customized return_basis_matrix
            //return_basis_matrix(basis_matrix, std::get<0>(_P->_cell2part[cell->index()][i]),*cell);
            return_basis_matrix(basis_matrix, std::get<0>(_P->_cell2part[cell->index()][i]),*cell,
                                phih.function_space()->element());

            Eigen::Map<Eigen::MatrixXd> basis_mat(basis_matrix.data(), value_size_loc_phih, space_dimension_phih);
            Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), space_dimension_phih);

            Eigen::VectorXd phi_p =  basis_mat * exp_coeffs ;

            // Then update particle properties (right now: FLIP)
            vec2d phip_vec(phi_p[0], phi_p[1]);
            vec2d dphip_vec = phip_vec - std::get<1>(_P->_cell2part[cell->index()][i]);
            if(stepnum > 0){
                std::get<1>(_P->_cell2part[cell->index()][i]) += 0.5*( dphip_vec + dup[cell->index()][i]);
            }else if (stepnum == 0){
                std::get<1>(_P->_cell2part[cell->index()][i]) += dphip_vec;
            }else{
                dolfin_error("advect_particles::update_particle_property","find correct stepnumber", "Stepnumber must be >=0");
            }
            // Finally overwrit dup
            dup[cell->index()][i] = dphip_vec;
        }
    }
}
//-----------------------------------------------------------------------------
/*
 * Just for testing purposes only
void advect_particles::update_particle_property(const Function& phih, const std::string update_method,
                                                const std::size_t stepnum){
    // TODO: Work in Progress...!
    // TODO: avoid (expensive?) point evaluation by reusing old information --> from project

    std::size_t space_dimension_phih, value_size_loc_phih;
    space_dimension_phih = phih.function_space()->element()->space_dimension();
    value_size_loc_phih  = 1;
    for (std::size_t i = 0; i < phih.function_space()->element()->value_rank(); i++)
      value_size_loc_phih *= phih.function_space()->element()->value_dimension(i);

    if(phih.function_space()->element()->value_rank() != 0 && update_particle == "scalar" ){
        dolfin_error("advect_particles.cpp::update_particle_property", "update particle property",
                     "Mismatch between function space and particle property to update");
    }else if (phih.function_space()->element()->value_rank() != 1 && update_particle == "vector"){
        dolfin_error("advect_particles.cpp::update_particle_property", "update particle property",
                     "Mismatch between function space and particle property to update");
    }

    std::size_t pproperty_rank = phih.function_space()->element()->value_rank();
    std::size_t update_method_int;
    if(update_method == "PIC"){
        update_method_int = 0;
    }else if(update_method == "PIC2"){
        update_method_int = 1;
    }else if(update_method == "FLIP"){
        update_method_int = 2;
    }else if(update_method == "FLIP2"){
        update_method_int = 3;
    }else{
        warning("Unknown particle updater type, choose PIC/PIC2/FLIP/FLIP2. Now ignoring update");
        return;
    }

    /*
    for( CellIterator cell(*(_P->_mesh)); !cell.end(); ++cell){
        std::vector<double> coeffs;
        return_expansion_coeffs(coeffs, *cell, phih);
        for(int i = 0; i < _P->_cell2part[cell->index()].size() ; i++)
        {
            std::vector<double> basis_matrix(space_dimension_phih * value_size_loc_phih);
            return_basis_matrix(basis_matrix, std::get<0>(_P->_cell2part[cell->index()][i]),*cell);

            Eigen::Map<Eigen::MatrixXd> basis_mat(basis_matrix.data(), value_size_loc_phih, space_dimension_phih);
            Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), space_dimension_phih);
            Eigen::VectorXd phi_p =  basis_mat * exp_coeffs ;

            // Then update particle properties (right now: FLIP)
            vec2d phip_vec(phi_p[0], phi_p[1]);
            vec2d dphip_vec = phip_vec - std::get<1>(_P->_cell2part[cell->index()][i]);
            if(stepnum > 0){
                std::get<1>(_P->_cell2part[cell->index()][i]) += 0.5*( dphip_vec + dup[cell->index()][i]);
            }else if (stepnum == 0){
                std::get<1>(_P->_cell2part[cell->index()][i]) += dphip_vec;
            }else{
                dolfin_error("advect_particles::update_particle_property","find correct stepnumber", "Stepnumber must be >=0");
            }
            // Finally overwrit dup
            dup[cell->index()][i] = dphip_vec;
        }
    }

    std::cout<<"Update method "<<update_method_int<<std::endl;
    std::cout<<"Pproperty rank "<<pproperty_rank<<std::endl;
    std::cout<<"Size"<<space_dimension_phih * value_size_loc_phih<<std::endl;

    for( CellIterator cell(*(_P->_mesh)); !cell.end(); ++cell){
        std::vector<double> coeffs;
        return_expansion_coeffs(coeffs, *cell, phih);
        for(int i = 0; i < _P->_cell2part[cell->index()].size() ; i++)
        {

            std::vector<double> basis_matrix(space_dimension_phih * value_size_loc_phih);
            return_basis_matrix(basis_matrix, std::get<0>(_P->_cell2part[cell->index()][i]),*cell,
                                phih.function_space()->element());

            Eigen::Map<Eigen::MatrixXd> basis_mat(basis_matrix.data(), value_size_loc_phih, space_dimension_phih);
            Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), space_dimension_phih);
            Eigen::VectorXd phi_p =  basis_mat * exp_coeffs ;

            // Then update particle properties (right now: PIC)

            if(update_method_int == 0){
                if(pproperty_rank == 0){
                    //double phip_vec = phi_p[0];
                    //std::cout<<"Particle property"<<phi_p<<std::endl;
                    //std::cout<<"Value to be used"<<phi_p[0]<<std::endl;
                    /*
                    if(cell->index() == 10){
                        std::cout<<"Basis matrix \n"<<basis_mat<<std::endl;
                        std::cout<<"Expansion coefficients \n"<<exp_coeffs<<std::endl;
                    }


                    std::get<2>(_P->_cell2part[cell->index()][i]) = phi_p[0];
                }else if(pproperty_rank == 1){
                    warning("PIC for vector not yet implemented, return without updating");
                    return;
                }
            }else if(update_method_int == 1){
                warning("PIC second order not yet implemented");
                return;
            }else if(update_method_int == 2){
                warning("FLIP first order not yet implemented");
                return;
            }else if(update_method_int == 3){
                if(pproperty_rank == 0){
                    warning("FLIP2 for scalar not yet implemented, return without update");
                    return;
                }else if(pproperty_rank == 1){
                    vec2d phip_vec(phi_p[0], phi_p[1]);
                    vec2d dphip_vec = phip_vec - std::get<1>(_P->_cell2part[cell->index()][i]);

                    if(stepnum > 0){
                        std::get<1>(_P->_cell2part[cell->index()][i]) += 0.5*( dphip_vec + dup[cell->index()][i]);
                    }else if (stepnum == 0){
                        std::get<1>(_P->_cell2part[cell->index()][i]) += dphip_vec;
                    }else{
                        dolfin_error("advect_particles::update_particle_property","find correct stepnumber", "Stepnumber must be >=0");
                    }
                    // Finally overwrit dup
                    dup[cell->index()][i] = dphip_vec;
                }
            }
        }
    }
}
*/
//-----------------------------------------------------------------------------
// FIXME: migrated to particle class
//void advect_particles::push_particle(const double dt, const Point& up, const std::size_t cidx, const std::size_t pidx){
//    _P->_cell2part[cidx][pidx][0] += up*dt;
//}
//-----------------------------------------------------------------------------
std::tuple<std::size_t, double> advect_particles::time2intersect(std::size_t cidx, double dt, const Point xp, const Point up){
    // Time to facet intersection
    double dt_int = std::numeric_limits<double>::max();
    std::size_t target_facet = std::numeric_limits<std::size_t>::max() ;

    for(std::size_t fidx : cell2facet[cidx]){
        Point normal = std::get<2>(facets_info[fidx]);

        // Make sure normal is outward pointing
        std::vector<std::size_t>::iterator it;
        it = std::find(std::get<3>(facets_info[fidx]).begin(), std::get<3>(facets_info[fidx]).end(),
                       cidx);
        std::size_t it_idx = std::distance( std::get<3>(facets_info[fidx]).begin(), it );
        if(it_idx == std::get<3>(facets_info[fidx]).size() ){
            dolfin_error("advect_particles.cpp::time2intersect","did not find matching facet cell connectivity",
                                      "Unknown");
        }else{
            bool outward_normal = std::get<4>(facets_info[fidx])[it_idx];
            if(!outward_normal) normal *= -1.;
        }

        // Compute distance to point. For procedure, see Haworth (2010). Though it is slightly modified
        double h = (std::get<0>(facets_info[fidx])).distance(xp);

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
    _P->_cell2part[cidx].erase(_P->_cell2part[cidx].begin() + pidx);
}
//-----------------------------------------------------------------------------
void advect_particles::apply_closed_bc(double dt, Point& up, std::size_t cidx, std::size_t pidx, std::size_t fidx ){
    // First push particle
    _P->push_particle(dt, up, cidx, pidx);
    // Mirror velocity
    Point normal = std::get<2>(facets_info[fidx]);
    up -= 2*(up.dot(normal))*normal;
}
//-----------------------------------------------------------------------------
void advect_particles::apply_periodic_bc(double dt, Point& up, std::size_t cidx,  std::size_t pidx, std::size_t fidx){
  const std::size_t gdim = _P->mesh()->geometry().dim();
    Point midpoint = std::get<1>(facets_info[fidx]);
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
    _P->_cell2part[cidx][pidx][0][component] +=
                (pbc_lims[row_friend][ component*2 ] - pbc_lims[row_match][ component*2 ]);

    // Corners can be tricky, therefore include this test
    for(std::size_t i = 0; i < gdim; i++){
        if( i == component ) continue; // Skip this
        if( _P->x(cidx, pidx)[i] < pbc_lims[row_match][ i*2 ] ){
            // Then we push the particle to the other end of domain
            _P->_cell2part[cidx][pidx][0][i] +=
                (pbc_lims[row_friend][ i*2 + 1 ] - pbc_lims[row_match][ i*2 ]);
        }else if( _P->x(cidx, pidx)[i] > pbc_lims[row_match][ i*2 + 1] ){
            _P->_cell2part[cidx][pidx][0][i] -=
                (pbc_lims[row_match][ i*2 + 1 ] - pbc_lims[row_friend][ i*2 ]);
        }
    }
}
//-----------------------------------------------------------------------------
void advect_particles::pbc_limits_violation(std::size_t cidx, std::size_t pidx){
    // This method guarantees that particles can cross internal bc -> periodic bc in one
    // time step without being deleted.
    // FIXME: more efficient implementation??
    // FIXME: can give troubles when domain decomposition results in one cell in domain corner
    // Check if periodic bcs are violated somewhere, if so, modify particle position
  std::size_t gdim = _P->mesh()->geometry().dim();

  for(std::size_t i = 0; i < pbc_lims.size()/2; i++){
        for(std::size_t j = 0; j< gdim; j++){
            if( std::abs(pbc_lims[2*i][2*j] - pbc_lims[2*i][2*j+1]) < 1E-13 ){
              if( _P->x(cidx, pidx)[j]  >  pbc_lims[2*i][2*j]
                        &&
                  _P->x(cidx, pidx)[j]  >  pbc_lims[2*i+1][2*j]   )
                {
                    _P->_cell2part[cidx][pidx][0][j] -=
                            (std::max(pbc_lims[2*i][2*j], pbc_lims[2*i+1][2*j]) -
                             std::min(pbc_lims[2*i][2*j], pbc_lims[2*i+1][2*j]) );
                    // Check whether the other bounds are violated, to handle corners
                    // FIXME: cannot handle cases where domain of friend in one direction is different from
                    // match, reason: looping over periodic bc pairs
                    for(std::size_t k = 0; k < gdim; k++){
                        if( k == j ) continue;
                        if( _P->x(cidx, pidx)[k] < pbc_lims[2*i][2*k]  ){
                            _P->_cell2part[cidx][pidx][0][k]  +=
                                 ( pbc_lims[2*i + 1][2*k + 1]  - pbc_lims[2*i][2*k]);
                        }else if( _P->x(cidx, pidx)[k] > pbc_lims[2*i][2*k + 1] ){
                            _P->_cell2part[cidx][pidx][0][k]  -=
                                 ( pbc_lims[2*i][2*k + 1] - pbc_lims[2*i + 1][2*k]);
                        }
                    }
                }else if(_P->x(cidx, pidx)[j]  <  pbc_lims[2*i][2*j]
                            &&
                         _P->x(cidx, pidx)[j]  <  pbc_lims[2*i+1][2*j]
                         )
                {
                    _P->_cell2part[cidx][pidx][0][j] +=
                            (std::max(pbc_lims[2*i][2*j], pbc_lims[2*i+1][2*j]) -
                             std::min(pbc_lims[2*i][2*j], pbc_lims[2*i+1][2*j]) );
                    // Check wheter the other bounds are violated, to handle corners
                    for(std::size_t k = 0; k < gdim; k++){
                        if( k == j ) continue;
                        if( _P->x(cidx, pidx)[k] < pbc_lims[2*i][2*k]  ){
                            _P->_cell2part[cidx][pidx][0][k]  +=
                                 ( pbc_lims[2*i + 1][2*k + 1]  - pbc_lims[2*i][2*k]);
                        }else if( _P->x(cidx, pidx)[k] > pbc_lims[2*i][2*k + 1] ){
                            _P->_cell2part[cidx][pidx][0][k]  -=
                                 ( pbc_lims[2*i][2*k + 1] - pbc_lims[2*i + 1][2*k]);
                        }
                    }
                } // else do nothing
            }
        }
    }
}
//-----------------------------------------------------------------------------
/* TODO: REMOVE OLD 2D FORMULATION
//-----------------------------------------------------------------------------
void advect_particles::apply_periodic_bc(double dt, Point& up, std::size_t cidx,  std::size_t pidx, std::size_t fidx){
    Point midpoint = std::get<1>(facets_info[fidx]);
    std::size_t row_match = std::numeric_limits<std::size_t>::max();
    std::size_t row_friend;
    std::size_t component;
    bool hit = false;
    for(std::size_t i = 0; i < pbc_lims.size(); i++){
        for(std::size_t j = 0; j < _P->_Ndim; j++){
            if( std::abs(midpoint[j] - pbc_lims[i][ j*_P->_Ndim ]) < 1E-10
                    &&
                std::abs(midpoint[j] - pbc_lims[i][ j * _P->_Ndim + 1 ]) < 1E-10 ){
                // Then we most likely found a match, but check if midpoint coordinates are in between specified limits
                hit = true;
                for(std::size_t k = 0; k < _P->_Ndim; k++){
                    if( k == j ) continue;
                    if( midpoint[k] <= pbc_lims[i][k * _P->_Ndim]  || midpoint[k] >= pbc_lims[i][k * _P->_Ndim + 1] ){
                        hit = false;
                    }
                }
                if(hit){
                    row_match = i ;
                    component = j;
                    // Break out of multiple loops
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
    _P->_cell2part[cidx][pidx][0][component] +=
                (pbc_lims[row_friend][component * _P->_Ndim] - pbc_lims[row_match][component * _P->_Ndim]);

    // Corners can be tricky, therefore include this test
    for(std::size_t i = 0; i < _P->_Ndim; i++){
        if( i == component ) continue; // Skip this
        if( _P->_cell2part[cidx][pidx][0][i] < pbc_lims[row_match][i * _P->_Ndim] ){
            // Then we push the particle to the other end of domain
            _P->_cell2part[cidx][pidx][0][i] +=
                (pbc_lims[row_friend][i * _P->_Ndim + 1 ] - pbc_lims[row_match][i * _P->_Ndim]);
        }else if( _P->_cell2part[cidx][pidx][0][i] > pbc_lims[row_match][i * _P->_Ndim + 1] ){
            _P->_cell2part[cidx][pidx][0][i] -=
                (pbc_lims[row_match][i * _P->_Ndim + 1 ] - pbc_lims[row_friend][i * _P->_Ndim ]);
        }
    }
}
//-----------------------------------------------------------------------------
void advect_particles::pbc_limits_violation(std::size_t cidx, std::size_t pidx){
    // This method guarantees that particles can cross internal bc -> periodic bc in one
    // time step without being deleted.
    // FIXME: more efficient implementation??
    // FIXME: can give troubles when domain decomposition results in one cell in domain corner
    // Check if periodic bcs are violated somewhere, if so, modify particle position
    for(std::size_t i = 0; i < pbc_lims.size()/2; i++){
        for(std::size_t j = 0; j<_P->_Ndim; j++){
            if( std::abs(pbc_lims[2*i][2*j] - pbc_lims[2*i][2*j+1]) < 1E-13 ){
                if( _P->_cell2part[cidx][pidx][0][j]  >  pbc_lims[2*i][2*j]
                        &&
                    _P->_cell2part[cidx][pidx][0][j]  >  pbc_lims[2*i+1][2*j]   )
                {
                    _P->_cell2part[cidx][pidx][0][j] -=
                            (std::max(pbc_lims[2*i][2*j], pbc_lims[2*i+1][2*j]) -
                             std::min(pbc_lims[2*i][2*j], pbc_lims[2*i+1][2*j]) );
                    // Check whether the other bounds are violated, to handle corners
                    // FIXME: cannot handle cases where domain of friend in one direction is different from
                    // match, reason: looping over periodic bc pairs
                    for(std::size_t k = 0; k<_P->_Ndim; k++){
                        if( k == j ) continue;
                        if( _P->_cell2part[cidx][pidx][0][k] < pbc_lims[2*i][2*k]  ){
                            _P->_cell2part[cidx][pidx][0][k]  +=
                                 ( pbc_lims[2*i + 1][2*k + 1]  - pbc_lims[2*i][2*k]);
                        }else if( _P->_cell2part[cidx][pidx][0][k] > pbc_lims[2*i][2*k + 1] ){
                            _P->_cell2part[cidx][pidx][0][k]  -=
                                 ( pbc_lims[2*i][2*k + 1] - pbc_lims[2*i + 1][2*k]);
                        }
                    }
                }else if(_P->_cell2part[cidx][pidx][0][j]  <  pbc_lims[2*i][2*j]
                            &&
                         _P->_cell2part[cidx][pidx][0][j]  <  pbc_lims[2*i+1][2*j]
                         )
                {
                    _P->_cell2part[cidx][pidx][0][j] +=
                            (std::max(pbc_lims[2*i][2*j], pbc_lims[2*i+1][2*j]) -
                             std::min(pbc_lims[2*i][2*j], pbc_lims[2*i+1][2*j]) );
                    // Check wheter the other bounds are violated, to handle corners
                    for(std::size_t k = 0; k<_P->_Ndim; k++){
                        if( k == j ) continue;
                        if( _P->_cell2part[cidx][pidx][0][k] < pbc_lims[2*i][2*k]  ){
                            _P->_cell2part[cidx][pidx][0][k]  +=
                                 ( pbc_lims[2*i + 1][2*k + 1]  - pbc_lims[2*i][2*k]);
                        }else if( _P->_cell2part[cidx][pidx][0][k] > pbc_lims[2*i][2*k + 1] ){
                            _P->_cell2part[cidx][pidx][0][k]  -=
                                 ( pbc_lims[2*i][2*k + 1] - pbc_lims[2*i + 1][2*k]);
                        }
                    }
                } // else do nothing
            }
        }
    }
}
*/
//-----------------------------------------------------------------------------
void advect_particles::do_substep(double dt, Point& up, const std::size_t cidx, std::size_t* pidx,
                                  const std::size_t step, const std::size_t num_steps,
                                  const std::size_t xp0_idx, const std::size_t up0_idx,
                                  std::vector<std::size_t>& reloc_local_c, std::vector<particle>& reloc_local_p,
                                  std::vector<std::vector<particle>>& comm_snd){
    double dt_rem = dt;
    //std::size_t cidx_recv = cidx;

    const std::size_t mpi_size = MPI::size(_P->mesh()->mpi_comm());
    std::size_t cidx_recv = std::numeric_limits<std::size_t>::max();

    if(step == 0 ){
        cidx_recv = cidx;
    }else{
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
        if(cidx_recv == std::numeric_limits<unsigned int>::max()){
            _P->push_particle(dt_rem, up, cidx, *pidx);
            if(pbc_active) pbc_limits_violation(cidx,*pidx);
            if( step == (num_steps - 1) ){
                // Copy current position to old position
                // so something like
                _P->_cell2part[cidx][*pidx][xp0_idx] = _P->_cell2part[cidx][*pidx][0];
            }
            // Apparently, this allways lead to a communicate, but why?
            _P->particle_communicator_collect(comm_snd, cidx, *pidx);
            (*pidx)--;
            return; // Stop right here
        }
    }

    bool hit_cbc = false; // Hit closed boundary condition (?!)
    while(dt_rem > 1E-15){
        // Returns facet which is intersected and the time it takes to do so
        std::tuple<std::size_t, double> intersect_info = time2intersect(cidx_recv, dt_rem,
                                                                        _P->x(cidx, *pidx), up);
        if(std::get<0>(intersect_info) == std::numeric_limits<std::size_t>::max() ){
            // Then remain within cell, finish time step
            _P->push_particle(dt_rem, up, cidx, *pidx);
            dt_rem *= 0.;

            if( step == (num_steps - 1) )
                // Copy current position to old position
              _P->_cell2part[cidx][*pidx][xp0_idx] = _P->x(cidx, *pidx);

            // If cidx_recv != ci->index(), particles crossed facet and hence need to be relocated
            if(cidx_recv != cidx ){
                // Then relocate local
                reloc_local_c.push_back(cidx_recv);
                reloc_local_p.push_back(_P->_cell2part[cidx][*pidx]);
                _P->_cell2part[cidx].erase(_P->_cell2part[cidx].begin() + *pidx);
                (*pidx)--; //Decrement iterator
            }
        }else{
            // Two options: if internal (==2) else if boundary
            if( std::get<3>(facets_info[std::get<0>(intersect_info)]).size() == 2 ){
                // Then we cross facet which has a neighboring cell
                _P->push_particle(std::get<1>(intersect_info), up, cidx, *pidx);

                // Update index of receiving cell
                for(std::size_t cidx_iter :
                    std::get<3>(facets_info[std::get<0>(intersect_info)])){
                    if( cidx_iter != cidx_recv ){
                        cidx_recv = cidx_iter;
                        break;
                    }
                }
                // Update remaining time
                dt_rem -= std::get<1>(intersect_info);
                if(dt_rem < 1E-15){
                    // Then terminate
                    dt_rem *= 0.;
                    // Copy current position to old position
                    if( step == (num_steps - 1) )
                      _P->_cell2part[cidx][*pidx][xp0_idx] = _P->x(cidx, *pidx);

                    if(cidx_recv != cidx ){
                        reloc_local_c.push_back(cidx_recv);
                        reloc_local_p.push_back(_P->_cell2part[cidx][*pidx]);
                        _P->_cell2part[cidx].erase(_P->_cell2part[cidx].begin() + *pidx);
                        (*pidx)--; //Decrement iterator
                    }
                }
            }else if(std::get<3>(facets_info[std::get<0>(intersect_info)]).size() == 1){
                // Then we hit a boundary, but which type?
                if(std::find(int_facets.begin(), int_facets.end(),  std::get<0>(intersect_info)) != int_facets.end()){
                    _P->push_particle(dt_rem, up, cidx, *pidx);
                    dt_rem *= 0.;

                    if(pbc_active) pbc_limits_violation(cidx,*pidx); // Updates particle position if pbc_limits is violated
                    // Copy current position to old position
                    if(step == (num_steps - 1) || hit_cbc)
                        _P->_cell2part[cidx][*pidx][xp0_idx] = _P->_cell2part[cidx][*pidx][0];


                     _P->particle_communicator_collect(comm_snd, cidx, *pidx);
                    (*pidx)--;
                    return; // Stop right here
                }else if(std::find(obc_facets.begin(), obc_facets.end(),  std::get<0>(intersect_info)) != obc_facets.end()){
                    // Particle leaves the domain. Simply erase!
                    apply_open_bc(cidx, *pidx);
                    dt_rem *= 0.;
                    (*pidx)--;
                }else if(std::find(cbc_facets.begin(), cbc_facets.end(),  std::get<0>(intersect_info)) != cbc_facets.end()){
                    apply_closed_bc(std::get<1>(intersect_info), up, cidx, *pidx, std::get<0>(intersect_info) );
                    dt_rem -= std::get<1>(intersect_info);

                    // TODO: CHECK THIS
                    dt_rem += (1. - dti[step]) * (dt/dti[step]); // Make timestep complete
                    // If we hit a closed bc, modify following, probably is first order:

                    // TODO: UPDATE AS PARTICLE!
                    std::vector<double> dummy_vel(_P->_Ndim, std::numeric_limits<double>::max());
                    _P->_cell2part[cidx][*pidx][up0_idx] = Point(_P->_Ndim, dummy_vel.data());
                    hit_cbc = true;
                }else if(std::find(pbc_facets.begin(), pbc_facets.end(),  std::get<0>(intersect_info)) != pbc_facets.end()){
                    //TODO: add support for periodic bcs
                    apply_periodic_bc(dt_rem, up, cidx, *pidx, std::get<0>(intersect_info) );

                    // Copy current position to old position
                    if(step == (num_steps - 1) )
                      _P->_cell2part[cidx][*pidx][xp0_idx] = _P->x(cidx, *pidx);

                    if (mpi_size > 1){ //Behavior in parallel
                        // Allways do a global push
                        _P->particle_communicator_collect(comm_snd, cidx, *pidx);
                    }else{
                        // Behavior in serial
                        // TODO: call particle locate
                      std::size_t cell_id = _P->mesh()->bounding_box_tree()->compute_first_entity_collision(_P->x(cidx, *pidx));

                        reloc_local_c.push_back(cell_id);
                        reloc_local_p.push_back(_P->_cell2part[cidx][*pidx]);
                        _P->_cell2part[cidx].erase(_P->_cell2part[cidx].begin() + *pidx);
                    }
                    dt_rem *= 0.;
                    (*pidx)--;
                }else{
                    dolfin_error("advect_particles.cpp::do_step","encountered unknown boundary", "Only internal boundaries implemented yet");
                }
            }else{
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

    for(std::size_t step = 0; step < num_substeps; step++){
        // Needed for local reloc
        std::vector<std::size_t> reloc_local_c;
        std::vector<particle>    reloc_local_p;
        // Needed for global push
        std::vector<std::vector<particle>> comm_snd(num_processes);

        for( CellIterator ci(*(_P->mesh())); !ci.end(); ++ci){
            if(step == 0 ){ // Restrict once per cell, once per timestep
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
                if(step == 0 ){
                    _P->_cell2part[ci->index()][i][up0_idx] = up;
                }else{
                    // Goto next particle, this particle hitted closed bound
                  if(_P->property(ci->index(), i, up0_idx)[0] == std::numeric_limits<double>::max()) continue;
                  up += _P->property(ci->index(), i, up0_idx);
                  up *= 0.5;
                }

                // Reset position to old
                if(step == 1)
                  _P->_cell2part[ci->index()][i][0] = _P->property(ci->index(), i, xp0_idx);

                // Do substep
                do_substep(dt, up, ci->index(), &i, step, num_substeps,
                           xp0_idx, up0_idx, reloc_local_c, reloc_local_p,
                           comm_snd);
            }
        }

        // Local relocate
        for(std::size_t i = 0; i<reloc_local_c.size(); i++){
            if(reloc_local_c[i] != std::numeric_limits<unsigned int>::max() ){
                _P->_cell2part[reloc_local_c[i]].push_back( reloc_local_p[i] );
            }else{
                dolfin_error("advection_rk2.cpp::do_step", "find a hosting cell on local process", "Unknown");
            }
        }

        // Global relocate
        if(num_processes > 1) _P->particle_communicator_push(comm_snd);
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
        // Needed for global push
        std::vector<std::vector<particle>> comm_snd(num_processes);

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

                Point up(gdim, u_p.data() );

                // Then reset position to the old position
                _P->_cell2part[ci->index()][i][0] = _P->property(ci->index(), i, xp0_idx);

                if(step == 0 ){
                    _P->_cell2part[ci->index()][i][up0_idx] =  up * (weights[step]) ;
                }else if(step == 1){
                    if(_P->_cell2part[ci->index()][i][up0_idx][0] == std::numeric_limits<double>::max()) continue;
                    _P->_cell2part[ci->index()][i][up0_idx] +=  up * (weights[step]) ;
                }else if(step == 2){
                    if(_P->_cell2part[ci->index()][i][up0_idx][0] == std::numeric_limits<double>::max()) continue;
                    up*=weights[step];
                    up+=_P->_cell2part[ci->index()][i][up0_idx];
                }

                // Reset position to old
                if(step == 1)
                  _P->_cell2part[ci->index()][i][0] = _P->property(ci->index(), i, xp0_idx);

                // Do substep
                do_substep(dt*dti[step], up, ci->index(), &i, step, num_substeps,
                           xp0_idx, up0_idx, reloc_local_c, reloc_local_p,
                           comm_snd);
            }
        }

        // Local relocate
        for(std::size_t i = 0; i<reloc_local_c.size(); i++){
            if(reloc_local_c[i] != std::numeric_limits<unsigned int>::max() ){
                _P->_cell2part[reloc_local_c[i]].push_back( reloc_local_p[i] );
            }else{
                dolfin_error("advection_rk2.cpp::do_step", "find a hosting cell on local process", "Unknown");
            }
        }

        // Global relocate
        if(num_processes > 1) _P->particle_communicator_push(comm_snd);
    }
}
//-----------------------------------------------------------------------------
advect_rk3::~advect_rk3(){}
