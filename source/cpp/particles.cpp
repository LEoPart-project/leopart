// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com

#include "particles.h"

using namespace dolfin;

particles::~particles(){
}

particles::particles(Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p_array,
                     const std::vector<unsigned int>& p_template, const Mesh &mesh)
  :_mesh(&mesh), _num_cells(mesh.num_cells()), _mpi_comm(mesh.mpi_comm()),
   _ptemplate(p_template),  _num_processes(MPI::size(mesh.mpi_comm()))
{
    // Note: p_array is 2D [num_particles, property_data]

    // Get geometry dimension of mesh
    _Ndim = mesh.geometry().dim();
    _Np   = p_array.rows();

    _cell2part.resize(_num_cells);

    // Initialize bounding boxes
    make_bounding_boxes();

    // _ptemplate.assign(p_template.begin(), p_template.end());

    // Calculate the offset for each particle property and overall size
    std::vector<unsigned int> offset = {0};
    for (const auto &p : p_template)
      offset.push_back(offset.back() + p);
    _plen = offset.back();

    // Loop over particles:
    for(std::size_t i=0; i<_Np; i++){
        // Position and get hosting cell
        Point xp(_Ndim, p_array.row(i).data());

        unsigned int cell_id = _mesh->bounding_box_tree()->compute_first_entity_collision(xp);
        if (cell_id != std::numeric_limits<unsigned int>::max())
        {
            // Initialize particle with position
            particle pnew = {xp};

            for(std::size_t j = 1; j < _ptemplate.size(); ++j)
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

void particles::interpolate(const Function &phih, const std::size_t property_idx){
    std::size_t space_dimension, value_size_loc;
    space_dimension = phih.function_space()->element()->space_dimension();
    value_size_loc  = 1;
    for (std::size_t i = 0; i < phih.function_space()->element()->value_rank(); i++)
      value_size_loc *= phih.function_space()->element()->value_dimension(i);

    if(value_size_loc != _ptemplate[property_idx])
        dolfin_error("particles::get_particle_contributions","get property idx",
                     "Local value size mismatches particle template property");

    for( CellIterator cell(*(_mesh)); !cell.end(); ++cell){
        std::vector<double> coeffs;
        Utils::return_expansion_coeffs(coeffs, *cell, &phih);
        for(std::size_t pidx = 0; pidx < _cell2part[cell->index()].size() ; pidx++)
        {
            std::vector<double> basis_matrix(space_dimension * value_size_loc);
            Utils::return_basis_matrix(basis_matrix, _cell2part[cell->index()][pidx][0], *cell,
                    phih.function_space()->element());

            Eigen::Map<Eigen::MatrixXd> basis_mat(basis_matrix.data(), value_size_loc, space_dimension);
            Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), space_dimension);
            Eigen::VectorXd phi_p =  basis_mat * exp_coeffs ;

            // Then update
            Point phi_point(_ptemplate[property_idx], phi_p.data());
            _cell2part[cell->index()][pidx][property_idx] = phi_point;
        }
    }
}

void particles::increment(const Function& phih_new, const Function& phih_old, const std::size_t property_idx){
    if ( !phih_new.in(*(phih_old.function_space())) )
    {
        dolfin_error("particles.cpp::incrment",
                     "Compute increment",
                     "Expected Functions to be in the same FunctionSpace");
    }

    std::size_t space_dimension, value_size_loc;
    space_dimension = phih_new.function_space()->element()->space_dimension();

    value_size_loc  = 1;
    for (std::size_t i = 0; i < phih_new.function_space()->element()->value_rank(); i++)
      value_size_loc *= phih_new.function_space()->element()->value_dimension(i);

    if(value_size_loc != _ptemplate[property_idx])
        dolfin_error("particles::get_particle_contributions","get property idx",
                     "Local value size mismatches particle template property");

    for( CellIterator cell(*(_mesh)); !cell.end(); ++cell){
        std::vector<double> coeffs_new, coeffs_old, coeffs;
        Utils::return_expansion_coeffs(coeffs_new, *cell, &phih_new);
        Utils::return_expansion_coeffs(coeffs_old, *cell, &phih_old);

        // Just average to get the coefficients
        for(std::size_t i = 0; i<coeffs_new.size(); i++)
            coeffs.push_back( coeffs_new[i] - coeffs_old[i] );

        for(std::size_t pidx = 0; pidx < _cell2part[cell->index()].size() ; pidx++)
        {
            std::vector<double> basis_matrix(space_dimension * value_size_loc);
            Utils::return_basis_matrix(basis_matrix, _cell2part[cell->index()][pidx][0], *cell,
                    phih_new.function_space()->element());

            Eigen::Map<Eigen::MatrixXd> basis_mat(basis_matrix.data(), value_size_loc, space_dimension);
            Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), space_dimension);
            Eigen::VectorXd delta_phi =  basis_mat * exp_coeffs ;

            // Then update
            Point delta_phi_p(_ptemplate[property_idx], delta_phi.data());
            _cell2part[cell->index()][pidx][property_idx] += delta_phi_p;
        }
    }
}

void particles::increment(const Function& phih_new, const Function& phih_old,
                       Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> property_idcs,
                       const double theta, const std::size_t step){
    if ( !phih_new.in(*(phih_old.function_space())) )
    {
        dolfin_error("particles.cpp::increment",
                     "Compute increment",
                     "Expected Functions to be in the same FunctionSpace");
    }

    // Check if size =2 and
    if(property_idcs.size() != 2 )
        dolfin_error("particles.cpp::increment", "Set property array", "Property indices must come in pairs");
    if(property_idcs[1] <= property_idcs[0])
        dolfin_error("particles.cpp::increment", "Set property array", "Property must be sorted in ascending order");

    // Check if length of slots matches
    if(_ptemplate[property_idcs[0]] != _ptemplate[property_idcs[1]] )
        dolfin_error("particles.cpp::increment", "Set property array", "Found none ore incorrect size at particle slot");

    std::size_t space_dimension, value_size_loc;
    space_dimension = phih_new.function_space()->element()->space_dimension();

    value_size_loc  = 1;
    for (std::size_t i = 0; i < phih_new.function_space()->element()->value_rank(); i++)
      value_size_loc *= phih_new.function_space()->element()->value_dimension(i);

    if(value_size_loc != _ptemplate[property_idcs[0]])
        dolfin_error("particles::get_particle_contributions","get property idx",
                     "Local value size mismatches particle template property");

    for( CellIterator cell(*(_mesh)); !cell.end(); ++cell){
        std::vector<double> coeffs_new, coeffs_old, coeffs;
        Utils::return_expansion_coeffs(coeffs_new, *cell, &phih_new);
        Utils::return_expansion_coeffs(coeffs_old, *cell, &phih_old);

        // Just average to get the coefficients
        for(std::size_t i = 0; i<coeffs_new.size(); i++)
            coeffs.push_back( coeffs_new[i] - coeffs_old[i] );

        for(std::size_t pidx = 0; pidx < _cell2part[cell->index()].size() ; pidx++)
        {
            std::vector<double> basis_matrix(space_dimension * value_size_loc);
            Utils::return_basis_matrix(basis_matrix, _cell2part[cell->index()][pidx][0], *cell,
                    phih_new.function_space()->element());

            Eigen::Map<Eigen::MatrixXd> basis_mat(basis_matrix.data(), value_size_loc, space_dimension);
            Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), space_dimension);
            Eigen::VectorXd delta_phi =  basis_mat * exp_coeffs ;

            Point delta_phi_p(_ptemplate[property_idcs[0]], delta_phi.data());
            // Do the update
            if(step == 1){
                _cell2part[cell->index()][pidx][property_idcs[0]] += delta_phi_p;
            }
            if(step != 1){
                _cell2part[cell->index()][pidx][property_idcs[0]] +=
                        theta * delta_phi_p + (1. - theta) * _cell2part[cell->index()][pidx][property_idcs[1]];
            }
            _cell2part[cell->index()][pidx][property_idcs[1]] = delta_phi_p;
        }
    }
}

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
particles::positions()
{
  // Could just use get_property(0)

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    xp(_Np, _Ndim);

  // Iterate over cells and particles in each cell
  std::size_t row = 0;
  for (const auto &c2p: _cell2part)
  {
    for (const auto &p: c2p)
    {
      for (std::size_t k = 0; k < _Ndim; ++k)
        xp(row, k) = p[0][k];
      ++row;
    }
  }

  return xp;
}

std::vector<double> particles::get_property(const std::size_t idx){

  // Test if idx is valid
  if (idx > _ptemplate.size())
    dolfin_error("particles::get_property",
                 "return index", "Requested index exceeds particle template");
  const std::size_t property_dim = _ptemplate[idx];

  std::vector<double> property_vector;
  property_vector.reserve(_Np * property_dim);

  // Iterate over cells and particles in each cell
  for (const auto &c2p: _cell2part)
  {
    for (const auto &p: c2p)
    {
      for(std::size_t k = 0; k < property_dim; k++)
        property_vector.push_back(p[idx][k]);
    }
  }
  return property_vector;
}

void particles::push_particle(const double dt, const Point& up, const std::size_t cidx, const std::size_t pidx)
{
    _cell2part[cidx][pidx][0] += up*dt;
}

void particles::make_bounding_boxes(){
    std::size_t _gdim = _Ndim;

    // Create bounding boxes of mesh
    std::vector<double> x_min_max(2*_gdim);
    std::vector<double> coordinates = _mesh->coordinates();
    for (std::size_t i = 0; i < _gdim; ++i)
    {
      for (auto it = coordinates.begin() + i; it < coordinates.end(); it += _gdim)
      {
        if (it == coordinates.begin() + i){
            x_min_max[i]         = *it;
            x_min_max[_gdim + i] = *it;
        }else{
            x_min_max[i]         = std::min(x_min_max[i], *it);
            x_min_max[_gdim + i] = std::max(x_min_max[_gdim + i], *it);
        }
      }
    }

    // Communicate bounding boxes
    MPI::all_gather(_mpi_comm, x_min_max, _bounding_boxes);
}

void particles::particle_communicator_collect(std::vector<std::vector<particle>>& comm_snd,
                                              const std::size_t cidx, const std::size_t pidx){
    // Assertion to check if comm_snd has size of num_procs
    dolfin_assert(comm_snd.size() == _num_processes);

    // Get position
    particle ptemp = _cell2part[cidx][pidx];
    std::vector<double> xp_temp(ptemp[0].coordinates(), ptemp[0].coordinates() + _Ndim);

    // Loop over processes
    for (std::size_t p = 0; p < _num_processes; p++)
    {
        // Check if in bounding box
        if (in_bounding_box(xp_temp, _bounding_boxes[p], 1e-12))
            comm_snd[p].push_back(ptemp);
    }

    // Erase particle
    _cell2part[cidx].erase(_cell2part[cidx].begin() + pidx);
    // Decrement particle iterator (?!)
}

void particles::particle_communicator_push(std::vector<std::vector<particle>>& comm_snd){
    // Assertion if sender has correct size
    dolfin_assert(comm_snd.size() == _num_processes);

    std::vector<std::vector<double>> comm_snd_vec(_num_processes);
    std::vector<std::vector<double>> comm_rcv_vec;

    // Prepare for communication
    for (std::size_t p = 0; p < _num_processes; p++){
        for(particle part : comm_snd[p] ){
            std::vector<double> unpacked = unpack_particle(part);
            comm_snd_vec[p].insert(comm_snd_vec[p].end(), unpacked.begin(), unpacked.end());
        }
    }

    // Communicate with all_to_all
    MPI::all_to_all(_mpi_comm, comm_snd_vec, comm_rcv_vec);

    // FIXME: the outer loop below can probably be removed
    // TODO: thoroughly test this unpacking -> sending -> composing loop

    for(std::size_t p = 0; p < _num_processes; p++){
        std::size_t pos_iter = 0;
        while(pos_iter < comm_rcv_vec[p].size()){
            // This is always the position, right?
            Point xp(_Ndim, &comm_rcv_vec[p][pos_iter]);
            unsigned int cell_id = _mesh->bounding_box_tree()->compute_first_entity_collision(xp);
            if (cell_id != std::numeric_limits<unsigned int>::max()){
                pos_iter += _Ndim; // Add geometric dimension to iterator
                particle pnew;
                pnew.push_back(xp);
                for(std::size_t j=1; j<_ptemplate.size(); j++){
                    Point property(_ptemplate[j], &comm_rcv_vec[p][pos_iter]);
                    pnew.push_back(property);
                    pos_iter += _ptemplate[j]; // Add property dimension to iterator
                }
                // Iterator position must be multiple of _plen
                dolfin_assert(pos_iter % _plen == 0);

                // Push back new particle to hosting cell
                _cell2part[cell_id].push_back(pnew);
            }else{
                // Jump to following particle in array
                pos_iter += _plen;
            }
        }
    }
}

bool particles::in_bounding_box(const std::vector<double>& point,
                                const std::vector<double>& bounding_box,
                                const double tol){
    // Return false if bounding box is empty
    if (bounding_box.empty())
        return false;

    const std::size_t gdim = point.size();
    dolfin_assert(bounding_box.size() == 2*gdim);
    for (std::size_t i = 0; i < gdim; ++i)
    {
        if (!(point[i] >= (bounding_box[i] - tol)
            && point[i] <= (bounding_box[gdim + i] + tol)))
        {
            return false;
        }
    }
    return true;
}

void particles::update_bounding_boxes(){
    // Private method for rebuilding bounding boxes (on moving meshes)
    _mesh->bounding_box_tree()->build(*(_mesh));
    // FIXME: more efficient than full rebuild of bounding boxes possible?
    make_bounding_boxes();
}

std::vector<double> particles::unpack_particle(const particle part){
    // Unpack particle into std::vector<double>
    std::vector<double> part_unpacked;
    for(std::size_t i = 0; i<_ptemplate.size(); i++)
        part_unpacked.insert(part_unpacked.end(),
                            part[i].coordinates(), part[i].coordinates()+_ptemplate[i]);
    return part_unpacked;
}


void particles::get_particle_contributions(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& q,
                                           Eigen::Matrix<double, Eigen::Dynamic, 1>& f,
                                           const Cell& dolfin_cell, std::shared_ptr<const FiniteElement> element,
                                           const std::size_t space_dimension, const std::size_t value_size_loc,
                                           const std::size_t property_idx)
{
    // TODO: some checks if element type matches property index
    if(value_size_loc != _ptemplate[property_idx])
        dolfin_error("particles::get_particle_contributions","get property idx","Local value size mismatches particle template property");

    // Get cell index and num particles
    std::size_t cidx = dolfin_cell.index();
    std::size_t _Npc = _cell2part[cidx].size();

    // Get and set cell data
    std::vector<double> vertex_coordinates;
    dolfin_cell.get_vertex_coordinates(vertex_coordinates);
    ufc::cell ufc_cell;
    dolfin_cell.get_cell_data(ufc_cell);

    // Resize return values
    q.resize(space_dimension,_Npc * value_size_loc);
    f.resize(_Npc * value_size_loc);

    if(_Npc > 0){
        for(std::size_t pidx=0; pidx<_Npc; pidx++){
            double basis_matrix[space_dimension][value_size_loc];
            element->evaluate_basis_all(&basis_matrix[0][0], _cell2part[cidx][pidx][0].coordinates(),
                                            vertex_coordinates.data(), ufc_cell.orientation);

            // Then insert in Eigen matrix and vector (rewrite this ugly loop!?)
            // Loop over number of cell dofs:
            for(std::size_t kk=0; kk < space_dimension; kk++){
                std::size_t lb = pidx* value_size_loc;
                std::size_t m  = 0; // Local counter
                // Place in matrix and vector
                for(std::size_t l=lb; l<lb+ value_size_loc; l++){
                    q(kk,l) = basis_matrix[kk][m];
                    f(l)    = _cell2part[cidx][pidx][property_idx][m];
                    m++; // Increment local iterator
                }
            }

            // TODO: TEST ME
            // element->evaluate_basis_all(&q(0,0), _cell2part[cidx][pidx][0].coordinates(),
            //                             vertex_coordinates.data(), ufc_cell.orientation);

            // Fill the vector
        }
    }else{
        // TODO: make function recognize FunctionSpace.ufl_element().family()!

        // Encountered empty cell
        for (std::size_t it = 0; it < vertex_coordinates.size(); it += _Ndim){
           std::cout<<"Coordinates vertex "<<it<<std::endl;
           for(std::size_t jt = 0; jt < _Ndim; ++jt){
               std::cout<<vertex_coordinates[it + jt]<<std::endl;
           }
        }
        dolfin_error("pdestaticcondensations.cpp::project", "perform projection",
                     "Cells without particle not yet handled, empty cell (%d)", cidx);
    }
}
