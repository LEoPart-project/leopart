// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifndef PARTICLES_H
#define PARTICLES_H

#include <Eigen/Dense>
#include <vector>

#include <dolfin/common/MPI.h>

#include "particle.h"

namespace dolfin
{
// Forward declarations
class Function;
class FiniteElement;
class Point;
class Mesh;

class particles
{

public:
  particles(Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>
                p_array,
            const std::vector<unsigned int>& p_template, const Mesh& mesh);

  ~particles();

  // Get the position of a particle in a cell
  // Just a shorthand for "property(cidx, pidx, 0)"
  const Point& x(int cell_index, int particle_index) const
  {
    return _cell2part[cell_index][particle_index][0];
  }

  // Return property i of particle in cell
  const Point& property(int cell_index, int particle_index, int i) const
  {
    return _cell2part[cell_index][particle_index][i];
  }

  unsigned int expand_template(int dim)
  {
    // Add new template item and initialise all particles with extra
    // empty slot
    _ptemplate.push_back(dim);
    _plen += dim;
    Point p(0.0, 0.0, 0.0);
    for (unsigned int cidx = 0; cidx < _mesh->num_cells(); ++cidx)
      for (unsigned int pidx = 0; pidx < num_cell_particles(cidx); ++pidx)
        _cell2part[cidx][pidx].push_back(p);

  // Resize members for new number of properties
  _empty_cell_property_values.resize(num_properties());

    return _ptemplate.size() - 1;
  }

  void set_property(int cell_index, int particle_index, int i, Point v)
  {
    _cell2part[cell_index][particle_index][i] = v;
  }

  // Pointer to the mesh
  const Mesh* mesh() const { return _mesh; }

  // Get size of property i
  unsigned int ptemplate(int i) const { return _ptemplate[i]; }

  // Number of properties
  unsigned int num_properties() const { return _ptemplate.size(); }

  // Number of particles in Cell c
  unsigned int num_cell_particles(int c) const { return _cell2part[c].size(); }

  // Add particle to cell returning particle index
  int add_particle(int c);

  // Remove ith particle from cell c
  void delete_particle(int c, int i)
  {
    _cell2part[c].erase(_cell2part[c].begin() + i);
  }

  // Interpolate function to particles
  void interpolate(const Function& phih, const size_t property_idx);

  // Increment
  void increment(const Function& phih_new, const Function& phih_old,
                 const size_t property_idx);

  // Increment using theta --> Consider replacing property_idcs
  void increment(const Function& phih_new, const Function& phih_old,
                 Eigen::Ref<const Eigen::Array<size_t, Eigen::Dynamic, 1>>
                     property_idcs,
                 const double theta, const size_t step);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  positions();
  std::vector<double> get_property(const size_t idx);

  void get_particle_contributions(
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& q,
      Eigen::Matrix<double, Eigen::Dynamic, 1>& f, const Cell& dolfin_cell,
      std::shared_ptr<const FiniteElement> element,
      const size_t space_dimension, const size_t value_size_loc,
      const size_t property_idx);

  // Push particle to new position
  void push_particle(const double dt, const Point& up, const size_t cidx,
                     const size_t pidx);

  // Particle collector, required in parallel
  void particle_communicator_collect(const size_t cidx,
                                     const size_t pidx);

  // Particle pusher, required in parallel
  void particle_communicator_push();

  // Relocate all particles, required on moving meshes
  void relocate();

  // Relocate particles, with known relocation data. Each entry is {cidx, pidx,
  // cidx_recv} using numeric_limits::max for cidx_recv to send to another
  // process
  void relocate(std::vector<std::array<size_t, 3>>& reloc);

  // If a cell has no particles, take the values from this cell function
  // as the default value in get_particle_contributions()
  void set_empty_cell_default_values(
    std::shared_ptr<dolfin::MeshFunction<double>> cell_function,
    const size_t property_idx)
  {
    if (_empty_cell_property_values.size() < num_properties())
        _empty_cell_property_values.resize(num_properties());

    _empty_cell_property_values[property_idx] = cell_function;
  }

private:
  std::vector<std::vector<particle>> _comm_snd;

  std::vector<double> unpack_particle(const particle part);

  // Attributes
  const Mesh* _mesh;
  size_t _Ndim;
  std::vector<std::vector<particle>> _cell2part;

  // Particle properties
  std::vector<unsigned int> _ptemplate;
  size_t _plen;
  std::vector<std::shared_ptr<dolfin::MeshFunction<double>>>
    _empty_cell_property_values;

  // Needed for parallel
  const MPI_Comm _mpi_comm;
};
} // namespace dolfin

#endif // PARTICLES_H
