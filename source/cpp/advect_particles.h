// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifndef ADVECT_PARTICLES_H
#define ADVECT_PARTICLES_H

#include <Eigen/Dense>
#include <vector>
#include <memory>

#include <dolfin/geometry/Point.h>

#include "particles.h"

namespace dolfin
{
  // Forward declarations
  class FunctionSpace;
  class Function;
  class FiniteElement;
  template<typename T> class MeshFunction;

// enum for external facet types
enum class facet_t : std::uint8_t
{
  internal,
  closed,
  open,
  periodic,
  bounded
};

// Facet info on each facet of mesh
typedef struct facet_info_t
{
  Point midpoint;
  Point normal;
  facet_t type;
} facet_info;

class advect_particles
{

public:
  // Constructors
  advect_particles(particles& P, FunctionSpace& U,
                   std::function<const Function&(int, double)> uhi,
                   const std::string type1);

  // Document
  advect_particles(
      particles& P, FunctionSpace& U,
      std::function<const Function&(int, double)> uhi,
      const std::string type1,
      Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits);

  // Document
  advect_particles(particles& P, FunctionSpace& U,
                   std::function<const Function&(int, double)> uhi,
                   const MeshFunction<std::size_t>& mesh_func);

  // Document
  advect_particles(
      particles& P, FunctionSpace& U,
      std::function<const Function&(int, double)> uhi,
      const MeshFunction<std::size_t>& mesh_func,
      Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits);

  // Document
  advect_particles(
      particles& P, FunctionSpace& U,
      std::function<const Function&(int, double)> uhi,
      const MeshFunction<std::size_t>& mesh_func,
      Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits,
      Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> bounded_limits);

  // Step forward in time dt
  void do_step(double dt);

  // Update facet info on moving mesh
  void update_facets_info();

  // Destructor
  ~advect_particles();

  virtual void init_weights() {}

protected:
  particles* _P;

  void set_bfacets(const std::string btype);
  void set_bfacets(const MeshFunction<std::size_t>& mesh_func);

  // Limits for periodic facets
  std::vector<std::vector<double>> pbc_lims; // Coordinates of limits
  bool pbc_active = false;

  // Limits for bounded facets
  std::vector<std::vector<double>> bounded_domain_lims; // Coordinates of limits
  bool bounded_domain_active = false;

  // Timestepping scheme related
  std::vector<double> dti;
  std::vector<double> weights;

  std::size_t _space_dimension, _value_size_loc;

  // Facet information
  // (normal, midpoint, type(internal, open, closed, periodic, bounded))
  std::vector<facet_info> facets_info;

  std::function<const Function&(int, double)> uh;
  std::shared_ptr<const FiniteElement> _element;

  // Must receive a point xp
  std::tuple<std::size_t, double>
  time2intersect(std::size_t cidx, double dt, const Point xp, const Point up);

  // Consider placing in particle class
  // void push_particle(const double dt, const Point& up, const std::size_t
  // cidx, const std::size_t pidx);

  // Methods for applying bc's
  void apply_open_bc(std::size_t cidx, std::size_t pidx);
  void apply_closed_bc(double dt, Point& up, std::size_t cidx, std::size_t pidx,
                       std::size_t fidx);
  void apply_periodic_bc(double dt, Point& up, std::size_t cidx,
                         std::size_t pidx, std::size_t fidx);
  void apply_bounded_domain_bc(double dt, Point& up, std::size_t cidx,
                               std::size_t pidx, std::size_t fidx);

  void pbc_limits_violation(std::size_t cidx, std::size_t pidx);
  void bounded_domain_violation(std::size_t cidx, std::size_t pidx);

  // TODO: Make pure virtual function for do_step?
  // Method for substepping in multistep schemes

  void do_substep(double dt, Point& up, const std::size_t cidx,
                  std::size_t pidx, const std::size_t step,
                  const std::size_t num_steps, const std::size_t xp0_idx,
                  const std::size_t up0_idx,
                  std::vector<std::array<std::size_t, 3>>& reloc);

  // Multi-stage scheme data
  std::size_t xp0_idx, up0_idx;

  private:

  void update_particle_template()
  {
    const std::size_t gdim = _P->mesh()->geometry().dim();
    xp0_idx = _P->expand_template(gdim);
    up0_idx = _P->expand_template(gdim);

    // Copy position to xp0 property
    for (unsigned int cidx = 0; cidx < _P->mesh()->num_cells(); ++cidx)
    {
      for (unsigned int pidx = 0; pidx < _P->num_cell_particles(cidx); ++pidx)
        _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));
    }
  }


};

class advect_rk2 : public advect_particles
{
public:
  using advect_particles::advect_particles;

  // Step forward in time dt
  void do_step(double dt);

  void init_weights()
  {
    dti = {1.0, 1.0};
    weights = {0.5, 0.5};
  }
};

class advect_rk3 : public advect_particles
{
public:
  using advect_particles::advect_particles;

  // Step forward in time dt
  void do_step(double dt);

  void init_weights()
  {
    dti = {0.5, 0.75, 1.0};
    weights = {2. / 9., 3. / 9., 4. / 9.};
  }
};

class advect_rk4 : public advect_particles
{
public:
  using advect_particles::advect_particles;

  // Step forward in time dt
  void do_step(double dt);

protected:

  void init_weights()
  {
    dti = {0.5, 0.5, 1.0, 1.0};
    weights = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
  }
};
} // namespace dolfin
#endif // ADVECT_PARTICLES_H
