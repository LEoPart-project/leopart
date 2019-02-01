// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifndef L2PROJECTION_H
#define L2PROJECTION_H

#include <vector>
#include <memory>


namespace dolfin
{
  class Function;
  class FunctionSpace;
  class particles;


  class l2projection
{
public:
  l2projection(particles& P, FunctionSpace& V, const std::size_t idx);
  ~l2projection();

  // l^2 map
  void project(Function& u);

  // l^2 map with bound constraint
  void project(Function& u, const double lb, const double ub);

  // l^2 map onto cg space (global problem)
  void project_cg(const Form& A, const Form& f, Function& u);

protected:
  particles* _P; // Pointer object to particles

  std::shared_ptr<const FiniteElement> _element;
  std::shared_ptr<const GenericDofMap> _dofmap;
  std::size_t _num_subspaces, _space_dimension, _num_dof_locs, _value_size_loc;

  // Workaround to access tuple elements
  // FIXME: what happens if we make this a constant?
  std::size_t _idx_pproperty;
};
} // namespace dolfin
#endif // L2PROJECTION_H
