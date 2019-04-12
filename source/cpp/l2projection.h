// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifndef L2PROJECTION_H
#define L2PROJECTION_H

#include <memory>
#include <vector>

namespace dolfin
{

namespace function
{
class Function;
class FunctionSpace;
} // namespace function

class particles;

class l2projection
{
public:
  l2projection(particles& P, function::FunctionSpace& V, const std::size_t idx);
  ~l2projection();

  // l^2 map
  void project(function::Function& u);

  // l^2 map with bound constraint
  void project(function::Function& u, const double lb, const double ub);

  // l^2 map onto cg space (global problem)
  void project_cg(const fem::Form& A, const fem::Form& f,
                  function::Function& u);

protected:
  particles* _P; // Pointer object to particles

  std::shared_ptr<const fem::FiniteElement> _element;
  std::shared_ptr<const fem::GenericDofMap> _dofmap;
  std::size_t _num_subspaces, _space_dimension, _num_dof_locs, _value_size_loc;

  // Workaround to access tuple elements
  // FIXME: what happens if we make this a constant?
  std::size_t _idx_pproperty;
};
} // namespace dolfin
#endif // L2PROJECTION_H
