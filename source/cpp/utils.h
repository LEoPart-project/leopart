// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifndef UTILS_H
#define UTILS_H

#include <vector>

#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Cell.h>
#include <ufc.h>

namespace dolfin
{
class Utils
{
  // Some utility functions for, header only
public:
  static void return_expansion_coeffs(std::vector<double>& coeffs,
                                      const mesh::Cell& cell,
                                      const function::Function* phih)
  {

    const mesh::Mesh& mesh = cell.mesh();

    // Prepare cell geometry
    const int tdim = mesh.topology().dim();
    const int gdim = mesh.geometry().dim();
    const mesh::Connectivity& connectivity_g
        = mesh.coordinate_dofs().entity_points(tdim);

    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
        = connectivity_g.entity_positions();

    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
        = connectivity_g.connections();

    // FIXME: Add proper interface for num coordinate dofs
    const int num_dofs_g = connectivity_g.size(0);
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        x_g
        = mesh.geometry().points();

    // Get expansion coefficients phi_i in N_i . phi_i
    std::shared_ptr<const fem::FiniteElement> element
        = phih->function_space()->element();

    // Loop over cells and tabulate dofs
    EigenRowArrayXXd coordinate_dofs(num_dofs_g, gdim);

    // Get cell coordinates
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    coeffs.resize(element->space_dimension());
    // Get coefficients
    phih->restrict(coeffs.data(), cell, coordinate_dofs);
  }

  // Compute basis matrix directly to pointer address
  static void
  return_basis_matrix(double* basis_matrix, const geometry::Point xp,
                      const mesh::Cell& cell,
                      std::shared_ptr<const fem::FiniteElement> element)
  {
    const mesh::Mesh& mesh = cell.mesh();

    // Prepare cell geometry
    const int tdim = mesh.topology().dim();
    const int gdim = mesh.geometry().dim();
    const mesh::Connectivity& connectivity_g
        = mesh.coordinate_dofs().entity_points(tdim);

    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
        = connectivity_g.entity_positions();

    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
        = connectivity_g.connections();

    // FIXME: Add proper interface for num coordinate dofs
    const int num_dofs_g = connectivity_g.size(0);
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        x_g
        = mesh.geometry().points();

    // Get cell coordinates
    EigenRowArrayXXd coordinate_dofs(num_dofs_g, gdim);
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    const auto cmap = mesh.geometry().coord_mapping;
    EigenRowArrayXXd X(1, gdim);
    Eigen::Tensor<double, 3, Eigen::RowMajor> J(1, gdim, tdim);
    EigenArrayXd detJ(1);
    Eigen::Tensor<double, 3, Eigen::RowMajor> K(1, tdim, gdim);
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        xparray;
    Eigen::Tensor<double, 3, Eigen::RowMajor> basis;

    cmap->compute_reference_geometry(X, J, detJ, K, xparray, coordinate_dofs);

    element->evaluate_reference_basis(basis, X);
  }

  static void cell_bounding_box(std::vector<double>& x_min_max,
                                const std::vector<double>& coordinates,
                                const std::size_t gdim)
  {
    // Consider merging with make_bounding_boxes in particles class?!
    x_min_max.resize(2 * gdim);
    for (std::size_t i = 0; i < gdim; ++i)
    {
      for (auto it = coordinates.begin() + i; it < coordinates.end();
           it += gdim)
      {
        if (it == coordinates.begin() + i)
        {
          x_min_max[i] = *it;
          x_min_max[gdim + i] = *it;
        }
        else
        {
          x_min_max[i] = std::min(x_min_max[i], *it);
          x_min_max[gdim + i] = std::max(x_min_max[gdim + i], *it);
        }
      }
    }
  }
};
} // namespace dolfin
#endif // UTILS_H
