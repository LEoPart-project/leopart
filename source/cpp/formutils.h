// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifndef FORMUTILS_H
#define FORMUTILS_H

#include <Eigen/Dense>
#include <vector>

#include <dolfin/fem/DirichletBC.h>

namespace dolfin
{
class Form;
class Cell;

class FormUtils
{
  // Some utility functions for operations on dolfin Forms.
  // Should contain only static methods.

public:
  // Test rank of form
  static void test_rank(const Form& a, const std::size_t rank);

  // Get local tensor size
  static std::pair<std::size_t, std::size_t>
  local_tensor_size(const Form& a, const Cell& cell);

  // Local assembler
  static void
  local_assembler(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>& A_e,
                  const Form& a, const Cell& cell, const std::size_t nrows,
                  const std::size_t ncols);

  // Apply Dirichlet BC to element contributions, so as to maintain symmetry
  static void apply_boundary_symmetric(
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
          LHS_e,
      Eigen::Matrix<double, Eigen::Dynamic, 1>& RHS_e,
      Eigen::Map<const Eigen::Array<dolfin::la_index, Eigen::Dynamic, 1>>
          cdof_rows,
      Eigen::Map<const Eigen::Array<dolfin::la_index, Eigen::Dynamic, 1>>
          cdof_cols,
      const std::vector<DirichletBC::Map>& boundary_values,
      const bool active_bcs);
};
} // namespace dolfin
#endif // FORMUTILS_H
