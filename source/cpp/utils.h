// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifndef UTILS_H
#define UTILS_H

#include <vector>

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
                                      const Cell& dolfin_cell,
                                      const Function* phih)
  {
    // Get expansion coefficients phi_i in N_i . phi_i
    std::vector<double> vertex_coordinates;
    dolfin_cell.get_vertex_coordinates(vertex_coordinates);
    ufc::cell ufc_cell;
    dolfin_cell.get_cell_data(ufc_cell);

    std::shared_ptr<const FiniteElement> element
        = phih->function_space()->element();
    coeffs.resize(element->space_dimension());
    // Get coefficients
    phih->restrict(coeffs.data(), *element, dolfin_cell,
                   vertex_coordinates.data(), ufc_cell);
  }

  // Compute basis matrix directly to pointer address
  static void return_basis_matrix(double* basis_matrix,
                                  const Point xp, const Cell& dolfin_cell,
                                  std::shared_ptr<const FiniteElement> element)
  {
    std::vector<double> vertex_coordinates;
    dolfin_cell.get_vertex_coordinates(vertex_coordinates);
    ufc::cell ufc_cell;
    dolfin_cell.get_cell_data(ufc_cell);

    element->evaluate_basis_all(basis_matrix, xp.coordinates(),
                                vertex_coordinates.data(),
                                ufc_cell.orientation);
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
