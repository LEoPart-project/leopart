#ifndef UTILS_H
#define UTILS_H

#include <ufc.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/FunctionSpace.h>

namespace dolfin{
    class Utils
    {
    friend class advect_particles;
    // Some utility functions for, header only
    public:
        Utils(){}
        ~Utils(){}

        static void return_expansion_coeffs(std::vector<double>& coeffs,
                                            Cell& dolfin_cell, const Function* phih)
        {
            // Get expansion coefficients phi_i in N_i . phi_i
            std::vector<double> vertex_coordinates;
            dolfin_cell.get_vertex_coordinates(vertex_coordinates);
            ufc::cell ufc_cell;
            dolfin_cell.get_cell_data(ufc_cell);

            std::shared_ptr<const FiniteElement> element = phih->function_space()->element();
            coeffs.resize( element->space_dimension());
            // Get coefficients
            phih->restrict(coeffs.data(), *element, dolfin_cell, vertex_coordinates.data() , ufc_cell);

        }

        static void return_basis_matrix(std::vector<double>& basis_matrix,
                                        const Point xp, const Cell& dolfin_cell,
                                        std::shared_ptr<const FiniteElement> element)
        {
            std::vector<double> vertex_coordinates;
            dolfin_cell.get_vertex_coordinates(vertex_coordinates);
            ufc::cell ufc_cell;
            dolfin_cell.get_cell_data(ufc_cell);

            element->evaluate_basis_all(basis_matrix.data(), xp.coordinates(),
                                        vertex_coordinates.data(),
                                        ufc_cell.orientation);
        }
    };
}
#endif // UTILS_H
