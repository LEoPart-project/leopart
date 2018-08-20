#include "formutils.h"

using namespace dolfin;

FormUtils::FormUtils(){}
//-----------------------------------------------------------------------------
FormUtils::~FormUtils(){}
//-----------------------------------------------------------------------------
void FormUtils::test_rank(const Form &a, const std::size_t rank){
    if (a.rank() !=  rank)
        dolfin_error("PDEStaticCondensation::test_rank", "get correct rank",
                     "Proper forms specified?");
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> FormUtils::local_tensor_size(const Form& a, const Cell& cell)
{
  if (a.rank() == 0)
    return std::make_pair(1, 1);
  else if (a.rank() == 1)
    return std::make_pair(a.function_space(0)->dofmap()->cell_dofs(cell.index()).size(), 1);
  else
    return std::make_pair(a.function_space(0)->dofmap()->cell_dofs(cell.index()).size(),
                          a.function_space(1)->dofmap()->cell_dofs(cell.index()).size());
}
//-----------------------------------------------------------------------------
// void FormUtils::local_tensor_info(const Form& a, const Cell& cell,
//                                   std::size_t* nrows, ArrayView<const dolfin::la_index>& cdof_rows,
//                                   std::size_t* ncols, ArrayView<const dolfin::la_index>& cdof_cols)
// {
//     if (a.rank() == 0)
//     {
//         *nrows = 1;
//         *ncols = 1;
//     }
//     else if (a.rank() == 1)
//     {
//         cdof_rows = a.function_space(0)->dofmap()->cell_dofs(cell.index());
//         *nrows = cdof_rows.size();
//         *ncols = 1;
//     }
//     else
//     {
//         cdof_rows = a.function_space(0)->dofmap()->cell_dofs(cell.index());
//         cdof_cols = a.function_space(1)->dofmap()->cell_dofs(cell.index());
//         *nrows = cdof_rows.size();
//         *ncols = cdof_cols.size();
//     }
// }
// //-----------------------------------------------------------------------------
void FormUtils::local_assembler(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &A_e,
                                const Form &a, const Cell &cell, const std::size_t nrows, const std::size_t ncols)
{
    // Method largely based on dolfin::assemble_local method in dolfin repo
    A_e.resize(nrows, ncols);
    UFC ufc(a);
    ufc::cell ufc_cell;
    std::vector<double> coordinate_dofs;

    // Extract cell_domains etc from the form
    const MeshFunction<std::size_t>* cell_domains =
        a.cell_domains().get();
    const MeshFunction<std::size_t>* exterior_facet_domains =
        a.exterior_facet_domains().get();
    const MeshFunction<std::size_t>* interior_facet_domains =
        a.interior_facet_domains().get();

    // Update to the local cell and assemble
    cell.get_coordinate_dofs(coordinate_dofs);
    LocalAssembler::assemble(A_e, ufc, coordinate_dofs,
                             ufc_cell, cell, cell_domains,
                             exterior_facet_domains, interior_facet_domains);
}
//-----------------------------------------------------------------------------
void FormUtils::apply_boundary_symmetric(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& LHS_e,
                                         Eigen::Matrix<double, Eigen::Dynamic, 1>& RHS_e,
                                         Eigen::Map<const Eigen::Array<dolfin::la_index, Eigen::Dynamic, 1>> cdof_rows,
                                         Eigen::Map<const Eigen::Array<dolfin::la_index, Eigen::Dynamic, 1>> cdof_cols,
                                        const std::vector<DirichletBC::Map>& boundary_values,
                                        const bool active_bcs)
{
    if (active_bcs)
    {
        // Square matrix with same FunctionSpace on each axis
        // Loop over columns/rows
        for (int i = 0; i < cdof_cols.size(); ++i)
        {
            const std::size_t ii = cdof_cols[i];
            DirichletBC::Map::const_iterator bc_value = boundary_values[0].find(ii);
            if (bc_value != boundary_values[0].end())
            {
                double rowsum = LHS_e.row(i).sum();
                // Zero row
                LHS_e.row(i).setZero();

                // Modify RHS (subtract (bc_column(A))*bc_val from b)
                RHS_e -= LHS_e.col(i)*bc_value->second;

                // Zero column
                LHS_e.col(i).setZero();

                // Place 1 on diagonal and bc on RHS (i th row ).
                // Scale these values...
                RHS_e(i)    = rowsum * bc_value->second;
                LHS_e(i, i) = rowsum * 1.0;
            }
        }
    }
}
