#ifndef STOKESSTATICCONDENSATION_H
#define STOKESSTATICCONDENSATION_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/Assembler.h>
#include "dolfin/fem/LocalAssembler.h"
#include "dolfin/fem/AssemblerBase.h"
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/DirichletBC.h>
#include "dolfin/fem/UFC.h"

#include <dolfin/common/ArrayView.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/solve.h>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>

#include "formutils.h"

namespace dolfin{
    class StokesStaticCondensation
    {
    public:
        // Constructors with assumed symmetry
        StokesStaticCondensation(const Mesh& mesh,  const Form& A, const Form& G, const Form& B,
                                                    const Form& Q, const Form& S);
        StokesStaticCondensation(const Mesh& mesh,  const Form& A, const Form& G, const Form& B,
                                                    const Form& Q, const Form& S,
                                                    std::vector<std::shared_ptr<const DirichletBC>> bcs);
        // Constructors assuming full [2x2] block specification
        StokesStaticCondensation(const Mesh& mesh,  const Form& A,  const Form& G,
                                                    const Form& GT, const Form& B,
                                                    const Form& Q,  const Form& S);

        StokesStaticCondensation(const Mesh& mesh,  const Form& A,  const Form& G,
                                                    const Form& GT, const Form& B,
                                                    const Form& Q,  const Form& S,
                                                    std::vector<std::shared_ptr<const DirichletBC>> bcs);

        // Destructor
        ~StokesStaticCondensation();

        // Public Methods
        void assemble_global();
        void assemble_global_lhs();
        void assemble_global_rhs();
        void assemble_global_system(bool assemble_lhs = true);

        void apply_boundary(DirichletBC& DBC);
        void solve_problem(Function& Uglobal, Function& Ulocal,
                           const std::string solver = "none", const std::string preconditioner = "default");

    private:
        // Private Methods
//        void local_tensor_info(const Form& a, const Cell& cell,
//                               std::size_t* nrows, ArrayView<const dolfin::la_index>& cdof_rows,
//                               std::size_t* ncols, ArrayView<const dolfin::la_index>& cdof_cols);
//        void local_assembler(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& A_e,
//                        const Form& a, const Cell& cell, const std::size_t nrows, const std::size_t ncols);

        void backsubtitute(const Function& Uglobal, Function& Ulocal);
        void test_rank(const Form& a, const std::size_t rank);
        void apply_boundary_symmetric(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& LHS_e,
                                      Eigen::Matrix<double, Eigen::Dynamic, 1>& RHS_e,
                                      const ArrayView<const dolfin::la_index>& cdof_rows,
                                      const ArrayView<const dolfin::la_index>& cdof_cols,
                                      const std::vector<DirichletBC::Map>& boundary_values,
                                      const bool active_bcs);

        // Private Attributes
        const Form* A;
        const Form* G;
        const Form* B;
        const Form* Q;
        const Form* S;
        const Form* GT;
        const Mesh* mesh;

        bool assume_symmetric;

        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > invAe_list;
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > Ge_list;
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > Be_list;
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1> > Qe_list;

        // Facilitate non-symmetric case
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > GTe_list;

        const MPI_Comm mpi_comm;
        Matrix A_g;
        Vector f_g;
        std::vector<std::shared_ptr<const DirichletBC>> bcs;
    };
}

#endif // STOKESSTATICCONDENSATION_H
