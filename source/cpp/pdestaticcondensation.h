// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com

#ifndef PDESTATICCONDENSATION_H
#define PDESTATICCONDENSATION_H

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

#include <dolfin/common/Array.h>
#include <dolfin/common/ArrayView.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include "dolfin/la/solve.h"

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>

#include "formutils.h"
#include "advect_particles.h"
#include "particles.h"
#include <ufc.h>

namespace dolfin{
    template<typename T> class Array;

    class PDEStaticCondensation
    { 
    // Class providing functionality for PDE constrained projection using static condensation.
    // Expect the dolfin Forms to comply with the algebraic form:
    //
    //    |  N   G   L | | psi     |    |  Q  |
    //    |  G^T 0   H | | lambda  | =  |  R  |
    //    |  L^T H^T B | | Psi_bar |    |  S  |
    //
    public:
        // Constructor
        PDEStaticCondensation(const Mesh& mesh, particles& P,
                              const Form& N, const Form& G, const Form& L,
                              const Form& H, const Form& B,
                              const Form& Q, const Form& R, const Form& S,
                              const std::size_t idx_pproperty);

        // Constructor including Dirichlet BC's
        PDEStaticCondensation(const Mesh& mesh, particles& P,
                              const Form& N, const Form& G, const Form& L,
                              const Form& H, const Form& B,
                              const Form& Q, const Form& R, const Form& S,
                              std::vector<std::shared_ptr<const DirichletBC>> bcs,
                              const std::size_t idx_pproperty);

        ~PDEStaticCondensation();

        // TO DO: assemble_on_config labels the rhs assembly
        void assemble(const bool assemble_all=true, const bool assemble_on_config = true );
        void assemble_state_rhs();

        void solve_problem(Function& Uglobal, Function& Ulocal,
                           const std::string solver="none", const std::string preconditioner="default");
        void solve_problem(Function& Uglobal, Function& Ulocal, Function& Lambda,
                           const std::string solver="none", const std::string preconditioner="default");
        void apply_boundary(DirichletBC& DBC);
    private:
        // Private Methods
        void backsubtitute(const Function &Uglobal, Function &Ulocal);
        void backsubtitute(const Function &Uglobal, Function &Ulocal, Function& Lambda);

        /* Comes from particles
        void get_particle_contributions(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& N_ep,
                                        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& R_ep,
                                        const Cell& dolfin_cell);
        */

        // Private Attributes
        const Form *N, *G, *L, *H, *B, *Q, *R, *S;
        const Mesh* mesh;
        particles* _P;

        const MPI_Comm mpi_comm;
        Matrix A_g;
        Vector f_g;

        std::shared_ptr<const FiniteElement> _element;
        std::size_t _num_subspaces, _space_dimension, _num_dof_locs,
                    _value_size_loc;
        // TODO: set _idx_pproperty
        const std::size_t _idx_pproperty;

        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > invKS_list;
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > Ge_list;
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > LHe_list;
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > Be_list;
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1> > Re_list, QRe_list;
        std::vector<std::shared_ptr<const DirichletBC>> bcs;

        // FIXME needed for momentum based l2 map
        Function* rhoh;
    };
}
#endif // PDESTATICCONDENSATION_H
