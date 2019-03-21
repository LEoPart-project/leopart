// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <memory>

#include <dolfin/fem/Assembler.h>
#include <dolfin/fem/AssemblerBase.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/solve.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>

#include "formutils.h"
#include "stokesstaticcondensation.h"

using namespace dolfin;

StokesStaticCondensation::StokesStaticCondensation(const Mesh& mesh,
                                                   const Form& A, const Form& G,
                                                   const Form& B, const Form& Q,
                                                   const Form& S)
    : mesh(&mesh), A(&A), B(&B), G(&G), Q(&Q), S(&S),
      invAe_list(mesh.num_cells()), Ge_list(mesh.num_cells()),
      Be_list(mesh.num_cells()), Qe_list(mesh.num_cells()),
      mpi_comm(mesh.mpi_comm())
{
  // Check that global problem is square, otherwise, raise error
  // TODO: Perform some checks on functionspaces
  test_rank(*(this->A), 2);
  test_rank(*(this->B), 2);
  test_rank(*(this->G), 2);
  test_rank(*(this->Q), 1);
  test_rank(*(this->S), 1);

  // Initialize matrix and vector with proper sparsity structures
  AssemblerBase assembler_base;
  assembler_base.init_global_tensor(A_g, *(this->B));
  assembler_base.init_global_tensor(f_g, *(this->S));

  assume_symmetric = true;
}
//-----------------------------------------------------------------------------
StokesStaticCondensation::StokesStaticCondensation(
    const Mesh& mesh, const Form& A, const Form& G, const Form& B,
    const Form& Q, const Form& S,
    std::vector<std::shared_ptr<const DirichletBC>> bcs)
    : StokesStaticCondensation::StokesStaticCondensation(mesh, A, G, B, Q, S)
{
  this->bcs = bcs;
  // TODO: perform check on bcs input, see SystemAssembler
  // TODO: assemble systems such that symmetry is preserved! see
  // SystemAssembler.cpp
  // TODO: Check that B is square, rectangular not yet implemented
}
//-----------------------------------------------------------------------------
StokesStaticCondensation::StokesStaticCondensation(const Mesh& mesh,
                                                   const Form& A, const Form& G,
                                                   const Form& GT,
                                                   const Form& B, const Form& Q,
                                                   const Form& S)
    : StokesStaticCondensation::StokesStaticCondensation(mesh, A, G, B, Q, S)
{
  this->GT = &GT;
  test_rank(*(this->GT), 2);

  GTe_list.resize(mesh.num_cells());
  // Set assume_symmetric to false
  assume_symmetric = false;
}
//-----------------------------------------------------------------------------
StokesStaticCondensation::StokesStaticCondensation(
    const Mesh& mesh, const Form& A, const Form& G, const Form& GT,
    const Form& B, const Form& Q, const Form& S,
    std::vector<std::shared_ptr<const DirichletBC>> bcs)
    : StokesStaticCondensation::StokesStaticCondensation(mesh, A, G, GT, B, Q,
                                                         S)
{
  this->bcs = bcs;
}
//-----------------------------------------------------------------------------
StokesStaticCondensation::~StokesStaticCondensation() {}
//-----------------------------------------------------------------------------
void StokesStaticCondensation::assemble_global()
{
  assemble_global_lhs();
  assemble_global_rhs();
}
//-----------------------------------------------------------------------------
void StokesStaticCondensation::assemble_global_lhs()
{
  // Reset matrix to zero, keep sparsity structure
  A_g.zero();
  // For each cell: G_e.T.dot(A_e.inv).dot(G_e) - B_e
  for (CellIterator cell(*(this->mesh)); !cell.end(); ++cell)
  {
    std::size_t nrowsA, ncolsA, nrowsB, ncolsB, nrowsG, ncolsG;

    std::tie(nrowsA, ncolsA) = FormUtils::local_tensor_size(*(this->A), *cell);
    std::tie(nrowsB, ncolsB) = FormUtils::local_tensor_size(*(this->B), *cell);
    std::tie(nrowsG, ncolsG) = FormUtils::local_tensor_size(*(this->G), *cell);

    // Get local matrices, give size upon initialization?
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_e,
        B_e, G_e;

    FormUtils::local_assembler(A_e, *(this->A), *cell, nrowsA, ncolsA);
    FormUtils::local_assembler(B_e, *(this->B), *cell, nrowsB, ncolsB);
    FormUtils::local_assembler(G_e, *(this->G), *cell, nrowsG, ncolsG);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        LHS_e;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        invA_e = A_e.inverse();

    Ge_list[cell->index()] = G_e;
    invAe_list[cell->index()] = invA_e;

    if (assume_symmetric)
    {
      LHS_e = G_e.transpose() * invA_e * G_e - B_e;
    }
    else
    {
      // No symmetry assumption
      std::size_t nrowsGT, ncolsGT;
      std::tie(nrowsGT, ncolsGT)
          = FormUtils::local_tensor_size(*(this->GT), *cell);

      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          GT_e;
      FormUtils::local_assembler(GT_e, *(this->GT), *cell, nrowsGT, ncolsGT);
      LHS_e = GT_e * invA_e * G_e - B_e;

      // And store for later use
      GTe_list[cell->index()] = GT_e;
    }

    auto cdof_rowsB
        = this->B->function_space(0)->dofmap()->cell_dofs(cell->index());
    auto cdof_colsB
        = this->B->function_space(1)->dofmap()->cell_dofs(cell->index());

    A_g.add_local(LHS_e.data(), nrowsB, cdof_rowsB.data(), ncolsB,
                  cdof_colsB.data());
  }
  A_g.apply("add");
}
//-----------------------------------------------------------------------------
void StokesStaticCondensation::assemble_global_rhs()
{
  // If invA_list or Ge_list empty, then throw error
  if (invAe_list[0].size() == 0 || Ge_list[0].size() == 0)
    dolfin_error(
        "StokesStaticCondensation::assemble_global_rhs",
        "assemble global RHS vector",
        "Global RHS vector depends on LHS matrix, so assemble LHS first");

  // Reset vector to zero, keep sparsity structure
  f_g.zero();
  // For each cell: G_e.T.dot(A_e.inv).dot(Q_e) - S_e
  for (CellIterator cell(*(this->mesh)); !cell.end(); ++cell)
  {
    std::size_t nrowsQ, ncolsQ, nrowsS, ncolsS;
    std::tie(nrowsQ, ncolsQ) = FormUtils::local_tensor_size(*(this->Q), *cell);
    std::tie(nrowsS, ncolsS) = FormUtils::local_tensor_size(*(this->S), *cell);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Q_e,
        S_e;

    FormUtils::local_assembler(Q_e, *(this->Q), *cell, nrowsQ, ncolsQ);
    FormUtils::local_assembler(S_e, *(this->S), *cell, nrowsS, ncolsS);

    Eigen::Matrix<double, Eigen::Dynamic, 1> RHS_e;
    if (assume_symmetric)
    {
      RHS_e
          = Ge_list[cell->index()].transpose() * invAe_list[cell->index()] * Q_e
            - S_e;
    }
    else
    {
      RHS_e = GTe_list[cell->index()] * invAe_list[cell->index()] * Q_e - S_e;
    }

    // Alternatively, use this->B?!
    auto cdof_rowsS
        = this->S->function_space(0)->dofmap()->cell_dofs(cell->index());

    f_g.add_local(RHS_e.data(), nrowsS, cdof_rowsS.data());
    Qe_list[cell->index()] = Q_e;
  }
  f_g.apply("add");
}
//-----------------------------------------------------------------------------
void StokesStaticCondensation::assemble_global_system(bool assemble_lhs)
{
  // Reassembles global system (lhs and rhs) and preserves symmetry
  // useful when using iterative solver (hopefully...), but note that it
  // requires the global system to be reassembled every time

  // You may skip repeated assemblage of A_e (when desired)
  if (assemble_lhs)
    A_g.zero();
  f_g.zero();

  // Collect bcs info, see dolfin::SystemAssembler
  bool active_bcs = (!bcs.empty());

  std::vector<DirichletBC::Map> boundary_values(1);
  if (active_bcs)
  {
    // Bin boundary conditions according to which form they apply to (if any)
    for (std::size_t i = 0; i < bcs.size(); ++i)
    {
      bcs[i]->get_boundary_values(boundary_values[0]);
      if (MPI::size(mpi_comm) > 1 && bcs[i]->method() != "pointwise")
        bcs[i]->gather(boundary_values[0]);
    }
  }

  for (CellIterator cell(*(this->mesh)); !cell.end(); ++cell)
  {
    // NOTE, You do not need nrowsS, ncolsS -- coincide with B,
    // check in constructor
    std::size_t nrowsA, ncolsA, nrowsB, ncolsB, nrowsG, ncolsG;

    std::tie(nrowsA, ncolsA) = FormUtils::local_tensor_size(*(this->A), *cell);
    std::tie(nrowsB, ncolsB) = FormUtils::local_tensor_size(*(this->B), *cell);
    std::tie(nrowsG, ncolsG) = FormUtils::local_tensor_size(*(this->G), *cell);

    // Assemble LHS (if needed)
    if (assemble_lhs)
    {
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          A_e, B_e, G_e;
      FormUtils::local_assembler(A_e, *(this->A), *cell, nrowsA, ncolsA);
      FormUtils::local_assembler(G_e, *(this->G), *cell, nrowsG, ncolsG);
      FormUtils::local_assembler(B_e, *(this->B), *cell, nrowsB, ncolsB);

      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          invA_e = A_e.inverse();
      invAe_list[cell->index()] = invA_e;
      Ge_list[cell->index()] = G_e;
      Be_list[cell->index()] = B_e;

      if (!assume_symmetric)
      {
        // No symmetry assumption
        std::size_t nrowsGT, ncolsGT;

        std::tie(nrowsGT, ncolsGT)
            = FormUtils::local_tensor_size(*(this->GT), *cell);

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            GT_e;
        FormUtils::local_assembler(GT_e, *(this->GT), *cell, nrowsGT, ncolsGT);
        GTe_list[cell->index()] = GT_e;
      }
    }

    // Assemble RHS, note: Q_e and S_e are actually vectors
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Q_e,
        S_e;

    FormUtils::local_assembler(Q_e, *(this->Q), *cell, nrowsA, 1);
    FormUtils::local_assembler(S_e, *(this->S), *cell, nrowsB, 1);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        LHS_e;
    Eigen::Matrix<double, Eigen::Dynamic, 1> RHS_e;

    if (assume_symmetric)
    {
      LHS_e = Ge_list[cell->index()].transpose() * invAe_list[cell->index()]
                  * Ge_list[cell->index()]
              - Be_list[cell->index()];
      RHS_e
          = Ge_list[cell->index()].transpose() * invAe_list[cell->index()] * Q_e
            - S_e;
    }
    else
    {
      LHS_e = GTe_list[cell->index()] * invAe_list[cell->index()]
                  * Ge_list[cell->index()]
              - Be_list[cell->index()];
      RHS_e = GTe_list[cell->index()] * invAe_list[cell->index()] * Q_e - S_e;
    }

    auto cdof_rowsB
        = this->B->function_space(0)->dofmap()->cell_dofs(cell->index());
    auto cdof_colsB
        = this->B->function_space(1)->dofmap()->cell_dofs(cell->index());

    // Apply BC's here (maintaining symmetry)
    if (active_bcs)
    {
      FormUtils::apply_boundary_symmetric(LHS_e, RHS_e, cdof_rowsB, cdof_colsB,
                                          boundary_values, active_bcs);
    }

    // Add to tensor if reassembling LHS
    if (assemble_lhs)
    {
      A_g.add_local(LHS_e.data(), nrowsB, cdof_rowsB.data(), ncolsB,
                    cdof_colsB.data());
    }

    // Add to vector
    f_g.add_local(RHS_e.data(), nrowsB, cdof_rowsB.data());
    Qe_list[cell->index()] = Q_e;
  }
  A_g.apply("add");
  f_g.apply("add");
}

//-----------------------------------------------------------------------------
void StokesStaticCondensation::solve_problem(Function& Uglobal,
                                             Function& Ulocal,
                                             const std::string solver,
                                             const std::string preconditioner)
{
  // TODO: Check if Uglobal, Ulocal are correct
  // Solve global system

  if (solver == "none")
  {
    // Direct solver
    solve(A_g, *(Uglobal.vector()), f_g);
  }
  else
  {
    // Iterative solver
    //    std::size_t num_it =
    solve(A_g, *(Uglobal.vector()), f_g, solver, preconditioner);
    // if(MPI::rank(mpi_comm) == 0) std::cout<<"Number of
    // iterations"<<num_it<<std::endl;
  }
  // Backsubtitution in Ulocal
  backsubtitute(Uglobal, Ulocal);
}

void StokesStaticCondensation::apply_boundary(DirichletBC& DBC)
{
  DBC.apply(A_g, f_g);
  if (MPI::size(mpi_comm) == 1)
    std::cout << "Matrix symmetry after apply_boundary? "
              << A_g.is_symmetric(1E-6) << std::endl;
}
//-----------------------------------------------------------------------------
void StokesStaticCondensation::backsubtitute(const Function& Uglobal,
                                             Function& Ulocal)
{
  for (CellIterator cell(*(this->mesh)); !cell.end(); ++cell)
  {
    std::size_t nrowsQ, ncolsQ, nrowsS, ncolsS;
    std::tie(nrowsQ, ncolsQ) = FormUtils::local_tensor_size(*(this->Q), *cell);
    std::tie(nrowsS, ncolsS) = FormUtils::local_tensor_size(*(this->S), *cell);

    // Alternatively, use B and A, rexpectively
    auto cdof_rowsS
        = this->S->function_space(0)->dofmap()->cell_dofs(cell->index());
    auto cdof_rowsQ
        = this->Q->function_space(0)->dofmap()->cell_dofs(cell->index());

    Eigen::Matrix<double, Eigen::Dynamic, 1> Uglobal_e, Ulocal_e;
    Uglobal_e.resize(nrowsS);

    Uglobal.vector()->get_local(Uglobal_e.data(), nrowsS, cdof_rowsS.data());
    Ulocal_e = invAe_list[cell->index()]
               * (Qe_list[cell->index()] - Ge_list[cell->index()] * Uglobal_e);
    Ulocal.vector()->set_local(Ulocal_e.data(), Ulocal_e.size(),
                               cdof_rowsQ.data());
  }
  Ulocal.vector()->apply("insert");
}
//-----------------------------------------------------------------------------
void StokesStaticCondensation::test_rank(const Form& a, const std::size_t rank)
{
  if (a.rank() != rank)
    dolfin_error("StokesStaticCondensation::test_rank", "get correct rank",
                 "Proper forms specified?");
}
//-----------------------------------------------------------------------------
