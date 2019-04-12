// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifndef PDESTATICCONDENSATION_H
#define PDESTATICCONDENSATION_H

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include <dolfin/common/MPI.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>

namespace dolfin
{
// Forward declarations

namespace fem
{
class Form;
class DirichletBC;
} // namespace fem

namespace mesh
{
class Mesh;
}

class particles;

namespace function
{
class Function;
}

class PDEStaticCondensation
{
  // Class providing functionality for PDE constrained projection using static
  // condensation. Expect the dolfin Forms to comply with the algebraic form:
  //
  //    |  N   G   L | | psi     |    |  Q  |
  //    |  G^T 0   H | | lambda  | =  |  R  |
  //    |  L^T H^T B | | Psi_bar |    |  S  |
  //
public:
  // Constructor
  PDEStaticCondensation(const mesh::Mesh& mesh, particles& P,
                        const fem::Form& N, const fem::Form& G,
                        const fem::Form& L, const fem::Form& H,
                        const fem::Form& B, const fem::Form& Q,
                        const fem::Form& R, const fem::Form& S,
                        const std::size_t idx_pproperty);

  // Constructor including Dirichlet BC's
  PDEStaticCondensation(
      const mesh::Mesh& mesh, particles& P, const fem::Form& N,
      const fem::Form& G, const fem::Form& L, const fem::Form& H,
      const fem::Form& B, const fem::Form& Q, const fem::Form& R,
      const fem::Form& S,
      std::vector<std::shared_ptr<const fem::DirichletBC>> bcs,
      const std::size_t idx_pproperty);

  ~PDEStaticCondensation();

  // TO DO: assemble_on_config labels the rhs assembly
  void assemble(const bool assemble_all = true,
                const bool assemble_on_config = true);
  void assemble_state_rhs();

  void solve_problem(function::Function& Uglobal, function::Function& Ulocal,
                     const std::string solver = "none",
                     const std::string preconditioner = "default");
  void solve_problem(function::Function& Uglobal, function::Function& Ulocal,
                     function::Function& Lambda,
                     const std::string solver = "none",
                     const std::string preconditioner = "default");
  void apply_boundary(fem::DirichletBC& DBC);

private:
  // Private Methods
  void backsubtitute(const function::Function& Uglobal,
                     function::Function& Ulocal);
  void backsubtitute(const function::Function& Uglobal,
                     function::Function& Ulocal, function::Function& Lambda);

  /* Comes from particles
  void get_particle_contributions(Eigen::Matrix<double, Eigen::Dynamic,
  Eigen::Dynamic, Eigen::RowMajor>& N_ep, Eigen::Matrix<double, Eigen::Dynamic,
  Eigen::Dynamic, Eigen::RowMajor>& R_ep, const Cell& dolfin_cell);
  */

  // Private Attributes
  const mesh::Mesh* mesh;
  particles* _P;
  const fem::Form *N, *G, *L, *H, *B, *Q, *R, *S;

  const MPI_Comm mpi_comm;
  la::PETScMatrix A_g;
  la::PETScVector f_g;

  std::shared_ptr<const fem::FiniteElement> _element;
  std::size_t _num_subspaces, _space_dimension, _num_dof_locs, _value_size_loc;

  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      invKS_list;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      LHe_list;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      Ge_list;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      Be_list;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> Re_list, QRe_list;

  // TODO: set _idx_pproperty
  const std::size_t _idx_pproperty;

  std::vector<std::shared_ptr<const fem::DirichletBC>> bcs;

  // FIXME needed for momentum based l2 map
  function::Function* rhoh;
};
} // namespace dolfin
#endif // PDESTATICCONDENSATION_H
