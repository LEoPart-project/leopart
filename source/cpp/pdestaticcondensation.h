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
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>

namespace dolfin
{
// Forward declarations
class Form;
class Mesh;
class particles;
class Function;
class DirichletBC;

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
  PDEStaticCondensation(std::shared_ptr<const Mesh> mesh, particles& P, std::shared_ptr<const Form> N,
                        std::shared_ptr<const Form> G, std::shared_ptr<const Form> L, std::shared_ptr<const Form> H,
                        std::shared_ptr<const Form> B, std::shared_ptr<const Form> Q, std::shared_ptr<const Form> R,
                        std::shared_ptr<const Form> S, const std::size_t idx_pproperty);

  // Constructor including Dirichlet BC's
  PDEStaticCondensation(std::shared_ptr<const Mesh> mesh, particles& P, std::shared_ptr<const Form> N,
                        std::shared_ptr<const Form> G, std::shared_ptr<const Form> L, std::shared_ptr<const Form> H,
                        std::shared_ptr<const Form> B, std::shared_ptr<const Form> Q, std::shared_ptr<const Form> R,
                        std::shared_ptr<const Form> S,
                        std::vector<std::shared_ptr<const DirichletBC>> bcs,
                        const std::size_t idx_pproperty);

  ~PDEStaticCondensation();

  // TO DO: assemble_on_config labels the rhs assembly
  void assemble(const bool assemble_all = true,
                const bool assemble_on_config = true);
  void assemble_state_rhs();

  void solve_problem(Function& Uglobal, Function& Ulocal,
                     const std::string solver = "none",
                     const std::string preconditioner = "default");
  void solve_problem(Function& Uglobal, Function& Ulocal, Function& Lambda,
                     const std::string solver = "none",
                     const std::string preconditioner = "default");
  void apply_boundary(DirichletBC& DBC);

private:
  // Private Methods
  void backsubtitute(const Function& Uglobal, Function& Ulocal);
  void backsubtitute(const Function& Uglobal, Function& Ulocal,
                     Function& Lambda);

  /* Comes from particles
  void get_particle_contributions(Eigen::Matrix<double, Eigen::Dynamic,
  Eigen::Dynamic, Eigen::RowMajor>& N_ep, Eigen::Matrix<double, Eigen::Dynamic,
  Eigen::Dynamic, Eigen::RowMajor>& R_ep, const Cell& dolfin_cell);
  */

  // Private Attributes
  std::shared_ptr<const Mesh> mesh;
  particles* _P;
  std::shared_ptr<const Form> N, G, L, H, B, Q, R, S;

  const MPI_Comm mpi_comm;
  Matrix A_g;
  Vector f_g;

  std::shared_ptr<const FiniteElement> _element;
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

  std::vector<std::shared_ptr<const DirichletBC>> bcs;

  // FIXME needed for momentum based l2 map
  std::shared_ptr<Function> rhoh;
};
} // namespace dolfin
#endif // PDESTATICCONDENSATION_H
