// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifndef STOKESSTATICCONDENSATION_H
#define STOKESSTATICCONDENSATION_H

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>

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

namespace function
{
class Function;
}

class StokesStaticCondensation
{
public:
  // Constructors with assumed symmetry
  StokesStaticCondensation(const mesh::Mesh& mesh, const fem::Form& A,
                           const fem::Form& G, const fem::Form& B,
                           const fem::Form& Q, const fem::Form& S);
  StokesStaticCondensation(
      const mesh::Mesh& mesh, const fem::Form& A, const fem::Form& G,
      const fem::Form& B, const fem::Form& Q, const fem::Form& S,
      std::vector<std::shared_ptr<const fem::DirichletBC>> bcs);
  // Constructors assuming full [2x2] block specification
  StokesStaticCondensation(const mesh::Mesh& mesh, const fem::Form& A,
                           const fem::Form& G, const fem::Form& GT,
                           const fem::Form& B, const fem::Form& Q,
                           const fem::Form& S);

  StokesStaticCondensation(
      const mesh::Mesh& mesh, const fem::Form& A, const fem::Form& G,
      const fem::Form& GT, const fem::Form& B, const fem::Form& Q,
      const fem::Form& S,
      std::vector<std::shared_ptr<const fem::DirichletBC>> bcs);

  // Destructor
  ~StokesStaticCondensation();

  // Public Methods
  void assemble_global();
  void assemble_global_lhs();
  void assemble_global_rhs();
  void assemble_global_system(bool assemble_lhs = true);

  void apply_boundary(fem::DirichletBC& DBC);
  void solve_problem(function::Function& Uglobal, function::Function& Ulocal,
                     const std::string solver = "none",
                     const std::string preconditioner = "default");

private:
  // Private Methods
  void backsubtitute(const function::Function& Uglobal,
                     function::Function& Ulocal);
  void test_rank(const fem::Form& a, const std::size_t rank);

  // Private Attributes
  const mesh::Mesh* mesh;
  const fem::Form* A;
  const fem::Form* B;
  const fem::Form* G;
  const fem::Form* Q;
  const fem::Form* S;
  const fem::Form* GT;

  bool assume_symmetric;

  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      invAe_list;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      Ge_list;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      Be_list;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> Qe_list;

  // Facilitate non-symmetric case
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      GTe_list;

  const MPI_Comm mpi_comm;
  la::PETScMatrix A_g;
  la::PETScVector f_g;
  std::vector<std::shared_ptr<const fem::DirichletBC>> bcs;
};
} // namespace dolfin

#endif // STOKESSTATICCONDENSATION_H
