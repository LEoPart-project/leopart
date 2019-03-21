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
  class Form;
  class Mesh;
  class DirichletBC;
  class Function;


  class StokesStaticCondensation
{
public:
  // Constructors with assumed symmetry
  StokesStaticCondensation(const Mesh& mesh, const Form& A, const Form& G,
                           const Form& B, const Form& Q, const Form& S);
  StokesStaticCondensation(const Mesh& mesh, const Form& A, const Form& G,
                           const Form& B, const Form& Q, const Form& S,
                           std::vector<std::shared_ptr<const DirichletBC>> bcs);
  // Constructors assuming full [2x2] block specification
  StokesStaticCondensation(const Mesh& mesh, const Form& A, const Form& G,
                           const Form& GT, const Form& B, const Form& Q,
                           const Form& S);

  StokesStaticCondensation(const Mesh& mesh, const Form& A, const Form& G,
                           const Form& GT, const Form& B, const Form& Q,
                           const Form& S,
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
                     const std::string solver = "none",
                     const std::string preconditioner = "default");

private:
  // Private Methods
  void backsubtitute(const Function& Uglobal, Function& Ulocal);
  void test_rank(const Form& a, const std::size_t rank);

  // Private Attributes
  const Mesh* mesh;
  const Form* A;
  const Form* B;
  const Form* G;
  const Form* Q;
  const Form* S;
  const Form* GT;

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
  Matrix A_g;
  Vector f_g;
  std::vector<std::shared_ptr<const DirichletBC>> bcs;
};
} // namespace dolfin

#endif // STOKESSTATICCONDENSATION_H
