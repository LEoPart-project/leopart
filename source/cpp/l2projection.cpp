// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <iostream>
#include <memory>

#include <dolfin/fem/Form.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/utils.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "QuadProg++.hh"
#include "particles.h"

#include "l2projection.h"

using namespace dolfin;

l2projection::l2projection(particles& P, function::FunctionSpace& V,
                           const std::size_t idx)
    : _P(&P), _element(V.element()), _dofmap(V.dofmap()), _idx_pproperty(idx)
{
  // Put an assertion here: we need to have a DG function space at the moment
  _num_subspaces = _element->num_sub_elements();
  _space_dimension = _element->space_dimension();

  if (_num_subspaces == 0)
    _num_dof_locs = _space_dimension;
  else
    _num_dof_locs = _space_dimension / _num_subspaces;

  _value_size_loc = 1;
  for (std::size_t i = 0; i < _element->value_rank(); i++)
    _value_size_loc *= _element->value_dimension(i);

  // Check if matches with stored particle template
  if (_value_size_loc != _P->ptemplate(_idx_pproperty))
    throw std::runtime_error("l2projection.cpp"
                             "Cannot set _value_size_loc. "
                             "Local value size ("
                             + std::to_string(_value_size_loc)
                             + ") mismatches particle template property "
                               "with size ("
                             + std::to_string(_P->ptemplate(_idx_pproperty))
                             + ")");
}
//-----------------------------------------------------------------------------
l2projection::~l2projection() {}
//-----------------------------------------------------------------------------
void l2projection::project(function::Function& u)
{
  // TODO: Check if u is indeed in V!
  // Initialize basis matrix as CONTIGUOUS array, Maybe consider using
  // std::array to make easier conversion to Eigen::Matrix? double
  // basis_matrix[_space_dimension][_value_size_loc];

  la::VecWrapper v(u.vector().vec());

  // TODO: new and compact formulation. WORK IN PROGRESS!
  std::int32_t num_cells
      = _P->mesh()->num_entities(_P->mesh()->topology().dim());

  for (std::int32_t i = 0; i < num_cells; ++i)
  {
    mesh::Cell cell(*_P->mesh(), i);

    // Get dofs local to cell
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> celldofs
        = _dofmap->cell_dofs(i);

    // Initialize the cell matrix and cell vector
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q;
    Eigen::Matrix<double, Eigen::Dynamic, 1> f;

    // Get particle contributions
    _P->get_particle_contributions(q, f, cell, _element, _space_dimension,
                                   _value_size_loc, _idx_pproperty);

    // Initialize and solve for u_i
    Eigen::Matrix<double, Eigen::Dynamic, 1> u_i;
    if (_P->num_cell_particles(i) >= _num_dof_locs)
    {
      // Overdetermined system, use normal equations (fast!)
      u_i = (q * (q.transpose())).ldlt().solve(q * f);
    }
    else
    {
      // Underdetermined system, use Jacobi (slower, but more robust)
      std::cout << "Underdetermined system in cell " << i
                << ". Using Jacobie solve" << std::endl;
      u_i = (q.transpose())
                .jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                .solve(f);
    }

    for (int j = 0; j < u_i.size(); ++j)
    {
      v.x[celldofs[j]] = u_i[j];
    }

    // Insert in vector
    // FIXME    u.vector().set_local(u_i.data(), u_i.size(), celldofs.data());
  }
}
//-----------------------------------------------------------------------------
void l2projection::project(function::Function& u, const double lb,
                           const double ub)
{
  // Check if u is indeed in V!!!!
  if (_value_size_loc > 1)
    throw std::runtime_error(
        "l2projection.cpp::project. "
        "Cannot handle value size >1. "
        "Bounded projection is implemented for scalar functions only");

  // Initialize the matrices/vectors for the bound constraints (constant
  // throughout projection)
  Eigen::MatrixXd CE, CI;
  Eigen::VectorXd ce0, ci0;

  CE.resize(_space_dimension, 0);
  ce0.resize(0);

  CI.resize(_space_dimension, _space_dimension * _value_size_loc * 2);
  CI.setZero();
  ci0.resize(_space_dimension * _value_size_loc * 2);
  ci0.setZero();
  for (std::size_t i = 0; i < _space_dimension; i++)
  {
    CI(i, i) = 1.;
    CI(i, i + _space_dimension) = -1;
    ci0(i) = -lb;
    ci0(i + _space_dimension) = ub;
  }

  la::VecWrapper v(u.vector().vec());
  std::int32_t num_cells
      = _P->mesh()->num_entities(_P->mesh()->topology().dim());
  for (std::int32_t i = 0; i < num_cells; ++i)
  {
    mesh::Cell cell(*_P->mesh(), i);

    // Get dofs local to cell
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> celldofs
        = _dofmap->cell_dofs(i);

    // Initialize the cell matrix and cell vector
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q;
    Eigen::Matrix<double, Eigen::Dynamic, 1> f;

    // Get particle contributions
    _P->get_particle_contributions(q, f, cell, _element, _space_dimension,
                                   _value_size_loc, _idx_pproperty);

    // Then solve bounded lstsq projection
    Eigen::MatrixXd AtA = q * (q.transpose());
    Eigen::VectorXd Atf = -q * f;
    Eigen::VectorXd u_i;
    quadprogpp::solve_quadprog(AtA, Atf, CE, ce0, CI, ci0, u_i);

    for (int j = 0; j < u_i.size(); ++j)
      v.x[celldofs[j]] = u_i[j];

    // FIXME    u.vector().set_local(u_i.data(), u_i.size(), celldofs.data());
  }
}
//-----------------------------------------------------------------------------
void l2projection::project_cg(const fem::Form& A, const fem::Form& f,
                              function::Function& u)
{
  // Initialize global problems
  // FIXME: need some checks!

  la::PETScMatrix A_g = fem::create_matrix(A);
  la::PETScVector f_g(*(f.function_space(0)->dofmap()->index_map()));

  //  AssemblerBase assembler_base;
  //  assembler_base.init_global_tensor(A_g, A);
  //  assembler_base.init_global_tensor(f_g, f);

  // Set to zero
  // FIXME  A_g.zero();
  //  f_g.zero();

  la::VecWrapper fv(f_g.vec());
  const std::int32_t num_cells
      = _P->mesh()->num_entities(_P->mesh()->topology().dim());
  for (std::int32_t i = 0; i < num_cells; ++i)
  {
    mesh::Cell cell(*_P->mesh(), i);

    // Get dofs local to cell
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> celldofs
        = _dofmap->cell_dofs(i);

    // Initialize the cell matrix and cell vector
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q;
    Eigen::Matrix<double, Eigen::Dynamic, 1> f;

    // Get particle contributions
    _P->get_particle_contributions(q, f, cell, _element, _space_dimension,
                                   _value_size_loc, _idx_pproperty);

    // Compute lstsq contributions
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> qTq;
    Eigen::Matrix<double, Eigen::Dynamic, 1> qTf;
    qTq = q * q.transpose();
    qTf = q * f;

    // Place in matrix/vector --> Check!
    A_g.add_local(qTq.data(), celldofs.size(), celldofs.data(), celldofs.size(),
                  celldofs.data());
    for (int j = 0; j < celldofs.size(); ++j)
      fv.x[celldofs[j]] += qTf[j];

    // f_g.add_local(qTf.data(), celldofs.size(), celldofs.data());
  }

  //  A_g.apply("add");
  //  f_g.apply("add");

  // solve(A_g, *(u.vector()), f_g);
}
//-----------------------------------------------------------------------------
