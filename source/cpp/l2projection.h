// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com

#ifndef L2PROJECTION_H
#define L2PROJECTION_H

#include <iostream>
#include <particles.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/common/ArrayView.h>
#include <dolfin/fem/GenericDofMap.h>

#include <dolfin/fem/Assembler.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include "dolfin/la/solve.h"

#include <eigquadprog.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Dense>

namespace dolfin{
    class l2projection
    {
    public:
        l2projection(particles& P, FunctionSpace& V, const std::size_t idx);
        ~l2projection();

        // l^2 map
        void project(Function& u);

        // l^2 map with bound constraint
        void project(Function &u, const double lb, const double ub);

        // l^2 map onto cg space (global problem)
        void project_cg(const Form& A, const Form& f, Function& u);

    protected:
        particles* _P;     // Pointer object to particles

        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > _Nixp;

        std::shared_ptr<const GenericDofMap> _dofmap;
        std::shared_ptr<const FiniteElement> _element;
        std::size_t _num_subspaces, _space_dimension, _num_dof_locs,
                    _value_size_loc;

        // Workaround to access tuple elements
        // FIXME: what happens if we make this a constant?
        std::size_t _idx_pproperty;

    };
}
#endif // L2PROJECTION_H
