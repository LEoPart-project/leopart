# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import (UserExpression, Expression, Point, BoxMesh, Function, FunctionSpace,
                    VectorFunctionSpace, assemble, dx, dot)
from leopart import (particles, l2projection,
                     RandomBox, RegularBox, AddDelete, assign_particle_values)
from mpi4py import MPI as pyMPI
import numpy as np
import pytest

comm = pyMPI.COMM_WORLD


class Ball(UserExpression):
    def __init__(self, radius, center, lb=0., ub=1., **kwargs):
        assert len(center) == 3
        self.r = radius
        self.center = center
        self.lb = lb
        self.ub = ub
        super().__init__(self, **kwargs)

    def eval(self, value, x):
        (xc, yc, zc) = self.center

        if (x[0] - xc)**2 + (x[1] - yc)**2 + (x[2] - zc)**2 <= self.r**2:
            value[0] = self.ub
        else:
            value[0] = self.lb

    def value_shape(self):
        return ()


@pytest.mark.parametrize('polynomial_order, in_expression', [(2, " pow(x[0],2) + pow(x[1],2)")])
def test_l2_projection_3D(polynomial_order, in_expression):
    xmin, ymin, zmin = 0., 0., 0.
    xmax, ymax, zmax = 1., 1., 1.
    nx = 25

    property_idx = 1
    mesh = BoxMesh(Point(xmin, ymin, zmin), Point(xmax, ymax, zmax), nx, nx, nx)

    interpolate_expression = Expression(in_expression, degree=3)

    if len(interpolate_expression.ufl_shape) == 0:
        V = FunctionSpace(mesh, "DG", polynomial_order)
    elif len(interpolate_expression.ufl_shape) == 1:
        V = VectorFunctionSpace(mesh, "DG", polynomial_order)

    v_exact = Function(V)
    v_exact.assign(interpolate_expression)

    x = RandomBox(Point(0., 0., 0.), Point(1., 1., 1.)).generate([4, 4, 4])
    s = assign_particle_values(x, interpolate_expression)

    # Just make a complicated particle, possibly with scalars and vectors mixed
    p = particles(x, [s], mesh)

    # Do AddDelete sweep
    AD = AddDelete(p, 13, 15, [v_exact])
    AD.do_sweep()

    vh = Function(V)
    lstsq_vh = l2projection(p, V, property_idx)
    lstsq_vh.project(vh.cpp_object())

    error_sq = abs(assemble(dot(v_exact - vh, v_exact - vh)*dx))
    if comm.Get_rank() == 0:
        assert error_sq < 1e-13


@pytest.mark.parametrize('polynomial_order, lb, ub', [(1, -3., -1.)])
def test_l2projection_bounded_3D(polynomial_order, lb, ub):
    xmin, ymin, zmin = 0., 0., 0.
    xmax, ymax, zmax = 1., 1., 1.
    nx = 10

    interpolate_expression = Ball(0.15, [0.5, 0.5, 0.5], degree=3, lb=lb, ub=ub)

    property_idx = 1
    mesh = BoxMesh(Point(xmin, ymin, zmin), Point(xmax, ymax, zmax), nx, nx, nx)

    V = FunctionSpace(mesh, "DG", polynomial_order)

    x = RegularBox(Point(0., 0., 0.), Point(1., 1., 1.)).generate([100, 100, 100])
    s = assign_particle_values(x, interpolate_expression)

    # Just make a complicated particle, possibly with scalars and vectors mixed
    p = particles(x, [s], mesh)

    vh = Function(V)
    lstsq_rho = l2projection(p, V, property_idx)
    lstsq_rho.project(vh.cpp_object(), lb, ub)

    # Assert if it stays within bounds
    assert np.any(vh.vector().get_local() < ub + 1e-12)
    assert np.any(vh.vector().get_local() > lb - 1e-12)

# TODO: Vector Function L2
# TODO: PDE constrained projection
