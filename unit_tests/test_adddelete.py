# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import (
    UnitSquareMesh,
    UnitCubeMesh,
    FunctionSpace,
    VectorFunctionSpace,
    Function,
    Expression,
    Point,
    sqrt,
    assemble,
    dx,
)
from leopart import particles, AddDelete, RandomRectangle, l2projection, advect_particles, RandomBox
from mpi4py import MPI as pyMPI
import numpy as np
import pytest

comm = pyMPI.COMM_WORLD


def assign_particle_values(x, u_exact):
    if comm.Get_rank() == 0:
        s = np.asarray([u_exact(x[i, :]) for i in range(len(x))], dtype=np.float_)
    else:
        s = None
    return s


# TODO: weighted sweeep
# TODO: test in 3d


def test_initialize_particles():
    interpolate_expression = Expression("x[0]", degree=1)
    mesh = UnitSquareMesh(5, 5)
    V = FunctionSpace(mesh, "DG", 1)

    v = Function(V)
    v.assign(interpolate_expression)

    np_min, np_max = 5, 10

    # Initialize particles
    x = RandomRectangle(Point(0.0, 0.0), Point(1.0, 1.0)).generate([1, 1])
    s = assign_particle_values(x, interpolate_expression)
    # Broadcast to other procs
    x = comm.bcast(x, root=0)
    s = comm.bcast(s, root=0)

    property_idx = 1
    p = particles(x, [s], mesh)
    AD = AddDelete(p, np_min, np_max, [v])
    AD.do_sweep()

    # Must recover linear
    lstsq_rho = l2projection(p, V, property_idx)
    lstsq_rho.project(v.cpp_object())

    error = sqrt(assemble((v - interpolate_expression) * (v - interpolate_expression) * dx))

    assert len(p.positions() == mesh.num_cells() * np_min)
    assert error < 1e-12


def test_remove_particles():
    interpolate_expression = Expression("x[0]", degree=1)
    mesh = UnitSquareMesh(5, 5)
    V = FunctionSpace(mesh, "DG", 1)

    v = Function(V)
    v.assign(interpolate_expression)

    np_min, np_max = 5, 10

    # Initialize particles
    x = RandomRectangle(Point(0.0, 0.0), Point(1.0, 1.0)).generate([100, 100])
    s = assign_particle_values(x, interpolate_expression)
    # Broadcast to other procs
    x = comm.bcast(x, root=0)
    s = comm.bcast(s, root=0)

    property_idx = 1
    p = particles(x, [s], mesh)
    AD = AddDelete(p, np_min, np_max, [v])
    AD.do_sweep()

    # Must recover linear
    lstsq_rho = l2projection(p, V, property_idx)
    lstsq_rho.project(v.cpp_object())

    error = sqrt(assemble((v - interpolate_expression) * (v - interpolate_expression) * dx))

    assert len(p.positions() == mesh.num_cells() * np_min)
    assert error < 1e-12


def test_failsafe_sweep():
    interpolate_expression = Expression("x[0]", degree=1)
    mesh = UnitSquareMesh(5, 5)
    V = FunctionSpace(mesh, "DG", 1)

    v = Function(V)
    v.assign(interpolate_expression)

    np_min, np_max = 1, 2
    np_failsafe = 4

    # Initialize particles
    x = RandomRectangle(Point(0.0, 0.0), Point(1.0, 1.0)).generate([100, 100])
    s = assign_particle_values(x, interpolate_expression)
    # Broadcast to other procs
    x = comm.bcast(x, root=0)
    s = comm.bcast(s, root=0)

    property_idx = 1
    p = particles(x, [s], mesh)
    AD = AddDelete(p, np_min, np_max, [v])
    AD.do_sweep_failsafe(np_failsafe)

    # Must recover linear
    lstsq_rho = l2projection(p, V, property_idx)
    lstsq_rho.project(v.cpp_object())

    error = sqrt(assemble((v - interpolate_expression) * (v - interpolate_expression) * dx))

    assert len(p.positions() == mesh.num_cells() * np_failsafe)
    assert error < 1e-12


@pytest.mark.parametrize("mesh_dimension", [2, 3])
def test_add_particles(mesh_dimension):
    interpolate_expression = Expression("5*x[0]", degree=1)
    if mesh_dimension == 2:
        mesh = UnitSquareMesh(5, 5)
    else:
        mesh = UnitCubeMesh(5, 5, 5)
    V = FunctionSpace(mesh, "DG", 1)

    v = Function(V)
    v.assign(interpolate_expression)

    # Initialize particles
    if mesh_dimension == 2:
        x = RandomRectangle(Point(0.0, 0.0), Point(1.0, 1.0)).generate([100, 100])
    else:
        x = RandomBox(Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)).generate([50, 50, 50])
    s = assign_particle_values(x, interpolate_expression)

    # Broadcast to other procs and split into pieces
    x = comm.bcast(x, root=0)
    x_0 = x[0:200, :]
    x_1 = x[200:500, :]
    x_2 = x[500::, :]

    s = comm.bcast(s, root=0)
    s_0 = s[0:200]
    s_1 = s[200:500]
    s_2 = s[500::]

    p = particles(x_0, [s_0, 2 * x_0], mesh)

    assert p.number_of_particles() == x_0.shape[0]
    # Check scalar valued property
    np.testing.assert_array_almost_equal(
        5 * np.array(p.get_property(0)).reshape(-1, mesh_dimension)[:, 0],
        np.array(p.get_property(1)),
    )
    # Check vector valued property
    np.testing.assert_array_almost_equal(
        2 * np.array(p.get_property(0)), np.array(p.get_property(2))
    )

    p.add_particles(x_1, [s_1, 2 * x_1])
    assert p.number_of_particles() == x_0.shape[0] + x_1.shape[0]
    # Check scalar valued property after addition
    np.testing.assert_array_almost_equal(
        5 * np.array(p.get_property(0)).reshape(-1, mesh_dimension)[:, 0],
        np.array(p.get_property(1))
    )
    # Check vector valued property after addition
    np.testing.assert_array_almost_equal(
        2 * np.array(p.get_property(0)), np.array(p.get_property(2))
    )

    # Initializing an advection class changes the particle template by
    # adding entries for xp0 (position copies) and up0 (velocites at previous time step,
    # initialized with 0). Check whether that behaves properly
    V = VectorFunctionSpace(mesh, "CG", 1)
    v = Function(V)
    advect_particles(p, V, v, "closed")
    p.add_particles(x_2, [s_2, 2 * x_2])
    assert p.number_of_particles() == x.shape[0]

    # Check scalar valued property after addition
    np.testing.assert_array_almost_equal(
        5 * np.array(p.get_property(0)).reshape(-1, mesh_dimension)[:, 0],
        np.array(p.get_property(1)),
    )
    # Check vector valued property after addition
    np.testing.assert_array_almost_equal(
        2 * np.array(p.get_property(0)), np.array(p.get_property(2))
    )

    np.testing.assert_array_almost_equal(np.array(p.get_property(3)), np.array(p.get_property(0)))
    np.testing.assert_array_almost_equal(
        np.array(p.get_property(4)), np.zeros(len(p.get_property(4)))
    )
