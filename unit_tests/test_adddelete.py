# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import UnitSquareMesh, FunctionSpace, Function, Expression, Point, sqrt, assemble, dx
from leopart import particles, AddDelete, RandomRectangle, l2projection
from mpi4py import MPI as pyMPI
import numpy as np

comm = pyMPI.COMM_WORLD


def assign_particle_values(x, u_exact):
    if comm.Get_rank() == 0:
        s = np.asarray([u_exact(x[i, :]) for i in range(len(x))], dtype=np.float_)
    else:
        s = None
    return s


# TODO: weighted sweeep
# TODO: test in 3d


def test_add_particles():
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
