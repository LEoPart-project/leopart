# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Unit tests for advection of single particle
"""

from dolfin import (
    SubDomain,
    UnitSquareMesh,
    RectangleMesh,
    Point,
    Constant,
    Expression,
    VectorFunctionSpace,
    Function,
    MeshFunction,
    near,
)
from leopart import particles, advect_particles, advect_rk2, advect_rk3, RandomRectangle
from mpi4py import MPI as pyMPI
import numpy as np
import pytest

comm = pyMPI.COMM_WORLD


class Boundaries(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class UnitSquareLeft(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)


class UnitSquareRight(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1)


class UnitSquareBottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0)


class UnitSquareTop(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1)


def compute_convergence(iterator, errorlist):
    assert len(iterator) == len(errorlist), "Iterator list and error list not of same length"
    alpha_list = []
    for i in range(len(iterator) - 1):
        conv_rate = np.log(errorlist[i + 1] / errorlist[i]) / np.log(iterator[i + 1] / iterator[i])
        alpha_list.append(conv_rate)
    return alpha_list


# Test accuracy of particle advection for euler, rk2 and rk3
@pytest.mark.parametrize("advection_scheme", ["euler", "rk2", "rk3"])
def test_advect_particle(advection_scheme):
    if comm.rank == 0:
        print("Run advect_particle")

    # Rotate one particle, and compute the error
    mesh = UnitSquareMesh(10, 10)

    # Particle
    x = np.array([[0.25, 0.25]])
    dt_list = [0.08, 0.04, 0.02, 0.01, 0.005]

    # Velocity field
    vexpr = Expression(("-pi*(x[1] - 0.5)", "pi*(x[0]-0.5)"), degree=3)
    V = VectorFunctionSpace(mesh, "CG", 1)
    v = Function(V)
    v.assign(vexpr)

    error_list = []

    for dt in dt_list:
        p = particles(x, [x, x], mesh)
        if advection_scheme == "euler":
            ap = advect_particles(p, V, v, "closed")
        elif advection_scheme == "rk2":
            ap = advect_rk2(p, V, v, "closed")
        elif advection_scheme == "rk3":
            ap = advect_rk3(p, V, v, "closed")
        else:
            assert False

        xp_0 = p.positions()
        t = 0.0
        while t < 2.0 - 1e-12:
            ap.do_step(dt)
            t += dt

        xp_end = p.positions()
        error_list.append(np.linalg.norm(xp_0 - xp_end))

    if not all(eps == 0 for eps in error_list):
        rate = compute_convergence(dt_list, error_list)
        if advection_scheme == "euler":
            # First order for euler
            assert any(i > 0.9 for i in rate)
        elif advection_scheme == "rk2":
            # Second order for rk2
            assert any(i > 1.95 for i in rate)
        elif advection_scheme == "rk3":
            # Third order for rk3
            assert any(i > 2.9 for i in rate)


# Test with all boundaries periodic
@pytest.mark.parametrize("advection_scheme", ["euler", "rk2", "rk3"])
def test_advect_periodic(advection_scheme):
    # FIXME: this unit test is sensitive to the ordering of the particle
    # array, i.e. xp0_root and xpE_root may contain exactly the same entries
    # but only in a different order. This will return an error right now

    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0
    pres = 3

    mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), 10, 10)

    lims = np.array(
        [
            [xmin, xmin, ymin, ymax],
            [xmax, xmax, ymin, ymax],
            [xmin, xmax, ymin, ymin],
            [xmin, xmax, ymax, ymax],
        ]
    )

    vexpr = Constant((1.0, 1.0))
    V = VectorFunctionSpace(mesh, "CG", 1)

    x = RandomRectangle(Point(0.05, 0.05), Point(0.15, 0.15)).generate([pres, pres])
    x = comm.bcast(x, root=0)
    dt = 0.05

    v = Function(V)
    v.assign(vexpr)

    p = particles(x, [x * 0, x ** 2], mesh)

    if advection_scheme == "euler":
        ap = advect_particles(p, V, v, "periodic", lims.flatten())
    elif advection_scheme == "rk2":
        ap = advect_rk2(p, V, v, "periodic", lims.flatten())
    elif advection_scheme == "rk3":
        ap = advect_rk3(p, V, v, "periodic", lims.flatten())
    else:
        assert False

    xp0 = p.positions()
    t = 0.0
    while t < 1.0 - 1e-12:
        ap.do_step(dt)
        t += dt
    xpE = p.positions()

    # Check if position correct
    xp0_root = comm.gather(xp0, root=0)
    xpE_root = comm.gather(xpE, root=0)

    num_particles = p.number_of_particles()

    if comm.Get_rank() == 0:
        xp0_root = np.float32(np.vstack(xp0_root))
        xpE_root = np.float32(np.vstack(xpE_root))

        # Sort on x positions
        xp0_root = xp0_root[xp0_root[:, 0].argsort(), :]
        xpE_root = xpE_root[xpE_root[:, 0].argsort(), :]

        error = np.linalg.norm(xp0_root - xpE_root)
        assert error < 1e-10
        assert num_particles - pres ** 2 == 0


# Same as previous, but other constructor
@pytest.mark.parametrize("advection_scheme", ["euler", "rk2", "rk3"])
def test_advect_periodic_facet_marker(advection_scheme):
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0

    mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), 10, 10)
    facet_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    facet_marker.set_all(0)
    boundaries = Boundaries()
    boundaries.mark(facet_marker, 3)

    lims = np.array(
        [
            [xmin, xmin, ymin, ymax],
            [xmax, xmax, ymin, ymax],
            [xmin, xmax, ymin, ymin],
            [xmin, xmax, ymax, ymax],
        ]
    )

    vexpr = Constant((1.0, 1.0))
    V = VectorFunctionSpace(mesh, "CG", 1)

    x = RandomRectangle(Point(0.05, 0.05), Point(0.15, 0.15)).generate([3, 3])
    x = comm.bcast(x, root=0)
    dt = 0.05

    v = Function(V)
    v.assign(vexpr)

    p = particles(x, [x * 0, x ** 2], mesh)

    if advection_scheme == "euler":
        ap = advect_particles(p, V, v, facet_marker, lims.flatten())
    elif advection_scheme == "rk2":
        ap = advect_rk2(p, V, v, facet_marker, lims.flatten())
    elif advection_scheme == "rk3":
        ap = advect_rk3(p, V, v, facet_marker, lims.flatten())
    else:
        assert False

    xp0 = p.positions()
    t = 0.0
    while t < 1.0 - 1e-12:
        ap.do_step(dt)
        t += dt
    xpE = p.positions()

    # Check if position correct
    xp0_root = comm.gather(xp0, root=0)
    xpE_root = comm.gather(xpE, root=0)

    if comm.Get_rank() == 0:
        xp0_root = np.float32(np.vstack(xp0_root))
        xpE_root = np.float32(np.vstack(xpE_root))
        error = np.linalg.norm(xp0_root - xpE_root)
        assert error < 1e-10


@pytest.mark.parametrize("advection_scheme", ["euler", "rk2", "rk3"])
def test_closed_boundary(advection_scheme):
    # FIXME: rk3 scheme does not bounces off the wall properly
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0

    mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), 10, 10)

    # Particle
    x = np.array([[0.975, 0.475]])

    # Given velocity field:
    vexpr = Constant((1.0, 0.0))
    # Given time do_step:
    dt = 0.05
    # Then bounced position is
    x_bounced = np.array([[0.975, 0.475]])

    p = particles(x, [x, x], mesh)

    V = VectorFunctionSpace(mesh, "CG", 1)
    v = Function(V)
    v.assign(vexpr)

    # Different boundary parts
    bound_left = UnitSquareLeft()
    bound_right = UnitSquareRight()
    bound_top = UnitSquareTop()
    bound_bottom = UnitSquareBottom()

    # Mark all facets
    facet_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    facet_marker.set_all(0)

    # Mark as closed
    bound_right.mark(facet_marker, 1)

    # Mark other boundaries as open
    bound_left.mark(facet_marker, 2)
    bound_top.mark(facet_marker, 2)
    bound_bottom.mark(facet_marker, 2)

    if advection_scheme == "euler":
        ap = advect_particles(p, V, v, facet_marker)
    elif advection_scheme == "rk2":
        ap = advect_rk2(p, V, v, facet_marker)
    elif advection_scheme == "rk3":
        ap = advect_rk3(p, V, v, facet_marker)
    else:
        assert False

    # Do one timestep, particle must bounce from wall of
    ap.do_step(dt)
    xpE = p.positions()

    # Check if particle correctly bounced off from closed wall
    xpE_root = comm.gather(xpE, root=0)
    if comm.rank == 0:
        xpE_root = np.float64(np.vstack(xpE_root))
        error = np.linalg.norm(x_bounced - xpE_root)
        assert error < 1e-10


@pytest.mark.parametrize("advection_scheme", ["euler", "rk2", "rk3"])
def test_open_boundary(advection_scheme):
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0
    pres = 3

    mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), 10, 10)

    # Particle
    x = RandomRectangle(Point(0.955, 0.45), Point(1.0, 0.55)).generate([pres, pres])
    x = comm.bcast(x, root=0)

    # Given velocity field:
    vexpr = Constant((1.0, 1.0))
    # Given time do_step:
    dt = 0.05

    p = particles(x, [x, x], mesh)

    V = VectorFunctionSpace(mesh, "CG", 1)
    v = Function(V)
    v.assign(vexpr)

    # Different boundary parts
    bound_left = UnitSquareLeft()
    bound_right = UnitSquareRight()
    bound_top = UnitSquareTop()
    bound_bottom = UnitSquareBottom()

    # Mark all facets
    facet_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    facet_marker.set_all(0)

    # Mark as open
    bound_right.mark(facet_marker, 2)

    # Mark other boundaries as closed
    bound_left.mark(facet_marker, 1)
    bound_top.mark(facet_marker, 1)
    bound_bottom.mark(facet_marker, 1)

    if advection_scheme == "euler":
        ap = advect_particles(p, V, v, facet_marker)
    elif advection_scheme == "rk2":
        ap = advect_rk2(p, V, v, facet_marker)
    elif advection_scheme == "rk3":
        ap = advect_rk3(p, V, v, facet_marker)
    else:
        assert False

    # Do one timestep, particle must bounce from wall of
    ap.do_step(dt)
    num_particles = p.number_of_particles()

    # Check if all particles left domain
    if comm.rank == 0:
        assert(num_particles == 0)


@pytest.mark.parametrize('xlims', [[0.0, 1.0]])
@pytest.mark.parametrize('ylims', [[0.0, 1.0]])
@pytest.mark.parametrize('advection_scheme', ['euler', 'rk2', 'rk3'])
def test_bounded_domain_boundary(xlims, ylims, advection_scheme):
    xmin, xmax = xlims
    ymin, ymax = ylims
    pres = 1

    mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), 10, 10)

    ymin += 0.0025
    lims = np.array([xmin, xmax, ymin, ymax])

    v_arr = np.array([-1.0, -1.0])
    vexpr = Constant(v_arr)
    V = VectorFunctionSpace(mesh, "CG", 1)

    x = RandomRectangle(Point(0.05, 0.05), Point(0.15, 0.15)).generate([pres, pres])
    dt = 0.005

    v = Function(V)
    v.assign(vexpr)

    p = particles(x, [x], mesh)

    if advection_scheme == 'euler':
        ap = advect_particles(p, V, v, 'bounded', lims.flatten())
    elif advection_scheme == 'rk2':
        ap = advect_rk2(p, V, v, 'bounded', lims.flatten())
    elif advection_scheme == 'rk3':
        ap = advect_rk3(p, V, v, 'bounded', lims.flatten())
    else:
        assert False

    original_num_particles = p.number_of_particles()
    t = 0.
    while t < 3.0-1e-12:
        ap.do_step(dt)
        t += dt

        assert p.number_of_particles() == original_num_particles

        xpn = np.array(p.get_property(0)).reshape((-1, 2))
        x0 = np.array(p.get_property(1)).reshape((-1, 2))

        analytical_position = x0 + t*v_arr

        analytical_position[:, 0] = np.maximum(np.minimum(xmax, analytical_position[:, 0]), xmin)
        analytical_position[:, 1] = np.maximum(np.minimum(ymax, analytical_position[:, 1]), ymin)

        error = np.abs(xpn - analytical_position)

        assert np.all(np.abs(error) < 1e-12)
