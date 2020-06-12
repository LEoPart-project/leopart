# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import (
    UnitCubeMesh,
    SubDomain,
    VectorFunctionSpace,
    Function,
    Constant,
    Point,
    MeshFunction,
    near,
)
from leopart import particles, advect_rk3, advect_rk2, advect_particles, RandomBox
from mpi4py import MPI as pyMPI
import numpy as np
import pytest

comm = pyMPI.COMM_WORLD


class Boundaries(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class UnitCubeRight(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1)


@pytest.mark.parametrize("advection_scheme", ["euler", "rk2", "rk3"])
def test_advect_periodic(advection_scheme):
    xmin, ymin, zmin = 0.0, 0.0, 0.0
    xmax, ymax, zmax = 1.0, 1.0, 1.0
    pres = 10

    mesh = UnitCubeMesh(10, 10, 10)

    lims = np.array(
        [
            [xmin, xmin, ymin, ymax, zmin, zmax],
            [xmax, xmax, ymin, ymax, zmin, zmax],
            [xmin, xmax, ymin, ymin, zmin, zmax],
            [xmin, xmax, ymax, ymax, zmin, zmax],
            [xmin, xmax, ymin, ymax, zmin, zmin],
            [xmin, xmax, ymin, ymax, zmax, zmax],
        ]
    )

    vexpr = Constant((1.0, 1.0, 1.0))
    V = VectorFunctionSpace(mesh, "CG", 1)
    v = Function(V)
    v.assign(vexpr)

    x = RandomBox(Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)).generate([pres, pres, pres])
    x = comm.bcast(x, root=0)
    dt = 0.05

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

    xp0_root = comm.gather(xp0, root=0)
    xpE_root = comm.gather(xpE, root=0)

    assert len(xp0) == len(xpE)
    num_particles = p.number_of_particles()

    if comm.Get_rank() == 0:
        xp0_root = np.float32(np.vstack(xp0_root))
        xpE_root = np.float32(np.vstack(xpE_root))

        # Sort on x positions
        xp0_root = xp0_root[xp0_root[:, 0].argsort(), :]
        xpE_root = xpE_root[xpE_root[:, 0].argsort(), :]

        error = np.linalg.norm(xp0_root - xpE_root)
        assert error < 1e-10
        assert num_particles - pres ** 3 == 0


@pytest.mark.parametrize("advection_scheme", ["euler", "rk2", "rk3"])
def test_advect_open(advection_scheme):
    pres = 3

    mesh = UnitCubeMesh(10, 10, 10)

    # Particle
    x = RandomBox(Point(0.955, 0.45, 0.5), Point(0.99, 0.55, 0.6)).generate([pres, pres, pres])
    x = comm.bcast(x, root=0)

    # Given velocity field:
    vexpr = Constant((1.0, 1.0, 1.0))
    # Given time do_step:
    dt = 0.05

    p = particles(x, [x, x], mesh)

    V = VectorFunctionSpace(mesh, "CG", 1)
    v = Function(V)
    v.assign(vexpr)

    # Different boundary parts
    bounds = Boundaries()
    bound_right = UnitCubeRight()

    # Mark all facets
    facet_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    facet_marker.set_all(0)
    bounds.mark(facet_marker, 1)
    bound_right.mark(facet_marker, 2)

    # Mark as open
    bound_right.mark(facet_marker, 2)

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
        assert num_particles == 0
