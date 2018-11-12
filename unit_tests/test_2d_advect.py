# __author__ = 'Jakob Maljaars <j.m.maljaars@tudelft.nl>'
# __date__   = '2018-08-02'
# __copyright__ = 'Copyright (C) 2011' + __author__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

"""
Unit tests for advection of single particle
"""

from dolfin import (SubDomain, UnitSquareMesh, BoundaryMesh, RectangleMesh,
                    Point, Constant, Expression, VectorFunctionSpace, Function, near)
from DolfinParticles import (particles, advect_particles, advect_rk2,
                             advect_rk3, RandomRectangle)
from mpi4py import MPI as pyMPI
import numpy as np

comm = pyMPI.COMM_WORLD


class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary
        # AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 0) and near(x[1], 1)) or
                          (near(x[0], 1) and near(x[1], 0)))) and
                    on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
        elif near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.


def compute_convergence(iterator, errorlist):
    assert len(iterator) == len(errorlist), \
           'Iterator list and error list not of same length'
    alpha_list = []
    for i in range(len(iterator)-1):
        conv_rate = np.log(errorlist[i+1]/errorlist[i])/np.log(iterator[i+1]/iterator[i])
        alpha_list.append(conv_rate)
    return alpha_list


def decorate_advect_particle(my_func):
    def wrapper():
        mesh = UnitSquareMesh(10, 10)
        bmesh = BoundaryMesh(mesh, 'exterior')

        vexpr = Expression(('-pi*(x[1] - 0.5)', 'pi*(x[0]-0.5)'), degree=3)
        V = VectorFunctionSpace(mesh, "CG", 1)
        x = np.array([[0.25, 0.25]])
        dt_list = [0.08, 0.04, 0.02, 0.01, 0.005]
        return my_func(mesh, bmesh, V, vexpr, x, dt_list)
    return wrapper


def test_advect_particle():
    if comm.Get_rank() == 0:
        print('Run advect_particle')

    # Rotate one particle, and compute the error
    mesh = UnitSquareMesh(10, 10)

    vexpr = Expression(('-pi*(x[1] - 0.5)', 'pi*(x[0]-0.5)'), degree=3)
    V = VectorFunctionSpace(mesh, "CG", 1)
    x = np.array([[0.25, 0.25]])
    dt_list = [0.08, 0.04, 0.02, 0.01, 0.005]

    v = Function(V)
    v.assign(vexpr)
    error_list = []

    for dt in dt_list:
        p = particles(x, [x, x], mesh)
        ap = advect_particles(p, V, v, 'closed')
        xp_0 = p.positions()
        t = 0.
        while t < 2.-1e-12:
            ap.do_step(dt)
            t += dt

        xp_end = p.positions()
        error_list.append(np.linalg.norm(xp_0 - xp_end))

    if not all(eps == 0 for eps in error_list):
        rate = compute_convergence(dt_list, error_list)
        assert any(i > 0.9 for i in rate)


def test_advect_particle_rk2():
    if comm.Get_rank() == 0:
        print('Run advect_particle_rk2')

    # Rotate one particle, and compute the error
    mesh = UnitSquareMesh(10, 10)

    vexpr = Expression(('-pi*(x[1] - 0.5)', 'pi*(x[0]-0.5)'), degree=3)
    V = VectorFunctionSpace(mesh, "CG", 1)
    x = np.array([[0.25, 0.25]])
    dt_list = [0.08, 0.04, 0.02, 0.01, 0.005]

    v = Function(V)
    v.assign(vexpr)
    error_list = []

    for dt in dt_list:
        p = particles(x, [x, x], mesh)
        ap = advect_rk2(p, V, v, 'closed')
        xp_0 = p.positions()

        t = 0.
        while t < 2.-1e-12:
            ap.do_step(dt)
            t += dt

        xp_end = p.positions()
        error_list.append(np.linalg.norm(xp_0 - xp_end))

    if not all(eps == 0 for eps in error_list):
        rate = compute_convergence(dt_list, error_list)
        assert any(i > 1.95 for i in rate)


def test_advect_particle_rk3():
    if comm.Get_rank() == 0:
        print('Run advect_particle_rk3')

    # Rotate one particle, and compute the error
    mesh = UnitSquareMesh(10, 10)

    vexpr = Expression(('-pi*(x[1] - 0.5)', 'pi*(x[0]-0.5)'), degree=3)
    V = VectorFunctionSpace(mesh, "CG", 1)
    x = np.array([[0.25, 0.25]])
    dt_list = [0.08, 0.04, 0.02, 0.01, 0.005]

    v = Function(V)
    v.assign(vexpr)

    error_list = []
    for dt in dt_list:
        p = particles(x, [x, x], mesh)
        ap = advect_rk3(p, V, v, 'closed')
        xp_0 = p.positions()

        t = 0.
        while t < 2.-1e-12:
            ap.do_step(dt)
            t += dt

        xp_end = p.positions()
        error_list.append(np.linalg.norm(xp_0 - xp_end))

    if not all(eps == 0 for eps in error_list):
        rate = compute_convergence(dt_list, error_list)
        assert any(i > 2.9 for i in rate)


def decorate_periodic_tests(my_func):
    def wrapper():
        xmin = 0.
        xmax = 1.
        ymin = 0.
        ymax = 1.

        mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), 10, 10)

        lims = np.array([[xmin, xmin, ymin, ymax], [xmax, xmax, ymin, ymax],
                         [xmin, xmax, ymin, ymin], [xmin, xmax, ymax, ymax]])

        vexpr = Constant((1., 1.))
        V = VectorFunctionSpace(mesh, "CG", 1)

        x = RandomRectangle(Point(0.05, 0.05), Point(0.15, 0.15)).generate([3, 3])
        x = comm.bcast(x, root=0)
        dt = 0.05

        xp0, xpE = my_func(mesh, lims, V, vexpr, x, dt)

        xp0_root = comm.gather(xp0, root=0)
        xpE_root = comm.gather(xpE, root=0)

        if comm.Get_rank() == 0:
            xp0_root = np.float32(np.vstack(xp0_root))
            xpE_root = np.float32(np.vstack(xpE_root))
            error = np.linalg.norm(xp0_root - xpE_root)
            if error > 1e-10:
                raise Exception("Error too high in function " + my_func.__name__)
        return
    return wrapper


def test_advect_particle_periodic():
    xmin = 0.
    xmax = 1.
    ymin = 0.
    ymax = 1.

    mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), 10, 10)

    lims = np.array([[xmin, xmin, ymin, ymax], [xmax, xmax, ymin, ymax],
                     [xmin, xmax, ymin, ymin], [xmin, xmax, ymax, ymax]])

    vexpr = Constant((1., 1.))
    V = VectorFunctionSpace(mesh, "CG", 1)

    x = RandomRectangle(Point(0.05, 0.05), Point(0.15, 0.15)).generate([3, 3])
    x = comm.bcast(x, root=0)
    dt = 0.05

    v = Function(V)
    v.assign(vexpr)

    p = particles(x, [x*0, x**2], mesh)
    ap = advect_particles(p, V, v, 'periodic', lims.flatten())

    xp0 = p.positions()
    t = 0.
    while t < 1.-1e-12:
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


def test_advect_particle_periodic_rk2():
    xmin = 0.
    xmax = 1.
    ymin = 0.
    ymax = 1.

    mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), 10, 10)

    lims = np.array([[xmin, xmin, ymin, ymax], [xmax, xmax, ymin, ymax],
                     [xmin, xmax, ymin, ymin], [xmin, xmax, ymax, ymax]])

    vexpr = Constant((1., 1.))
    V = VectorFunctionSpace(mesh, "CG", 1)

    x = RandomRectangle(Point(0.05, 0.05), Point(0.15, 0.15)).generate([3, 3])
    x = comm.bcast(x, root=0)
    dt = 0.05

    v = Function(V)
    v.assign(vexpr)

    p = particles(x, [x * 0, x**2], mesh)
    ap = advect_rk2(p, V, v, 'periodic', lims.flatten())

    xp0 = p.positions()
    t = 0.
    while t < 1.-1e-12:
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


def test_advect_particle_periodic_rk3():
    xmin = 0.
    xmax = 1.
    ymin = 0.
    ymax = 1.

    mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), 10, 10)

    lims = np.array([[xmin, xmin, ymin, ymax], [xmax, xmax, ymin, ymax],
                     [xmin, xmax, ymin, ymin], [xmin, xmax, ymax, ymax]])

    vexpr = Constant((1., 1.))
    V = VectorFunctionSpace(mesh, "CG", 1)

    x = RandomRectangle(Point(0.05, 0.05), Point(0.15, 0.15)).generate([3, 3])
    x = comm.bcast(x, root=0)
    dt = 0.05

    v = Function(V)
    v.assign(vexpr)

    p = particles(x, [x[:, 0] * 0, x**2], mesh)
    ap = advect_rk2(p, V, v, 'periodic', lims.flatten())

    xp0 = p.positions()
    t = 0.
    while t < 1.-1e-12:
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
