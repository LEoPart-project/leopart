# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import (
    Expression,
    Constant,
    UnitSquareMesh,
    VectorFunctionSpace,
    Function,
    FunctionSpace,
    VectorElement,
    FiniteElement,
    MixedElement,
    Identity,
    DirichletBC,
    split,
    dx,
    dot,
    div,
    sym,
    grad,
    assemble,
    sqrt,
    assign,
    interpolate,
    near,
)
from leopart import FormsStokes, StokesStaticCondensation
import numpy as np
from mpi4py import MPI as pyMPI

np.set_printoptions(precision=1, linewidth=150)
comm = pyMPI.COMM_WORLD


# Define boundary
def Gamma(x, on_boundary):
    return on_boundary


def Corner(x, on_boundary):
    return near(x[0], 1.0) and near(x[1], 1.0)


# TODO: Consider merging with test_steady_stokes.py
def test_unsteady_stokes():
    nx, ny = 15, 15
    k = 1
    nu = Constant(1.0e-0)
    dt = Constant(2.5e-2)
    num_steps = 20
    theta0 = 1.0  # Initial theta value
    theta1 = 0.5  # Theta after 1 step
    theta = Constant(theta0)

    mesh = UnitSquareMesh(nx, ny)

    # The 'unsteady version' of the benchmark in the 2012 paper by Labeur&Wells
    u_exact = Expression(
        (
            "sin(t) * x[0]*x[0]*(1.0 - x[0])*(1.0 - x[0])*(2.0*x[1] \
                           -6.0*x[1]*x[1] + 4.0*x[1]*x[1]*x[1])",
            "-sin(t)* x[1]*x[1]*(1.0 - x[1])*(1.0 - x[1])*(2.0*x[0] \
                           - 6.0*x[0]*x[0] + 4.0*x[0]*x[0]*x[0])",
        ),
        t=0,
        degree=7,
        domain=mesh,
    )
    p_exact = Expression("sin(t) * x[0]*(1.0 - x[0])", t=0, degree=7, domain=mesh)
    du_exact = Expression(
        (
            "cos(t) * x[0]*x[0]*(1.0 - x[0])*(1.0 - x[0])*(2.0*x[1] \
                            - 6.0*x[1]*x[1] + 4.0*x[1]*x[1]*x[1])",
            "-cos(t)* x[1]*x[1]*(1.0 - x[1])*(1.0 - x[1])*(2.0*x[0] \
                            -6.0*x[0]*x[0] + 4.0*x[0]*x[0]*x[0])",
        ),
        t=0,
        degree=7,
        domain=mesh,
    )

    ux_exact = Expression(
        (
            "x[0]*x[0]*(1.0 - x[0])*(1.0 - x[0])*(2.0*x[1] \
                            - 6.0*x[1]*x[1] + 4.0*x[1]*x[1]*x[1])",
            "-x[1]*x[1]*(1.0 - x[1])*(1.0 - x[1])*(2.0*x[0] \
                            - 6.0*x[0]*x[0] + 4.0*x[0]*x[0]*x[0])",
        ),
        degree=7,
        domain=mesh,
    )

    px_exact = Expression("x[0]*(1.0 - x[0])", degree=7, domain=mesh)

    sin_ext = Expression("sin(t)", t=0, degree=7, domain=mesh)

    f = du_exact + sin_ext * div(px_exact * Identity(2) - 2 * sym(grad(ux_exact)))

    Vhigh = VectorFunctionSpace(mesh, "DG", 7)
    Phigh = FunctionSpace(mesh, "DG", 7)

    # New syntax:
    V = VectorElement("DG", mesh.ufl_cell(), k)
    Q = FiniteElement("DG", mesh.ufl_cell(), k - 1)
    Vbar = VectorElement("DGT", mesh.ufl_cell(), k)
    Qbar = FiniteElement("DGT", mesh.ufl_cell(), k)

    mixedL = FunctionSpace(mesh, MixedElement([V, Q]))
    mixedG = FunctionSpace(mesh, MixedElement([Vbar, Qbar]))
    V2 = FunctionSpace(mesh, V)

    Uh = Function(mixedL)
    Uhbar = Function(mixedG)
    U0 = Function(mixedL)
    Uhbar0 = Function(mixedG)
    u0, p0 = split(U0)
    ubar0, pbar0 = split(Uhbar0)
    ustar = Function(V2)

    # Then the boundary conditions
    bc0 = DirichletBC(mixedG.sub(0), Constant((0, 0)), Gamma)
    bc1 = DirichletBC(mixedG.sub(1), Constant(0), Corner, "pointwise")
    bcs = [bc0, bc1]

    alpha = Constant(6 * k * k)
    forms_stokes = FormsStokes(mesh, mixedL, mixedG, alpha).forms_unsteady(ustar, dt, nu, f)
    ssc = StokesStaticCondensation(
        mesh,
        forms_stokes["A_S"],
        forms_stokes["G_S"],
        forms_stokes["G_ST"],
        forms_stokes["B_S"],
        forms_stokes["Q_S"],
        forms_stokes["S_S"],
    )

    t = 0.0
    step = 0
    for step in range(num_steps):
        step += 1
        t += float(dt)
        if comm.Get_rank() == 0:
            print("Step " + str(step) + " Time " + str(t))

        # Set time level in exact solution
        u_exact.t = t
        p_exact.t = t

        du_exact.t = t - (1 - float(theta)) * float(dt)
        sin_ext.t = t - (1 - float(theta)) * float(dt)

        ssc.assemble_global_lhs()
        ssc.assemble_global_rhs()
        for bc in bcs:
            ssc.apply_boundary(bc)

        ssc.solve_problem(Uhbar, Uh, "none", "default")
        assign(U0, Uh)
        assign(ustar, U0.sub(0))
        assign(Uhbar0, Uhbar)
        if step == 1:
            theta.assign(theta1)

        udiv_e = sqrt(assemble(div(Uh.sub(0)) * div(Uh.sub(0)) * dx))

    u_ex_h = interpolate(u_exact, Vhigh)
    p_ex_h = interpolate(p_exact, Phigh)

    u_error = sqrt(assemble(dot(Uh.sub(0) - u_ex_h, Uh.sub(0) - u_ex_h) * dx))
    p_error = sqrt(assemble(dot(Uh.sub(1) - p_ex_h, Uh.sub(1) - p_ex_h) * dx))

    assert udiv_e < 1e-12
    assert u_error < 1.5e-4
    assert p_error < 1e-2
