# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import (VectorElement, FiniteElement, Expression,
                    FunctionSpace, UnitSquareMesh, Function,
                    Constant, MixedElement, DirichletBC, DOLFIN_EPS,
                    Identity, sym, grad, div, assemble, dx, dot)
import numpy as np
from mpi4py import MPI as pyMPI
from leopart import StokesStaticCondensation, FormsStokes
import pytest

comm = pyMPI.COMM_WORLD


def Gamma(x, on_boundary): return on_boundary


def Corner(x, on_boundary): return x[0] < DOLFIN_EPS and x[1] < DOLFIN_EPS


def exact_solution(domain):
    P7 = VectorElement("Lagrange", "triangle", degree=8, dim=2)
    P2 = FiniteElement("Lagrange", "triangle", 3)
    u_exact = Expression(("x[0]*x[0]*(1.0 - x[0])*(1.0 - x[0])*(2.0*x[1] \
                           - 6.0*x[1]*x[1] + 4.0*x[1]*x[1]*x[1])",
                          "-x[1]*x[1]*(1.0 - x[1])*(1.0 - x[1])*(2.0*x[0] \
                           - 6.0*x[0]*x[0] + 4.0*x[0]*x[0]*x[0])"), element=P7, domain=domain)
    p_exact = Expression("x[0]*(1.0 - x[0])", element=P2, domain=domain)
    return u_exact, p_exact


def compute_convergence(iterator, errorlist):
    assert len(iterator) == len(errorlist), 'Iterator list and error list not of same length'
    alpha_list = []
    for i in range(len(iterator)-1):
        conv_rate = np.log(errorlist[i+1]/errorlist[i])/np.log(iterator[i+1]/iterator[i])
        alpha_list.append(conv_rate)
    return alpha_list


@pytest.mark.parametrize('k', [1, 2, 3])
def test_steady_stokes(k):
    # Polynomial order and mesh resolution
    nx_list = [4, 8, 16]

    nu = Constant(1)

    if comm.Get_rank() == 0:
        print('{:=^72}'.format('Computing for polynomial order '+str(k)))

    # Error listst
    error_u, error_p, error_div = [], [], []

    for nx in nx_list:
        if comm.Get_rank() == 0:
            print('# Resolution '+str(nx))

        mesh = UnitSquareMesh(nx, nx)

        # Get forcing from exact solutions
        u_exact, p_exact = exact_solution(mesh)
        f = div(p_exact*Identity(2) - 2*nu*sym(grad(u_exact)))

        # Define FunctionSpaces and functions
        V = VectorElement("DG", mesh.ufl_cell(), k)
        Q = FiniteElement("DG", mesh.ufl_cell(), k-1)
        Vbar = VectorElement("DGT", mesh.ufl_cell(), k)
        Qbar = FiniteElement("DGT", mesh.ufl_cell(), k)

        mixedL = FunctionSpace(mesh, MixedElement([V, Q]))
        mixedG = FunctionSpace(mesh, MixedElement([Vbar, Qbar]))

        Uh = Function(mixedL)
        Uhbar = Function(mixedG)

        # Set forms
        alpha = Constant(6*k*k)
        forms_stokes = FormsStokes(mesh, mixedL, mixedG, alpha).forms_steady(nu, f)

        # No-slip boundary conditions, set pressure in one of the corners
        bc0 = DirichletBC(mixedG.sub(0), Constant((0, 0)), Gamma)
        bc1 = DirichletBC(mixedG.sub(1), Constant(0), Corner, "pointwise")
        bcs = [bc0, bc1]

        # Initialize static condensation class
        ssc = StokesStaticCondensation(mesh,
                                       forms_stokes['A_S'], forms_stokes['G_S'],
                                       forms_stokes['B_S'],
                                       forms_stokes['Q_S'], forms_stokes['S_S'], bcs)

        # Assemble global system and incorporates bcs
        ssc.assemble_global_system(True)
        # Solve using mumps
        ssc.solve_problem(Uhbar, Uh, "mumps", "default")

        # Compute velocity/pressure/local div error
        uh, ph = Uh.split()
        e_u = np.sqrt(np.abs(assemble(dot(uh-u_exact, uh-u_exact)*dx)))
        e_p = np.sqrt(np.abs(assemble((ph-p_exact) * (ph-p_exact)*dx)))
        e_d = np.sqrt(np.abs(assemble(div(uh)*div(uh)*dx)))

        if comm.rank == 0:
            error_u.append(e_u)
            error_p.append(e_p)
            error_div.append(e_d)
            print('Error in velocity '+str(error_u[-1]))
            print('Error in pressure '+str(error_p[-1]))
            print('Local mass error '+str(error_div[-1]))

    if comm.rank == 0:
        iterator_list = [1./float(nx) for nx in nx_list]
        conv_u = compute_convergence(iterator_list, error_u)
        conv_p = compute_convergence(iterator_list, error_p)

        assert any(conv > k+0.75 for conv in conv_u)
        assert any(conv > (k-1)+0.75 for conv in conv_p)
