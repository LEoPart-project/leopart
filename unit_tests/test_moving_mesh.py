# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import (
    UserExpression,
    FunctionSpace,
    VectorFunctionSpace,
    Function,
    FiniteElement,
    RectangleMesh,
    Measure,
    SubDomain,
    MeshFunction,
    Point,
    FacetNormal,
    Constant,
    ALE,
    assign,
    conditional,
    ge,
    project,
    dot,
    assemble,
    near,
    dx,
    dS,
    ds,
)
from leopart import (
    particles,
    PDEStaticCondensation,
    FormsPDEMap,
    RandomRectangle,
    l2projection,
    GaussianPulse,
)
from mpi4py import MPI as pyMPI
import numpy as np

comm = pyMPI.COMM_WORLD


# Moving mesh with velocity
class PeriodicVelocity(UserExpression):
    def __init__(self, x0, xL, dt, t, **kwargs):
        self.t = t
        self.dt = dt
        self.x0 = x0
        self.xL = xL
        super().__init__(self, **kwargs)

    def compute_ubc(self):
        self.du = self.__moving_bound()
        self.slope = self.du / (self.x0 - self.xL)

    def update(self):
        self.x0 += self.du * self.dt
        self.t += self.dt

    def eval(self, value, x):
        if x[0] > self.xL:
            value[0] = 0
        else:
            value[0] = self.slope * (x[0] - self.xL)
        value[1] = 0.0

    def value_shape(self):
        return (2,)

    def __moving_bound(self):
        u_bc = 0.5 * np.cos(np.pi * self.t)
        return u_bc


class Left(SubDomain):
    def __init__(self, bvalue):
        SubDomain.__init__(self)
        self.bvalue = bvalue

    def inside(self, x, on_boundary):
        return near(x[0], self.bvalue)


def assign_particle_values(x, u_exact):
    if comm.Get_rank() == 0:
        s = np.asarray([u_exact(x[i, :]) for i in range(len(x))], dtype=np.float_)
    else:
        s = None
    return s


def facet_integral(integrand):
    return integrand("-") * dS + integrand("+") * dS + integrand * ds


def test_moving_mesh():
    t = 0.0
    dt = 0.025
    num_steps = 20
    xmin, ymin = 0.0, 0.0
    xmax, ymax = 2.0, 2.0
    xc, yc = 1.0, 1.0
    nx, ny = 20, 20
    pres = 150
    k = 1

    mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), nx, ny)
    n = FacetNormal(mesh)

    # Class for mesh motion
    dU = PeriodicVelocity(xmin, xmax, dt, t, degree=1)

    Qcg = VectorFunctionSpace(mesh, "CG", 1)

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    leftbound = Left(xmin)

    leftbound.mark(boundaries, 99)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Create function spaces
    Q_E_Rho = FiniteElement("DG", mesh.ufl_cell(), k)
    T_1 = FunctionSpace(mesh, "DG", 0)
    Qbar_E = FiniteElement("DGT", mesh.ufl_cell(), k)

    Q_Rho = FunctionSpace(mesh, Q_E_Rho)
    Qbar = FunctionSpace(mesh, Qbar_E)

    phih, phih0 = Function(Q_Rho), Function(Q_Rho)
    phibar = Function(Qbar)

    # Advective velocity
    uh = Function(Qcg)
    uh.assign(Constant((0.0, 0.0)))
    # Mesh velocity
    umesh = Function(Qcg)
    # Total velocity
    uadvect = uh - umesh

    # Now throw in the particles
    x = RandomRectangle(Point(xmin, ymin), Point(xmax, ymax)).generate([pres, pres])
    s = assign_particle_values(
        x,
        GaussianPulse(center=(xc, yc), sigma=float(0.25), U=[0, 0], time=0.0, height=1.0, degree=3),
    )
    x = comm.bcast(x, root=0)
    s = comm.bcast(s, root=0)
    p = particles(x, [s], mesh)

    # Define projections problem
    FuncSpace_adv = {"FuncSpace_local": Q_Rho, "FuncSpace_lambda": T_1, "FuncSpace_bar": Qbar}
    FormsPDE = FormsPDEMap(mesh, FuncSpace_adv, ds=ds)
    forms_pde = FormsPDE.forms_theta_linear(phih0, uadvect, dt, Constant(1.0), zeta=Constant(0.0))
    pde_projection = PDEStaticCondensation(
        mesh,
        p,
        forms_pde["N_a"],
        forms_pde["G_a"],
        forms_pde["L_a"],
        forms_pde["H_a"],
        forms_pde["B_a"],
        forms_pde["Q_a"],
        forms_pde["R_a"],
        forms_pde["S_a"],
        [],
        1,
    )

    # Initialize the initial condition at mesh by an l2 projection
    lstsq_rho = l2projection(p, Q_Rho, 1)
    lstsq_rho.project(phih0.cpp_object())

    for step in range(num_steps):
        # Compute old area at old configuration
        old_area = assemble(phih0 * dx)

        # Pre-assemble rhs
        pde_projection.assemble_state_rhs()

        # Move mesh
        dU.compute_ubc()
        umesh.assign(project(dU, Qcg))

        ALE.move(mesh, project(dU * dt, Qcg))
        dU.update()

        # Relocate particles as a result of mesh motion
        # NOTE: if particles were advected themselve,
        # we had to run update_facets_info() here as well
        p.relocate()

        # Assemble left-hand side on new config, but not the right-hand side
        pde_projection.assemble(True, False)
        pde_projection.solve_problem(phibar.cpp_object(), phih.cpp_object(), "mumps", "none")

        # Needed to compute conservation, note that there
        # is an outgoing flux at left boundary
        new_area = assemble(phih * dx)
        gamma = conditional(ge(dot(uadvect, n), 0), 0, 1)
        bflux = assemble((1 - gamma) * dot(uadvect, n) * phih * ds)

        # Update solution
        assign(phih0, phih)

        # Put assertion on (global) mass balance, local mass balance is
        # too time consuming but should pass also
        assert new_area - old_area + bflux * dt < 1e-12

        # Assert that max value of phih stays close to 2 and
        # min value close to 0. This typically will fail if
        # we do not do a correct relocate of particles
        assert np.amin(phih.vector().get_local()) > -0.015
        assert np.amax(phih.vector().get_local()) < 1.04
