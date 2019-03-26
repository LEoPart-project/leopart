# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import (SubDomain, Constant, Expression, FunctionSpace, VectorElement,
                    MixedElement, FiniteElement, Function, DirichletBC, near, assemble, dot, dx,
                    Point, as_vector, assign, div, sqrt, XDMFFile, RectangleMesh, project,
                    Timer, list_timings, TimingClear, TimingType, MPI)
import numpy as np
from mpi4py import MPI as pyMPI
from leopart import (particles, advect_rk3, RandomRectangle,
                     AddDelete, PDEStaticCondensation,
                     StokesStaticCondensation,
                     FormsPDEMap, FormsStokes, assign_particle_values)

comm = pyMPI.COMM_WORLD


class PeriodicBoundary(SubDomain):
    def __init__(self, ddict):
        SubDomain.__init__(self)
        self.xmin, self.xmax = ddict['xmin'], ddict['xmax']
        self.ymin, self.ymax = ddict['ymin'], ddict['ymax']

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax
        # return True if on left or bottom boundary AND NOT
        # on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], xmin) or near(x[1], ymin)) and
                    (not ((near(x[0], xmin) and near(x[1], ymax)) or
                          (near(x[0], xmax) and near(x[1], ymin)))) and on_boundary)

    def map(self, x, y):
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax
        # If near top left corner
        if near(x[0], xmax) and near(x[1], ymax):
            y[0] = x[0] - (xmax - xmin)
            y[1] = x[1] - (ymax - ymin)
        elif near(x[0], xmax):
            y[0] = x[0] - (xmax - xmin)
            y[1] = x[1]
        else:
            y[0] = x[0]
            y[1] = x[1] - (ymax - ymin)


class Corner(SubDomain):
    def __init__(self, ddict):
        SubDomain.__init__(self)
        self.xmax, self.ymax = ddict['xmax'], ddict['ymax']

    def inside(self, x, on_boundary):
        return near(x[0], self.xmax) and near(x[1], self.ymax)


# USER INPUT
geometry = {'xmin': -1., 'ymin': -1., 'xmax': 1., 'ymax': 1.}

# Mesh resolution
nx = 128  # 64
ny = 128  # 64

# Particle resolution
pres = 960  # 480

# Time stepping
Tend = .2
dt = Constant(1.25e-2)

# Viscosity
nu = Constant(2e-3)

# Stokes related
k = 2
kbar = k

alpha = Constant(6*k*k)
beta = Constant(0.)

# Theta value to compute v*
theta_init = 1.0
theta_next = 0.5
theta_L = Constant(theta_init)

# Theta value for updating particles
theta_p = 0.5

# Taylor-Green wave mode
mode = 1.

# Directory for output
outdir_base = './../../results/TaylorGreen_2D/'

# Exact/Initial solution Taylor-Green
U_exact = ('-U*exp(-2*nu*pow(pi,2)*t)*(cos(mode*pi*(x[0]))*sin(mode*pi*(x[1])))',
           ' U*exp(-2*nu*pow(pi,2)*t)*(cos(mode*pi*(x[1]))*sin(mode*pi*(x[0])))')
u_exact = Expression(U_exact, degree=7, U=float(1.), nu=float(nu), t=float(0.), mode=mode)

P_exact = ('-rho*0.25*exp(-4*nu*pow(pi,2)*t)*(cos(2*pi*(x[0])) + cos(2*pi*(x[1]))-2.0)')
p_exact = Expression(P_exact, degree=3, rho=1., nu=float(nu), t=float(0.))

f = Constant((0., 0.))

# Create mesh
xmin, ymin = geometry['xmin'], geometry['ymin']
xmax, ymax = geometry['xmax'], geometry['ymax']

mesh = RectangleMesh(MPI.comm_world, Point(xmin, ymin), Point(xmax, ymax), nx, ny)
pbc = PeriodicBoundary(geometry)

# xdmf output
xdmf_u = XDMFFile(mesh.mpi_comm(), outdir_base+"u.xdmf")
xdmf_p = XDMFFile(mesh.mpi_comm(), outdir_base+"p.xdmf")

# Required elements
W_E_2 = VectorElement("DG", mesh.ufl_cell(), k)
T_E_2 = VectorElement("DG", mesh.ufl_cell(), 0)
Wbar_E_2 = VectorElement("DGT", mesh.ufl_cell(), kbar)
Wbar_E_2_H12 = VectorElement("CG", mesh.ufl_cell(), kbar)["facet"]

Q_E = FiniteElement("DG", mesh.ufl_cell(), k-1)
Qbar_E = FiniteElement("DGT", mesh.ufl_cell(), k)

# Function spaces for projection
W_2 = FunctionSpace(mesh, W_E_2)
T_2 = FunctionSpace(mesh, T_E_2)
Wbar_2 = FunctionSpace(mesh, Wbar_E_2, constrained_domain=pbc)
Wbar_2_H12 = FunctionSpace(mesh, Wbar_E_2_H12, constrained_domain=pbc)

# Function spaces for Stokes
mixedL = FunctionSpace(mesh, MixedElement([W_E_2, Q_E]), constrained_domain=pbc)
mixedG = FunctionSpace(mesh, MixedElement([Wbar_E_2_H12, Qbar_E]), constrained_domain=pbc)

# Define functions
u0_a, ustar = Function(W_2), Function(W_2)
duh0, duh00 = Function(W_2), Function(W_2)

ubar0_a = Function(Wbar_2_H12)
ubar_a = Function(Wbar_2)
Udiv = Function(W_2)

U0, Uh = Function(mixedL), Function(mixedL)
Uhbar = Function(mixedG)

u0_a.assign(u_exact)
ubar0_a.assign(u_exact)
Udiv.assign(u_exact)

# Initialize particles
x = RandomRectangle(Point(xmin, ymin), Point(xmax, ymax)).generate([pres, pres])
s = assign_particle_values(x, u_exact)

lims = np.array([[xmin, xmin, ymin, ymax], [xmax, xmax, ymin, ymax],
                 [xmin, xmax, ymin, ymin], [xmin, xmax, ymax, ymax]])

# Particle specific momentum is stored at slot 1
# the second slot will be to store old velocities at particle level
property_idx = 1
p = particles(x, [s, s], mesh)
ap = advect_rk3(p, W_2, Udiv, 'periodic', lims.flatten())

# Particle management
AD = AddDelete(p, 10, 20, [Udiv, duh0])

# Forms PDE map
funcspace_dict = {'FuncSpace_local': W_2, 'FuncSpace_lambda': T_2, 'FuncSpace_bar': Wbar_2}
forms_adv = FormsPDEMap(mesh, funcspace_dict).forms_theta_nlinear(u0_a, ubar0_a, dt,
                                                                  theta_map=Constant(1.0),
                                                                  theta_L=theta_L,
                                                                  duh0=duh0, duh00=duh00)
pde_projection = PDEStaticCondensation(mesh, p,
                                       forms_adv['N_a'], forms_adv['G_a'], forms_adv['L_a'],
                                       forms_adv['H_a'],
                                       forms_adv['B_a'],
                                       forms_adv['Q_a'], forms_adv['R_a'], forms_adv['S_a'],
                                       property_idx)

# Forms Stokes
# Set pressure in corner to zero
bc1 = DirichletBC(mixedG.sub(1), Constant(0), Corner(geometry), "pointwise")
bcs = [bc1]

forms_stokes = FormsStokes(mesh, mixedL, mixedG, alpha).forms_unsteady(ustar, dt, nu, f)

ssc = StokesStaticCondensation(mesh,
                               forms_stokes['A_S'], forms_stokes['G_S'],
                               forms_stokes['B_S'],
                               forms_stokes['Q_S'], forms_stokes['S_S'])


ex = as_vector((1., 0.))
ey = as_vector((0., 1.))

# Prepare time stepping loop
num_steps = np.rint(Tend/float(dt))
step = 0
t = 0.

timer = Timer()
timer.start()

while step < num_steps:
    step += 1
    t += float(dt)

    if comm.Get_rank() == 0:
        print('Step number '+str(step))

    t1 = Timer("[P] Sweep and step")
    # Limit number of particles
    AD.do_sweep()

    # Advect particles
    ap.do_step(float(dt))

    # Do failsafe sweep
    AD.do_sweep_failsafe(7)
    del(t1)

    # Do constrained projection
    t1 = Timer("[P] Assemble")
    pde_projection.assemble(True, True)
    del(t1)
    t1 = Timer("[P] Solve")
    pde_projection.solve_problem(ubar_a.cpp_object(), ustar.cpp_object(), 'bicgstab', 'hypre_amg')
    del(t1)

    # Solve Stokes
    t1 = Timer("[P] Stokes assemble")
    ssc.assemble_global_system(True)
    del(t1)
    t1 = Timer("[P] Stokes solve")
    for bc in bcs:
        ssc.apply_boundary(bc)
    ssc.solve_problem(Uhbar.cpp_object(), Uh.cpp_object(), "mumps", "default")
    del(t1)

    t1 = Timer("[P] Assign and output")
    # Needed for particle advection
    assign(Udiv, Uh.sub(0))

    # Needed for constrained map
    assign(ubar0_a, Uhbar.sub(0))
    assign(u0_a, ustar)
    assign(duh00, duh0)
    assign(duh0, project(Uh.sub(0)-ustar, W_2))

    p.increment(Udiv.cpp_object(), ustar.cpp_object(),
                np.array([1, 2], dtype=np.uintp), theta_p, step)

    if step == 2:
        theta_L.assign(theta_next)

    # Probably can be combined into one file?
    xdmf_u.write(Uh.sub(0), t)
    xdmf_p.write(Uh.sub(1), t)
    del(t1)

timer.stop()

# Compute errors
u_exact.t = t
p_exact.t = t

u_error = sqrt(assemble(dot(Uh.sub(0) - u_exact, Uh.sub(0) - u_exact)*dx))
p_error = sqrt(assemble(dot(Uh.sub(1) - p_exact, Uh.sub(1) - p_exact)*dx))
udiv = sqrt(assemble(div(Uh.sub(0))*div(Uh.sub(0))*dx))

momentum = assemble((dot(Uh.sub(0), ex) + dot(Uh.sub(0), ey)) * dx)

if comm.Get_rank() == 0:
    print("Velocity error "+str(u_error))
    print("Pressure error "+str(p_error))
    print("Momentum "+str(momentum))
    print("Divergence "+str(udiv))
    print('Elapsed time '+str(timer.elapsed()[0]))

list_timings(TimingClear.keep, [TimingType.wall])
