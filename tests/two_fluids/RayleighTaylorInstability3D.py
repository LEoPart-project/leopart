# -*- coding: utf-8 -*-
# Copyright (C) 2020 Nathan Sime
# Contact: nsime _at_ carnegiescience.edu
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import numpy as np
from dolfin import (Cell, UserExpression, BoxMesh, parameters, Constant, Point, CellType,
                    Expression, VectorFunctionSpace, interpolate, ALE, MeshFunction,
                    CompiledSubDomain, Measure, FiniteElement, FunctionSpace, Function,
                    VectorElement, DirichletBC, MixedElement, MPI, XDMFFile, info,
                    assemble, FunctionAssigner, Timer, dot, list_timings, TimingClear, TimingType)
from leopart import (particles, AddDelete, FormsPDEMap, PDEStaticCondensation,
                     FormsStokes,
                     StokesStaticCondensation, advect_rk3, RandomBox)
from mpi4py import MPI as pyMPI

'''
    3D extrusion of the Rayleigh-Taylor instability benchmark problem in geodynamics 
    as documented in https://doi.org/10.1029/97JB01353
'''

comm = pyMPI.COMM_WORLD

parameters["std_out_all_processes"] = False

# Buoyant layer thickness
db = 0.2
# Aspect ratio
lmbdax, lmbdaz = Constant(0.9142), Constant(0.8142)
xmin, xmax = 0.0, float(lmbdax)
ymin, ymax = 0.0, 1.0
zmin, zmax = 0.0, float(lmbdaz)

# Number of cells
nx, ny, nz = 20, 20, 20


# Initial composition field
class StepFunction(UserExpression):

    def eval_cell(self, values, x, cell):
        c = Cell(mesh, cell.index)
        if c.midpoint()[1] > db + 0.02*np.cos(np.pi*x[0]/float(lmbdax))*np.cos(np.pi*x[2]/float(lmbdaz)):
            values[0] = 1.0
        else:
            values[0] = 0.0


mesh = BoxMesh.create(
    comm, [Point(0.0, 0.0, 0.0), Point(float(lmbdax), 1.0, float(lmbdaz))],
    [nx, ny, nz], CellType.Type.tetrahedron)

# Shift the mesh to line up with the initial step function condition
scale = db * (1.0 - db)
shift = Expression(("0.0", "x[1]*(H - x[1])/S*A*cos(pi/Lx*x[0])*cos(pi/Lz*x[2])", "0.0"),
                   A=0.02, Lx=lmbdax, Lz=lmbdaz, H=1.0, S=scale, degree=4)

V = VectorFunctionSpace(mesh, "CG", 1)
displacement = interpolate(shift, V)
ALE.move(mesh, displacement)

# Entrainment functional measures
de = 1
cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
CompiledSubDomain("x[1] > db - DOLFIN_EPS", db=db).mark(cf, de)
dx = Measure("dx", subdomain_data=cf)

# Setup particles
pres = 25
x = RandomBox(Point(xmin, ymin, zmin), Point(xmax, ymax, zmax)).generate([pres, pres, pres])
s = np.zeros((len(x), 1), dtype=np.float_)

# Interpolate initial function onto particles, index slot 1
property_idx = 1
ptcls = particles(x, [s], mesh)

# Define the variational (projection problem)
k = 1
W_e = FiniteElement("DG", mesh.ufl_cell(), k)
T_e = FiniteElement("DG", mesh.ufl_cell(), 0)
Wbar_e = FiniteElement("DGT", mesh.ufl_cell(), k)

# Composition field space
Wh = FunctionSpace(mesh, W_e)
Th = FunctionSpace(mesh, T_e)
Wbarh = FunctionSpace(mesh, Wbar_e)

phi = interpolate(StepFunction(), Wh)
gamma0 = interpolate(StepFunction(), Wh)

ad = AddDelete(ptcls, 50, 55, [phi], [1], [0.0, 1.0])
ptcls.interpolate(phi, property_idx)
ad.do_sweep()

lambda_h = Function(Th)
psibar_h = Function(Wbarh)


# Elements for Stokes
W_e_2 = VectorElement("DG", mesh.ufl_cell(), k)
T_e_2 = VectorElement("DG", mesh.ufl_cell(), 0)
Wbar_e_2 = VectorElement("DGT", mesh.ufl_cell(), k)
Wbar_e_2_H12 = VectorElement("CG", mesh.ufl_cell(), k)["facet"]

Q_E = FiniteElement("DG", mesh.ufl_cell(), k-1)
Qbar_E = FiniteElement("DGT", mesh.ufl_cell(), k)

W_2 = FunctionSpace(mesh, W_e_2)
u_vec = Function(W_2)

# Simulation time and time step Constants
t = Constant(0.0)
dt = Constant(1e-2)

# Initialise advection forms
FuncSpace_adv = {'FuncSpace_local': Wh, 'FuncSpace_lambda': Th, 'FuncSpace_bar': Wbarh}
forms_pde = FormsPDEMap(mesh, FuncSpace_adv).forms_theta_linear(gamma0, u_vec,
                                                                dt, Constant(1.0),
                                                                theta_L=Constant(1.0),
                                                                zeta=Constant(25.0))
phi_bcs = [DirichletBC(Wbarh, Constant(0.0), "near(x[1], 0.0)", "geometric"),
           DirichletBC(Wbarh, Constant(1.0), "near(x[1], 1.0)", "geometric")]
pde_projection = PDEStaticCondensation(mesh, ptcls,
                                       forms_pde['N_a'], forms_pde['G_a'], forms_pde['L_a'],
                                       forms_pde['H_a'],
                                       forms_pde['B_a'],
                                       forms_pde['Q_a'], forms_pde['R_a'], forms_pde['S_a'],
                                       phi_bcs,
                                       property_idx)

# Function spaces for Stokes
mixedL = FunctionSpace(mesh, MixedElement([W_e_2, Q_E]))
mixedG = FunctionSpace(mesh, MixedElement([Wbar_e_2_H12, Qbar_E]))

U0, Uh = Function(mixedL), Function(mixedL)
Uhbar = Function(mixedG)

# BCs
bcs = [DirichletBC(mixedG.sub(0), Constant((0, 0, 0.0)), "near(x[1], 0.0) or near(x[1], 1.0)"),
       DirichletBC(mixedG.sub(0).sub(0), Constant(0),
                   CompiledSubDomain("near(x[0], 0.0) or near(x[0], lmbda)", lmbda=lmbdax)),
       DirichletBC(mixedG.sub(0).sub(2), Constant(0),
                   CompiledSubDomain("near(x[2], 0.0) or near(x[2], lmbda)", lmbda=lmbdaz))]

# Forms Stokes
alpha = Constant(6*k*k)
Rb = Constant(1.0)
eta_top = Constant(1.0)
eta_bottom = Constant(0.01)
eta = eta_bottom + phi * (eta_top - eta_bottom)
forms_stokes = FormsStokes(mesh, mixedL, mixedG, alpha) \
    .forms_steady(eta, Rb * phi * Constant((0, -1, 0)))

ssc = StokesStaticCondensation(mesh,
                               forms_stokes['A_S'], forms_stokes['G_S'],
                               forms_stokes['B_S'],
                               forms_stokes['Q_S'], forms_stokes['S_S'])

# Particle advector
C_CFL = 0.5
hmin = MPI.min(comm, mesh.hmin())
ap = advect_rk3(ptcls, u_vec.function_space(), u_vec, "closed")


# Write particles and their values to XDMF file
particles_directory = "./particles/"
points_list = list(Point(*pp) for pp in ptcls.positions())
particles_values = ptcls.get_property(property_idx)
XDMFFile(os.path.join(particles_directory, "step%.4d.xdmf" % 0)) \
    .write(points_list, particles_values)

n_particles = MPI.sum(comm, len(points_list))
info("Solving with %d particles" % n_particles)

# Write the intitial compostition field to XDMF file
XDMFFile("composition.xdmf").write_checkpoint(phi, "composition", float(t), append=False)
conservation0 = assemble(phi * dx)

# Function assigners to copy subsets of functions between spaces
velocity_assigner = FunctionAssigner(u_vec.function_space(), mixedL.sub(0))
gamma_assigner = FunctionAssigner(gamma0.function_space(), phi.function_space())

# Functionals output filename
data_filename = "data_nx%d_ny%d_Rb%f_CFL%f_k%d_nparticles%d.dat" \
                % (nx, ny, float(Rb), C_CFL, k, n_particles)


# Function to output an iterable of numbers to file
def output_functionals(fname, vals, append=True):
    if comm.rank == 0:
        with open(fname, "a" if append else "w") as fi:
            fi.write(",".join(map(lambda v: "%.6e" % v, vals)) + "\n")


# Compute and output functionals
def output_data_step(append=False):
    urms = (1.0 / (lmbdax*lmbdaz) * assemble(dot(u_vec, u_vec) * dx)) ** 0.5
    conservation = abs(assemble(phi * dx) - conservation0)
    entrainment = assemble(1.0 / (lmbdax * lmbdaz * Constant(db)) * phi * dx(de))
    output_functionals(data_filename, [float(t), float(dt), urms, conservation, entrainment],
                       append=append)


# Initial Stokes solve
time = Timer("ZZZ Stokes assemble")
ssc.assemble_global_system(True)
del time
time = Timer("ZZZ Stokes solve")
for bc in bcs:
    ssc.apply_boundary(bc)
ssc.solve_problem(Uhbar.cpp_object(), Uh.cpp_object(), "mumps", "default")
del time

# Transfer the computed velocity function and compute functionals
velocity_assigner.assign(u_vec, Uh.sub(0))
output_data_step(append=False)

time_snap_shot_interval = 5.0
next_snap_shot_time = time_snap_shot_interval

for j in range(50000):
    max_u_vec = u_vec.vector().norm("linf")
    dt.assign(C_CFL * hmin / max_u_vec)

    t.assign(float(t) + float(dt))
    if float(t) > 2000.0:
        break

    info("Timestep %d, dt = %.3e, t = %.3e" % (j, float(dt), float(t)))

    time = Timer("ZZZ Do_step")
    ap.do_step(float(dt))
    del time

    time = Timer("ZZZ PDE project assemble")
    pde_projection.assemble(True, True)
    del time

    time = Timer("ZZZ PDE project solve")
    pde_projection.solve_problem(psibar_h.cpp_object(), phi.cpp_object(),
                                 lambda_h.cpp_object(), 'mumps', 'default')
    del time

    gamma_assigner.assign(gamma0, phi)

    # Solve Stokes
    time = Timer("ZZZ Stokes assemble")
    ssc.assemble_global_system(True)
    del time
    time = Timer("ZZZ Stokes solve")
    for bc in bcs:
        ssc.apply_boundary(bc)
    ssc.solve_problem(Uhbar.cpp_object(), Uh.cpp_object(), "mumps", "default")
    del time

    velocity_assigner.assign(u_vec, Uh.sub(0))
    output_data_step(append=True)

    # Output particles and composition field
    if float(t) > next_snap_shot_time:
        points_list = list(Point(*pp) for pp in ptcls.positions())
        particles_values = ptcls.get_property(property_idx)
        XDMFFile(os.path.join(particles_directory, "step%.4d.xdmf" % (j+1))) \
            .write(points_list, particles_values)
        XDMFFile("composition.xdmf").write_checkpoint(phi, "composition", float(t), append=True)

        next_snap_shot_time += time_snap_shot_interval

list_timings(TimingClear.clear, [TimingType.wall])
