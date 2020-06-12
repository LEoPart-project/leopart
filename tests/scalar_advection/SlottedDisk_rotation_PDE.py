# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
    Testing the advection of a slotted disk (with sharp discontinuities)
    , using a bounded l2 projection.
    Since no global solve is involved, this
    test is well-suited for assessing the scaling in parallel.
    Note: conservation properties are lost with this approach.
"""

from dolfin import (
    Expression,
    Point,
    VectorFunctionSpace,
    Mesh,
    Constant,
    FunctionSpace,
    assemble,
    dx,
    refine,
    Function,
    XDMFFile,
    Timer,
    assign,
    DirichletBC,
    linear_solver_methods,
)
from mpi4py import MPI as pyMPI
import numpy as np

# Load from package
from leopart import (
    particles,
    advect_rk3,
    PDEStaticCondensation,
    FormsPDEMap,
    RandomCircle,
    SlottedDisk,
    assign_particle_values,
    l2projection,
)

comm = pyMPI.COMM_WORLD

# Domain properties
(x0, y0) = (0.0, 0.0)
(xc, yc) = (-0.15, 0.0)
(r, rdisk) = (0.5, 0.2)
rwidth = 0.05
(lb, ub) = (0.0, 1.0)

# Mesh/particle resolution
pres = 750

# Polynomial order for bounded l2 map
k = 1

# zeta parameter
zeta = Constant(30.0)

# Set solver
if "superlu_dist" in linear_solver_methods():
    solver = "superlu_dist"
else:
    solver = "mumps"

# Magnitude solid body rotation .
Uh = np.pi

# Timestepping
Tend = 2.0
dt = Constant(0.02)
num_steps = np.rint(Tend / float(dt))

# Output directory
store_step = 5
outdir = "./../../results/SlottedDisk_PDE_zeta" + str(int(float(zeta))) + "/"

# Mesh
mesh = Mesh("./../../meshes/circle_0.xml")
mesh = refine(refine(refine(mesh)))

outfile = XDMFFile(mesh.mpi_comm(), outdir + "psi_h.xdmf")

# Particle output
fname_list = [outdir + "xp.pickle", outdir + "rhop.pickle"]
property_list = [0, 1]

# Set slotted disk
psi0_expr = SlottedDisk(
    radius=rdisk, center=[xc, yc], width=rwidth, depth=0.0, degree=3, lb=lb, ub=ub
)

# Function space and velocity field
W = FunctionSpace(mesh, "DG", k)
T = FunctionSpace(mesh, "DG", 0)
Wbar = FunctionSpace(mesh, "DGT", k)
(psi_h, psi_h0, psi_h00) = (Function(W), Function(W), Function(W))
psibar_h = Function(Wbar)

V = VectorFunctionSpace(mesh, "DG", 3)
uh = Function(V)
uh.assign(Expression(("-Uh*x[1]", "Uh*x[0]"), Uh=Uh, degree=3))

# Boundary conditions
bc = DirichletBC(Wbar, Constant(0.0), "on_boundary")

# Generate particles
x = RandomCircle(Point(x0, y0), r).generate([pres, pres])
s = assign_particle_values(x, psi0_expr)

p = particles(x, [s], mesh)
# Initialize advection class, use RK3 scheme
ap = advect_rk3(p, V, uh, "closed")

# Define projections problem
FuncSpace_adv = {"FuncSpace_local": W, "FuncSpace_lambda": T, "FuncSpace_bar": Wbar}
forms_pde = FormsPDEMap(mesh, FuncSpace_adv).forms_theta_linear(
    psi_h0, uh, dt, Constant(1.0), zeta=zeta, h=Constant(0.0)
)
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
    [bc],
    1,
)

# Init projection
lstsq_psi = l2projection(p, W, 1)

# Do projection to get initial field
lstsq_psi.project(psi_h0, lb, ub)
assign(psi_h00, psi_h0)

step = 0
t = 0.0
area_0 = assemble(psi_h0 * dx)

(psi_h_min, psi_h_max) = (0.0, 0.0)
(psibar_min, psibar_max) = (0.0, 0.0)

timer = Timer()
timer.start()

# Write initial field and dump initial particle field
outfile.write_checkpoint(psi_h0, function_name="psi", time_step=0)
p.dump2file(mesh, fname_list, property_list, "wb")

while step < num_steps:
    step += 1
    t += float(dt)

    if comm.Get_rank() == 0:
        print("Step " + str(step))

    # Advect particles
    ap.do_step(float(dt))
    # Project
    pde_projection.assemble(True, True)
    pde_projection.solve_problem(psibar_h, psi_h, solver, "default")

    assign(psi_h0, psi_h)

    psi_h_min = min(psi_h_min, psi_h.vector().min())
    psi_h_max = max(psi_h_max, psi_h.vector().max())
    psibar_min = min(psibar_min, psibar_h.vector().min())
    psibar_max = max(psibar_max, psibar_h.vector().max())
    if comm.rank == 0:
        print("Min max phi {} {}".format(psi_h_min, psi_h_max))
        print("Min max phibar {} {}".format(psibar_min, psibar_max))

    if step % store_step == 0:
        outfile.write_checkpoint(psi_h, function_name="psi", time_step=t, append=True)
        # Dump particles to file
        p.dump2file(mesh, fname_list, property_list, "ab")

timer.stop()

# Dump particles to file
p.dump2file(mesh, fname_list, property_list, "wb")

area_end = assemble(psi_h * dx)
num_part = p.number_of_particles()
l2_error = np.sqrt(abs(assemble((psi_h00 - psi_h) * (psi_h00 - psi_h) * dx)))
if comm.Get_rank() == 0:
    print("Num cells " + str(mesh.num_entities_global(2)))
    print("Num particles " + str(num_part))
    print("Elapsed time " + str(timer.elapsed()[0]))
    print("Area error " + str(abs(area_end - area_0)))
    print("Error " + str(l2_error))
    print("Min max phi {} {}".format(psi_h_min, psi_h_max))
    print("Min max phibar {} {}".format(psibar_min, psibar_max))

outfile.close()
