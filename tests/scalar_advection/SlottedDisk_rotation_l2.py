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
)
from mpi4py import MPI as pyMPI
import numpy as np

# Load from package
from leopart import (
    particles,
    advect_rk3,
    l2projection,
    RandomCircle,
    assign_particle_values,
    SlottedDisk,
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

# Magnitude solid body rotation .
Uh = np.pi

# Timestepping
Tend = 2.0
dt = Constant(0.02)
num_steps = np.rint(Tend / float(dt))

# Output directory
store_step = 5
outdir = "./../../results/SlottedDisk_l2/"

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
(psi_h0, psi_h) = (Function(W), Function(W))

V = VectorFunctionSpace(mesh, "DG", 3)
uh = Function(V)
uh.assign(Expression(("-Uh*x[1]", "Uh*x[0]"), Uh=Uh, degree=3))

# Generate particles
x = RandomCircle(Point(x0, y0), r).generate([pres, pres])
s = assign_particle_values(x, psi0_expr)
p = particles(x, [s], mesh)

# Initialize advection class, use RK3 scheme
ap = advect_rk3(p, V, uh, "closed")

# Init projection and get initial condition
lstsq_psi = l2projection(p, W, 1)
lstsq_psi.project(psi_h, lb, ub)
assign(psi_h0, psi_h)

step = 0
t = 0.0
area_0 = assemble(psi_h * dx)
(psi_h_min, psi_h_max) = (0.0, 0.0)

timer = Timer()
timer.start()

# Write initial field and dump initial particle field
outfile.write_checkpoint(psi_h, function_name="psi", time_step=0)
p.dump2file(mesh, fname_list, property_list, "wb")

while step < num_steps:
    step += 1
    t += float(dt)

    if comm.Get_rank() == 0:
        print("Step " + str(step))

    # Advect and project
    ap.do_step(float(dt))
    lstsq_psi.project(psi_h, lb, ub)

    psi_h_min = min(psi_h_min, psi_h.vector().min())
    psi_h_max = max(psi_h_max, psi_h.vector().max())
    if comm.rank == 0:
        print("Min max phi {} {}".format(psi_h_min, psi_h_max))

    if step % store_step == 0:
        outfile.write_checkpoint(psi_h, function_name="psi", time_step=t, append=True)
        # Dump particles to file
        p.dump2file(mesh, fname_list, property_list, "ab")

timer.stop()

area_end = assemble(psi_h * dx)
num_part = p.number_of_particles()
l2_error = np.sqrt(abs(assemble((psi_h0 - psi_h) * (psi_h0 - psi_h) * dx)))
if comm.Get_rank() == 0:
    print("Num cells " + str(mesh.num_entities_global(2)))
    print("Num particles " + str(num_part))
    print("Elapsed time " + str(timer.elapsed()[0]))
    print("Area error " + str(abs(area_end - area_0)))
    print("Error " + str(l2_error))
    print("Min max phi {} {}".format(psi_h_min, psi_h_max))

outfile.close()
