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
    UserExpression,
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
)
from mpi4py import MPI as pyMPI
import numpy as np

# Load from package
from leopart import (
    particles,
    advect_rk3,
    l2projection,
    RandomCircle,
    AddDelete,
    assign_particle_values,
)

comm = pyMPI.COMM_WORLD


# TODO: consider placing in InitialConditions
class SlottedDisk(UserExpression):
    def __init__(self, radius, center, width, depth, lb=0.0, ub=1.0, **kwargs):
        self.r = radius
        self.width = width
        self.depth = depth
        self.center = center
        self.lb = lb
        self.ub = ub
        super().__init__(self, **kwargs)

    def eval(self, value, x):
        xc = self.center[0]
        yc = self.center[1]

        if ((x[0] - xc) ** 2 + (x[1] - yc) ** 2 <= self.r ** 2) and not (
            (xc - self.width) <= x[0] <= (xc + self.width) and x[1] >= yc + self.depth
        ):
            value[0] = self.ub
        else:
            value[0] = self.lb

    def value_shape(self):
        return ()


# Domain properties
x0, y0 = 0.0, 0.0
xc, yc = -0.15, 0.0
r = 0.5
rdisk = 0.2
rwidth = 0.05
lb = -1.0
ub = 3.0

# Mesh/particle resolution
nx = 64
pres = 800

# Polynomial order for bounded l2 map
k = 1

# Magnitude solid body rotation .
Uh = np.pi

# Timestepping
Tend = 2.0
dt = Constant(0.02)
num_steps = np.rint(Tend / float(dt))

# Output directory
store_step = 1
outdir = "./../../results/SlottedDisk_Rotation_AddDelete/"

# Mesh
mesh = Mesh("./../../meshes/circle_0.xml")
mesh = refine(mesh)
mesh = refine(mesh)
mesh = refine(mesh)

outfile = XDMFFile(mesh.mpi_comm(), outdir + "psi_h.xdmf")

# Set slotted disk
psi0_expr = SlottedDisk(
    radius=rdisk, center=[xc, yc], width=rwidth, depth=0.0, degree=3, lb=lb, ub=ub
)

# Function space and velocity field
W = FunctionSpace(mesh, "DG", k)
psi_h = Function(W)

V = VectorFunctionSpace(mesh, "DG", 3)
uh = Function(V)
uh.assign(Expression(("-Uh*x[1]", "Uh*x[0]"), Uh=Uh, degree=3))

# Generate particles
x = RandomCircle(Point(x0, y0), r).generate([pres, pres])
s = assign_particle_values(x, psi0_expr)

p = particles(x, [s], mesh)
# Initialize advection class, use RK3 scheme
ap = advect_rk3(p, V, uh, "closed")
# Init projection
lstsq_psi = l2projection(p, W, 1)

# Do projection to get initial field
lstsq_psi.project(psi_h.cpp_object(), lb, ub)
AD = AddDelete(p, 10, 20, [psi_h], [1], [lb, ub])

step = 0
t = 0.0
area_0 = assemble(psi_h * dx)
timer = Timer()
timer.start()

outfile.write(psi_h, t)
while step < num_steps:
    step += 1
    t += float(dt)

    if comm.Get_rank() == 0:
        print("Step " + str(step))

    AD.do_sweep()
    ap.do_step(float(dt))
    AD.do_sweep_failsafe(4)

    lstsq_psi.project(psi_h.cpp_object(), lb, ub)

    if step % store_step == 0:
        outfile.write(psi_h, t)

timer.stop()

area_end = assemble(psi_h * dx)
if comm.Get_rank() == 0:
    print("Num cells " + str(mesh.num_entities_global(2)))
    print("Num particles " + str(len(x)))
    print("Elapsed time " + str(timer.elapsed()[0]))
    print("Area error " + str(abs(area_end - area_0)))
