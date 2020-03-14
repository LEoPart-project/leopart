# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import (
    RectangleMesh,
    FiniteElement,
    VectorElement,
    MixedElement,
    FunctionSpace,
    Function,
    SubDomain,
    Constant,
    Point,
    XDMFFile,
    Expression,
    MeshFunction,
    Measure,
    DirichletBC,
    assign,
    project,
    near,
    assemble,
    between,
    MPI,
    Timer,
    TimingClear,
    TimingType,
    timings,
)
from leopart import (
    particles,
    PDEStaticCondensation,
    RandomRectangle,
    advect_rk3,
    StokesStaticCondensation,
    BinaryBlock,
    l2projection,
    FormsPDEMap,
    FormsStokes,
    assign_particle_values,
)
from mpi4py import MPI as pyMPI
import numpy as np
import pickle
import shutil as sht


"""
    Dambreak problem, with the same setup as Lobovsky et al (2014)
    https://doi.org/10.1016/j.jfluidstructs.2014.03.009
"""


comm = pyMPI.COMM_WORLD


class Boundaries(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class RightBoundary(SubDomain):
    def __init__(self, xmax):
        SubDomain.__init__(self)
        self.xmax = xmax

    def inside(self, x, on_boundary):
        return near(x[0], self.xmax)


class RightBoundary_Segment(SubDomain):
    def __init__(self, xmax, ymin, ymax):
        SubDomain.__init__(self)
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def inside(self, x, on_boundary):
        return near(x[0], self.xmax) and between(x[1], (self.ymin, self.ymax))


class Corner(SubDomain):
    def __init__(self, xc, yc):
        SubDomain.__init__(self)
        self.xc, self.yc = xc, yc

    def inside(self, x, on_boundary):
        return near(x[0], self.xc) and near(x[1], self.yc)


# Lores
# xmin, xmax = 0., 1.62
# ymin, ymax = 0., 1.
# nx, ny = 81, 100
# xmin_rho1 = 0.
# xmax_rho1 = 0.6
# ymin_rho1 = 0.
# ymax_rho1 = 0.3
# pres = 700
# res = 'low'
# dt = Constant(2.5e-3)
# store_step = 40

# Medium
# xmin, xmax = 0., 1.61
# ymin, ymax = 0., 1.
# nx, ny = 161, 100
# xmin_rho1 = 0.
# xmax_rho1 = 0.6
# ymin_rho1 = 0.
# ymax_rho1 = 0.3
# pres = 1200
# res = 'medium'
# dt = Constant(1.e-3)
# store_step = 100

# Hires
xmin, xmax = 0.0, 1.61
ymin, ymax = 0.0, 1.0
nx, ny = 322, 200
xmin_rho1 = 0.0
xmax_rho1 = 0.6
ymin_rho1 = 0.0
ymax_rho1 = 0.3
pres = 2200
res = "high"
dt = Constant(5.0e-4)
store_step = 200

mu = 1e-2
theta_p = 0.5
theta_L = Constant(1.0)

probe_radius = 0.01
probe1_y = 0.003
probe2_y = 0.015
probe3_y = 0.03
probe4_y = 0.08

probe1_loc = Point(xmax - 1e-10, probe1_y)
probe2_loc = Point(xmax - 1e-10, probe2_y)
probe3_loc = Point(xmax - 1e-10, probe3_y)
probe4_loc = Point(xmax - 1e-10, probe4_y)

# Specify body force
f = Constant((0, -9.81))

geometry = {"xmin": xmin_rho1, "xmax": xmax_rho1, "ymin": ymin_rho1, "ymax": ymax_rho1}

rho1 = Constant(1000.0)
rho2 = Constant(1.0)

# Polynomial order
k = 1
kbar = k
alpha = Constant(6.0 * k * k)

# Time stepping
T_end = 1.4
num_steps = int(T_end // float(dt) + 1)
print(num_steps)

# Directory for output
outdir_base = (
    "./../../results/Dambreak_mu"
    + str(float(mu))
    + "_theta"
    + str(float(theta_p))
    + "_res_"
    + res
    + "/"
)

# Particle output
fname_list = [outdir_base + "xp.pickle", outdir_base + "up.pickle", outdir_base + "rhop.pickle"]
property_list = [0, 2, 1]
pressure_table = outdir_base + "wall_pressure.pickle"

mesh = RectangleMesh(MPI.comm_world, Point(xmin, ymin), Point(xmax, ymax), nx, ny, "left")
bbt = mesh.bounding_box_tree()

# xdmf output
xdmf_u = XDMFFile(mesh.mpi_comm(), outdir_base + "u.xdmf")
xdmf_p = XDMFFile(mesh.mpi_comm(), outdir_base + "p.xdmf")
xdmf_rho = XDMFFile(mesh.mpi_comm(), outdir_base + "rho.xdmf")

# Function Spaces density tracking/pressure
T_1 = FunctionSpace(mesh, "DG", 0)
Q_E_Rho = FiniteElement("DG", mesh.ufl_cell(), k)

# Vector valued function spaces for specific momentum tracking
W_E_2 = VectorElement("DG", mesh.ufl_cell(), k)
T_E_2 = VectorElement("DG", mesh.ufl_cell(), 0)
Wbar_E_2 = VectorElement("DGT", mesh.ufl_cell(), kbar)
Wbar_E_2_H12 = VectorElement("CG", mesh.ufl_cell(), kbar)["facet"]

# Function spaces for Stokes
Q_E = FiniteElement("DG", mesh.ufl_cell(), 0)
Qbar_E = FiniteElement("DGT", mesh.ufl_cell(), k)

# For Stokes
mixedL = FunctionSpace(mesh, MixedElement([W_E_2, Q_E]))
mixedG = FunctionSpace(mesh, MixedElement([Wbar_E_2_H12, Qbar_E]))

W_2 = FunctionSpace(mesh, W_E_2)
T_2 = FunctionSpace(mesh, T_E_2)
Wbar_2 = FunctionSpace(mesh, Wbar_E_2)
Wbar_2_H12 = FunctionSpace(mesh, Wbar_E_2_H12)
Q_Rho = FunctionSpace(mesh, Q_E_Rho)
Qbar = FunctionSpace(mesh, Qbar_E)

# Define some functions
rho, rho0, rho00 = Function(Q_Rho), Function(Q_Rho), Function(Q_Rho)
rhobar = Function(Qbar)
u0, ustar = Function(W_2), Function(W_2)
ustar_bar = Function(Wbar_2)
duh0, duh00 = Function(W_2), Function(W_2)

ubar0_a = Function(Wbar_2_H12)
Udiv = Function(W_2)
Uh = Function(mixedL)
Uhbar = Function(mixedG)
U0 = Function(mixedL)
Uhbar0 = Function(mixedG)

# Set initial density field
initial_density = BinaryBlock(geometry, float(rho1), float(rho2), degree=1)
zero_expression = Expression(("0.", "0."), degree=1)

# Initialize particles
x = RandomRectangle(Point(xmin, ymin), Point(xmax, ymax)).generate(
    [pres, int(pres * (ymax - ymin) / (xmax - xmin))]
)
up = assign_particle_values(x, zero_expression)
rhop = assign_particle_values(x, initial_density)

# Increment requires dup to be stored, init zero
dup = up

p = particles(x, [rhop, up, dup], mesh)

# Init rho0 field
lstsq_rho = l2projection(p, Q_Rho, 1)
lstsq_rho.project(rho0, float(rho2), float(rho1))

# Initialize advection class
ap = advect_rk3(p, W_2, Udiv, "closed")

# Set-up boundary conditions (free slip)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
all_bounds = Boundaries()
all_bounds.mark(boundaries, 98)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Mark pressure boundary
pressure_transducer = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
pressure_transducer.set_all(0)
probe_1 = RightBoundary_Segment(xmax, probe1_y - probe_radius, probe1_y + probe_radius)
probe_2 = RightBoundary_Segment(xmax, probe2_y - probe_radius, probe2_y + probe_radius)
probe_3 = RightBoundary_Segment(xmax, probe3_y - probe_radius, probe3_y + probe_radius)
probe_4 = RightBoundary_Segment(xmax, probe4_y - probe_radius, probe4_y + probe_radius)
probe_1.mark(pressure_transducer, 1)
probe_2.mark(pressure_transducer, 2)
probe_3.mark(pressure_transducer, 3)
probe_4.mark(pressure_transducer, 4)
dsp = Measure("ds", domain=mesh, subdomain_data=pressure_transducer)

# Set-up density projection
funcspaces_rho = {"FuncSpace_local": Q_Rho, "FuncSpace_lambda": T_1, "FuncSpace_bar": Qbar}
forms_rho = FormsPDEMap(mesh, funcspaces_rho).forms_theta_linear(
    rho0, ubar0_a, dt, theta_map=Constant(1.0), theta_L=Constant(0.0), zeta=Constant(20.0)
)
pde_rho = PDEStaticCondensation(
    mesh,
    p,
    forms_rho["N_a"],
    forms_rho["G_a"],
    forms_rho["L_a"],
    forms_rho["H_a"],
    forms_rho["B_a"],
    forms_rho["Q_a"],
    forms_rho["R_a"],
    forms_rho["S_a"],
    1,
)

# Set-up momentum projection
FuncSpace_u = {"FuncSpace_local": W_2, "FuncSpace_lambda": T_2, "FuncSpace_bar": Wbar_2}
forms_u = FormsPDEMap(mesh, FuncSpace_u).forms_theta_nlinear_multiphase(
    rho,
    rho0,
    rho00,
    rhobar,
    u0,
    ubar0_a,
    dt,
    theta_map=Constant(1.0),
    theta_L=theta_L,
    duh0=duh0,
    duh00=duh00,
)
pde_u = PDEStaticCondensation(
    mesh,
    p,
    forms_u["N_a"],
    forms_u["G_a"],
    forms_u["L_a"],
    forms_u["H_a"],
    forms_u["B_a"],
    forms_u["Q_a"],
    forms_u["R_a"],
    forms_u["S_a"],
    2,
)

# Set-up Stokes Solve
forms_stokes = FormsStokes(mesh, mixedL, mixedG, alpha, ds=ds).forms_multiphase(
    rho0, ustar, dt, mu, f
)
ssc = StokesStaticCondensation(
    mesh,
    forms_stokes["A_S"],
    forms_stokes["G_S"],
    forms_stokes["B_S"],
    forms_stokes["Q_S"],
    forms_stokes["S_S"],
)

# Set pressure in upper left corner to zero
bc1 = DirichletBC(mixedG.sub(1), Constant(0), Corner(xmin, ymax), "pointwise")
bcs = [bc1]


lstsq_u = l2projection(p, W_2, 2)

# Loop and output
step = 0
t = 0.0

# Store at step 0
xdmf_rho.write(rho0, t)
xdmf_u.write(Uh.sub(0), t)
xdmf_p.write(Uh.sub(1), t)

p.dump2file(mesh, fname_list, property_list, "wb")

dump_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
with open(pressure_table, "wb") as PT:
    pickle.dump(dump_list, PT)


timer = Timer("[P] Total time consumed")
timer.start()

while step < num_steps:
    step += 1
    t += float(dt)

    if comm.Get_rank() == 0:
        print("Step " + str(step) + ", time = " + str(t))

    # Advect
    t1 = Timer("[P] advect particles")
    ap.do_step(float(dt))
    del t1

    # Project density and specific momentum
    t1 = Timer("[P] density projection")
    pde_rho.assemble(True, True)
    pde_rho.solve_problem(rhobar, rho, "mumps", "default")
    del t1

    t1 = Timer("[P] momentum projection")
    pde_u.assemble(True, True)

    try:
        pde_u.solve_problem(ustar_bar, ustar, "mumps", "default")
    except Exception:
        # FIXME: work-around
        lstsq_u.project(ustar)
    del t1

    # Solve Stokes
    t1 = Timer("[P] Stokes assemble ")
    ssc.assemble_global()
    for bc in bcs:
        ssc.apply_boundary(bc)
    del t1

    t1 = Timer("[P] Stokes solve ")
    ssc.solve_problem(Uhbar, Uh, "mumps", "default")
    del t1

    t1 = Timer("[P] Update mesh fields")
    # Needed for particle advection
    assign(Udiv, Uh.sub(0))

    # Needed for constrained map
    assign(rho0, rho)
    assign(ubar0_a, Uhbar.sub(0))
    assign(u0, ustar)
    assign(duh00, duh0)
    assign(duh0, project(Uh.sub(0) - ustar, W_2))
    del t1

    t1 = Timer("[P] Update particle field")
    p.increment(Udiv, ustar, np.array([2, 3], dtype=np.uintp), theta_p, step)
    del t1

    if step == 2:
        theta_L.assign(1.0)

    if step % store_step == 0:
        # Set output, also throw out particle output
        xdmf_rho.write(rho, t)
        xdmf_u.write(Uh.sub(0), t)
        xdmf_p.write(Uh.sub(1), t)

        # Save particle data
        p.dump2file(mesh, fname_list, property_list, "ab", False)
        comm.barrier()

    # Save pressure measurements
    P1 = assemble(Uh.sub(1) * dsp(1)) / assemble(Constant(1.0) * dsp(1))
    P2 = assemble(Uh.sub(1) * dsp(2)) / assemble(Constant(1.0) * dsp(2))
    P3 = assemble(Uh.sub(1) * dsp(3)) / assemble(Constant(1.0) * dsp(3))
    P4 = assemble(Uh.sub(1) * dsp(4)) / assemble(Constant(1.0) * dsp(4))

    P1_bar = assemble(Uhbar.sub(1) * dsp(1)) / assemble(Constant(1.0) * dsp(1))
    P2_bar = assemble(Uhbar.sub(1) * dsp(2)) / assemble(Constant(1.0) * dsp(2))
    P3_bar = assemble(Uhbar.sub(1) * dsp(3)) / assemble(Constant(1.0) * dsp(3))
    P4_bar = assemble(Uhbar.sub(1) * dsp(4)) / assemble(Constant(1.0) * dsp(4))

    P1_point, P2_point, P3_point, P4_point = None, None, None, None

    if bool(bbt.compute_entity_collisions(probe1_loc)):
        P1_point = Uh.sub(1)(probe1_loc)

    if bool(bbt.compute_entity_collisions(probe2_loc)):
        P2_point = Uh.sub(1)(probe2_loc)

    if bool(bbt.compute_entity_collisions(probe3_loc)):
        P3_point = Uh.sub(1)(probe3_loc)

    if bool(bbt.compute_entity_collisions(probe4_loc)):
        P4_point = Uh.sub(1)(probe4_loc)

    # Gather on zero
    P1_point = comm.gather(P1_point, root=0)
    P2_point = comm.gather(P2_point, root=0)
    P3_point = comm.gather(P3_point, root=0)
    P4_point = comm.gather(P4_point, root=0)

    if comm.rank == 0:
        P1_val = next(pval for pval in P1_point if pval is not None)
        P2_val = next(pval for pval in P2_point if pval is not None)
        P3_val = next(pval for pval in P3_point if pval is not None)
        P4_val = next(pval for pval in P4_point if pval is not None)
        dump_list = [
            t,
            P1,
            P2,
            P3,
            P4,
            P1_bar,
            P2_bar,
            P3_bar,
            P4_bar,
            P1_val,
            P2_val,
            P3_val,
            P4_val,
        ]
        with open(pressure_table, "ab") as PT:
            pickle.dump(dump_list, PT)

xdmf_u.close()
xdmf_rho.close()
xdmf_p.close()

time_table = timings(TimingClear.keep, [TimingType.wall])
with open(outdir_base + "timings" + str(nx) + ".log", "w") as out:
    out.write(time_table.str(True))

if comm.rank == 0:
    sht.copy2("./Dambreak_Lobovsky.py", outdir_base)
