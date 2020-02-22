# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
    Tests the advection of a Gaussian pulse
    on a circular disk, using solid body rotation
"""

from dolfin import (
    Mesh,
    FiniteElement,
    Constant,
    VectorFunctionSpace,
    Function,
    FunctionSpace,
    Expression,
    Point,
    DirichletBC,
    assign,
    sqrt,
    dot,
    assemble,
    dx,
    refine,
    XDMFFile,
    Timer,
    TimingType,
    TimingClear,
    timings,
)
from mpi4py import MPI as pyMPI
import numpy as np
import os

# Load from package
from leopart import (
    particles,
    advect_rk3,
    PDEStaticCondensation,
    RandomCircle,
    FormsPDEMap,
    GaussianPulse,
    AddDelete,
)

comm = pyMPI.COMM_WORLD

# Geometric properties
x0, y0 = 0.0, 0.0
xc, yc = -0.15, 0.0
r = 0.5
sigma = Constant(0.1)

# Mesh/particle properties, use safe number of particles
i_list = [i for i in range(5)]
nx_list = [pow(2, i) for i in i_list]
pres_list = [160 * pow(2, i) for i in i_list]

# Polynomial order
k_list = [1, 2]  # Third order does not make sense for 3rd order advection scheme
l_list = [0] * len(k_list)
kbar_list = k_list

# Magnitude solid body rotation .
Uh = np.pi

# Timestepping info, Tend corresponds to 2 rotations
Tend = 2.0
dt_list = [Constant(0.08 / (pow(2, i))) for i in i_list]
storestep_list = [1 * pow(2, i) for i in i_list]

# Directory for output
outdir_base = "./../../results/GaussianPulse_Rotation/"

# Then start the loop over the tests set-ups
for (k, l, kbar) in zip(k_list, l_list, kbar_list):
    outdir = (
        outdir_base
        + "k"
        + str(k)
        + "l"
        + str(l)
        + "kbar"
        + str(kbar)
        + "_nproc"
        + str(comm.Get_size())
        + "/"
    )

    output_table = outdir + "output_table.txt"
    if comm.rank == 0:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        with open(output_table, "w") as write_file:
            write_file.write(
                "%-12s %-15s %-20s %-10s %-20s %-20s %-10s %-20s \n"
                % (
                    "Time step",
                    "Number of cells",
                    "Number of particles",
                    "L2 T_half",
                    "Global mass T_half",
                    "L2 T_end",
                    "Global mass T_end",
                    "Wall clock time",
                )
            )

    for (nx, dt, pres, store_step) in zip(nx_list, dt_list, pres_list, storestep_list):
        if comm.Get_rank() == 0:
            print("Starting computation with grid resolution " + str(nx))

        # Compute num steps till completion
        num_steps = np.rint(Tend / float(dt))

        # Generate mesh
        mesh = Mesh("./../../meshes/circle_0.xml")
        n = nx
        while n > 1:
            mesh = refine(mesh)
            n /= 2

        output_field = XDMFFile(mesh.mpi_comm(), outdir + "psi_h_nx" + str(nx) + ".xdmf")

        # Velocity and initial condition
        V = VectorFunctionSpace(mesh, "DG", 3)
        uh = Function(V)
        uh.assign(Expression(("-Uh*x[1]", "Uh*x[0]"), Uh=Uh, degree=3))

        psi0_expression = GaussianPulse(
            center=(xc, yc), sigma=float(sigma), U=[Uh, Uh], time=0.0, height=1.0, degree=3
        )

        # Generate particles
        x = RandomCircle(Point(x0, y0), r).generate([pres, pres])
        s = np.zeros((len(x), 1), dtype=np.float_)

        # Initialize particles with position x and scalar property s at the mesh
        p = particles(x, [s], mesh)
        property_idx = 1  # Scalar quantity is stored at slot 1

        # Initialize advection class, use RK3 scheme
        ap = advect_rk3(p, V, uh, "open")

        # Define the variational (projection problem)
        W_e = FiniteElement("DG", mesh.ufl_cell(), k)
        T_e = FiniteElement("DG", mesh.ufl_cell(), 0)
        Wbar_e = FiniteElement("DGT", mesh.ufl_cell(), k)

        W = FunctionSpace(mesh, W_e)
        T = FunctionSpace(mesh, T_e)
        Wbar = FunctionSpace(mesh, Wbar_e)

        psi_h, psi0_h = Function(W), Function(W)
        lambda_h = Function(T)
        psibar_h = Function(Wbar)

        # Boundary conditions
        bc = DirichletBC(Wbar, Constant(0.0), "on_boundary")

        # Initialize forms
        FuncSpace_adv = {"FuncSpace_local": W, "FuncSpace_lambda": T, "FuncSpace_bar": Wbar}
        forms_pde = FormsPDEMap(mesh, FuncSpace_adv).forms_theta_linear(
            psi0_h, uh, dt, Constant(1.0)
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
            property_idx,
        )

        # Set initial condition at mesh and particles
        psi0_h.interpolate(psi0_expression)
        p.interpolate(psi0_h, property_idx)

        # Initialize add/delete for safety
        AD = AddDelete(p, 15, 25, [psi0_h])

        step = 0
        t = 0.0
        area_0 = assemble(psi0_h * dx)
        timer = Timer()

        timer.start()
        while step < num_steps:
            step += 1
            t += float(dt)

            if comm.rank == 0:
                print("Step  " + str(step))

            # Advect particle, assemble and solve pde projection
            t1 = Timer("[P] Advect particles step")
            ap.do_step(float(dt))
            AD.do_sweep_failsafe(4 * k)
            del t1

            t1 = Timer("[P] Assemble PDE system")
            pde_projection.assemble(True, True)
            # pde_projection.apply_boundary(bc)
            del t1

            t1 = Timer("[P] Solve PDE constrained projection")
            pde_projection.solve_problem(psibar_h, psi_h, "mumps", "default")
            del t1

            t1 = Timer("[P] Update and store")
            # Update old solution
            assign(psi0_h, psi_h)

            # Store field
            if step % store_step == 0 or step == 1:
                output_field.write(psi_h, t)

            # Avoid getting accused of cheating, compute
            # L2 error and mass error at half rotation
            if int(np.floor(2 * step - num_steps)) == 0:
                psi0_expression.t = step * float(dt)
                l2_error_half = sqrt(
                    assemble(dot(psi_h - psi0_expression, psi_h - psi0_expression) * dx)
                )
                area_half = assemble(psi_h * dx)
            del t1
        timer.stop()

        # Compute error (we should accurately recover initial condition)
        psi0_expression.t = step * float(dt)
        l2_error = sqrt(assemble(dot(psi_h - psi0_expression, psi_h - psi0_expression) * dx))

        # The global mass conservation error should be zero
        area_end = assemble(psi_h * dx)

        if comm.Get_rank() == 0:
            print("l2 error " + str(l2_error))

            # Store in error error table
            num_cells_t = mesh.num_entities_global(2)
            num_particles = len(x)
            try:
                area_error_half = np.float64((area_half - area_0))
            except BaseException:
                area_error_half = float("NaN")
                l2_error_half = float("NaN")

            area_error_end = np.float64((area_end - area_0))

            with open(output_table, "a") as write_file:
                write_file.write(
                    "%-12.5g %-15d %-20d %-10.2e %-20.3g %-20.2e %-20.3g %-20.3g \n"
                    % (
                        float(dt),
                        int(num_cells_t),
                        int(num_particles),
                        float(l2_error_half),
                        np.float64(area_error_half),
                        float(l2_error),
                        np.float64(area_error_end),
                        np.float(timer.elapsed()[0]),
                    )
                )

        time_table = timings(TimingClear.keep, [TimingType.wall])
        with open(outdir + "timings" + str(nx) + ".log", "w") as out:
            out.write(time_table.str(True))
