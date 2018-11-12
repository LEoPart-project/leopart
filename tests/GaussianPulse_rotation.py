# __author__ = 'Jakob Maljaars <j.m.maljaars@tudelft.nl>'
# __date__   = '2018-08'
# __copyright__ = 'Copyright (C) 2011' + __author__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

"""
    Tests the advection of a Gaussian pulse
    on a circular disk, using solid body rotation
"""

from dolfin import *
from mpi4py import MPI as pyMPI
import numpy as np
import os

# Load from package
from DolfinParticles import (particles, advect_rk3,
                     PDEStaticCondensation, RandomCircle,
                     FormsPDEMap, GaussianPulse)

#set_log_level(PROGRESS)
comm = pyMPI.COMM_WORLD

# Geometric properties
x0,y0   = 0., 0.        # Center of domain
xc,yc   = -0.15, 0.     # Center of Gaussian
r       = 0.5            # Radius of domain
sigma   = Constant(0.1) # stdev of Gaussian

# Mesh/particle properties, use safe number of particles
nx_list   = [1, 2, 4, 8, 16, 32]
pres_list = [120 * pow(2,i) for i in range(len(nx_list))]

# Polynomial order
k_list = [1, 2]          # Third order does not make sense for 3rd order advection scheme
l_list = [0] * len(k_list)
kbar_list = k_list

# Magnitude solid body rotation .
Uh = np.pi

# Timestepping info, Tend corresponds to 2 rotations
Tend            = 2.
dt_list         = [Constant(0.08/( pow(2,i)) ) for i in range(len(nx_list)) ]
storestep_list  = [1 * pow(2,i) for i in range(len(dt_list))]

# Directory for output
outdir_base = './../results/GaussianPulse_Rotation/'

# Then start the loop over the tests set-ups
for (k,l,kbar) in zip(k_list, l_list, kbar_list):
    outdir      = outdir_base+'k'+str(k)+'l'+str(l)+'kbar' \
                    +str(kbar)+'_nproc'+str(comm.Get_size())+'/'

    output_table = outdir+'output_table.txt'
    if comm.rank == 0:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        with open(output_table, "w") as write_file:
             write_file.write("%-12s %-15s %-20s %-10s %-20s %-20s %-10s %-20s \n" % ("Time step", "Number of cells", "Number of particles",
                                                                          "L2 T_half", "Global mass T_half",
                                                                          "L2 T_end", "Global mass T_end", "Wall clock time"))

    for (nx, dt, pres, store_step) in zip(nx_list, dt_list, pres_list, storestep_list):
        if comm.Get_rank() == 0:
            print("Starting computation with grid resolution "+str(nx))
        output_field = File(outdir+'psi_h'+'_nx'+str(nx)+'.pvd')

        # Compute num steps till completion
        num_steps = np.rint(Tend/float(dt))

        # Generate mesh
        mesh = Mesh('./../meshes/circle_0.xml')
        n = nx
        while (n > 1):
            mesh = refine(mesh)
            n /= 2
        print("Mesh (", nx, ") contains ", mesh.num_cells(), " cells")

        # Velocity and initial condition
        V   = VectorFunctionSpace(mesh,'DG', 3)
        uh  = Function(V)
        uh.assign( Expression( ('-Uh*x[1]','Uh*x[0]'),Uh = Uh, degree=3) )

        psi0_expression = GaussianPulse( center = (xc, yc), sigma = float(sigma),
                                        U = [Uh, Uh], time = 0., height = 1., degree = 3 )

        # Generate particles
        if comm.Get_rank() == 0:
            x    =  RandomCircle(Point(x0, y0), r).generate([pres, pres])
            s    =  np.zeros((len(x), 1), dtype = np.float_)
        else:
            x = None
            s = None

        x = comm.bcast(x, root=0)
        s = comm.bcast(s, root=0)

        # Initialize particles with position x and scalar property s at the mesh
        p   = particles(x, [s], mesh)
        property_idx = 1 # Scalar quantity is stored at slot 1

        # Initialize advection class, use RK3 scheme
        ap  = advect_rk3(p, V, uh, 'open')

        # Define the variational (projection problem)
        W_e    = FiniteElement("DG", mesh.ufl_cell(), k)
        T_e    = FiniteElement("DG", mesh.ufl_cell(), 0)
        Wbar_e = FiniteElement("DGT", mesh.ufl_cell(), k)

        W      = FunctionSpace(mesh, W_e)
        T      = FunctionSpace(mesh, T_e)
        Wbar   = FunctionSpace(mesh, Wbar_e)

        psi_h    = Function(W); psi0_h = Function(W)
        lambda_h = Function(T)
        psibar_h = Function(Wbar)

        # Boundary conditions
        bc = DirichletBC(Wbar, Constant(0.), "on_boundary")

        # Initialize forms
        FuncSpace_adv = {'FuncSpace_local': W, 'FuncSpace_lambda': T, 'FuncSpace_bar': Wbar}
        forms_pde     = FormsPDEMap(mesh, FuncSpace_adv).forms_theta_linear(psi0_h, uh, dt, Constant(1.0) )
        pde_projection = PDEStaticCondensation(mesh,p,  forms_pde['N_a'], forms_pde['G_a'], forms_pde['L_a'],
                                                                                            forms_pde['H_a'],
                                                                                            forms_pde['B_a'],
                                                        forms_pde['Q_a'], forms_pde['R_a'], forms_pde['S_a'],
                                                                                            [],property_idx)

        # Set initial condition at mesh and particles
        psi0_h.interpolate(psi0_expression)
        p.interpolate(psi0_h.cpp_object(), property_idx)

        step = 0
        area_0   = assemble(psi0_h*dx)
        timer    = Timer()
        if comm.Get_rank() == 0:
            progress = Progress("Doing step...", int(num_steps))

        timer.start()
        while step < num_steps:
            step += 1

            # Advect particle, assemble and solve pde projection
            ap.do_step(float(dt))
            pde_projection.assemble(True, True)
            pde_projection.apply_boundary(bc)
            pde_projection.solve_problem(psibar_h.cpp_object(),
                                         psi_h.cpp_object(), lambda_h.cpp_object(),
                                         'gmres', 'hypre_amg')
            # Update old solution
            assign(psi0_h, psi_h)

            # Store field
            if step % store_step is 0 or step is 1:
                output_field << psi_h

            # In order to avoid getting accused of cheating, compute
            # L2 error and mass error at half rotation
            if int(np.floor(2*step  - num_steps )) == 0:
                l2_error_half     = sqrt(assemble(dot(psi_h - psi0_expression, psi_h - psi0_expression)*dx) )
                area_half = assemble(psi_h*dx)

            # Update progress bar
            if comm.Get_rank() == 0:
                progress += 1
        timer.stop()

        # Compute error (we should accurately recover initial condition)
        psi0_expression.t = step * float(dt)
        l2_error = sqrt(assemble(dot(psi_h - psi0_expression, psi_h - psi0_expression)*dx) )

        # The global mass conservation error should be zero
        area_end = assemble(psi_h*dx)

        if comm.Get_rank() == 0:
            print("l2 error "+str(l2_error))

            # Store in error error table
            num_cells_t   = mesh.num_entities_global(2)
            num_particles = len(x)
            try:
                area_error_half = np.float64((area_half-area_0))
            except:
                area_error_half = float('NaN')
                l2_error_half   = float('NaN')

            area_error_end  = np.float64((area_end-area_0))

            with open(output_table, "a") as write_file:
                write_file.write("%-12.5g %-15d %-20d %-10.2e %-20.3g %-20.2e %-20.3g %-20.3g \n" %
                                (float(dt), int(num_cells_t), int(num_particles),
                                 float(l2_error_half), np.float64(area_error_half),
                                 float(l2_error), np.float64(area_error_end), np.float(timer.elapsed()[0])))
