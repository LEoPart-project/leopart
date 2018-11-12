# __author__ = 'Jakob Maljaars <j.m.maljaars@tudelft.nl>'
# __date__   = '2018-08'
# __copyright__ = 'Copyright (C) 2018' + __author__
# __license__  = 'PLEASE DO NOT SHARE WITHOUT AUTHOR CONSENT'

from dolfin import *
import numpy as np
import time as tm
from mpi4py import MPI as pyMPI
from DolfinParticles import (particles, advect_rk3, RandomBox,
                             AddDelete, PDEStaticCondensation,
                             StokesStaticCondensation, l2projection,
                             FormsPDEMap, FormsStokes)
import os

# parameters["krylov_solver"]["monitor_convergence"] = True
# parameters["krylov_solver"]["report"] = True
#parameters.parse()

#set_log_level(10)

comm = pyMPI.COMM_WORLD

class PeriodicBoundary(SubDomain):
    def __init__(self, ddict):
        SubDomain.__init__(self)
        self.xmin, self.xmax = ddict['xmin'], ddict['xmax']
        self.ymin, self.ymax = ddict['ymin'], ddict['ymax']
        self.zmin, self.zmax = ddict['zmin'], ddict['zmax']

    def inside(self, x, on_boundary):
        xmin, xmax, ymin, ymax, zmin, zmax = self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax
        return  bool( ( near(x[0], xmin) or  near(x[1], ymin) or near(x[2], zmin)) and \
                    (not (( near(x[0], xmin) and near(x[1], ymax) ) or \
                          ( near(x[0], xmin) and near(x[2], zmax) ) or \
                          ( near(x[1], ymin) and near(x[2], zmax) ) or \
                          ( near(x[1], ymax) and near(x[2], zmin) ) or \
                          ( near(x[0], xmax) and near(x[2], zmin) ) or \
                          ( near(x[0], xmax) and near(x[1], ymin) )    )) )

    def map(self, x, y):
        xmin, xmax, ymin, ymax, zmin, zmax = self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax
        # Is this corner projection really needded?
        if near(x[0], xmax) and near(x[1], ymax) and near(x[2], zmax):
            y[0] = x[0] - (xmax - xmin)
            y[1] = x[1] - (ymax - ymin)
            y[2] = x[2] - (zmax - zmin)
        ## This handles the outer edges
        elif near(x[0], xmax) and near(x[2], zmax):
            y[0] = x[0] - (xmax - xmin)
            y[1] = x[1]
            y[2] = x[2] - (zmax - zmin)
        elif near(x[0], xmax) and near(x[1], ymax):
            y[0] = x[0] - (xmax - xmin)
            y[1] = x[1] - (ymax - ymin)
            y[2] = x[2]
        elif near(x[1], ymax) and near(x[2], zmax):
            y[0] = x[0]
            y[1] = x[1] - (ymax - ymin)
            y[2] = x[2] - (zmax - zmin)
        # This finally handles the locations on the inner edges
        elif near(x[0], xmax):
            y[0] = x[0] - (xmax - xmin)
            y[1] = x[1]
            y[2] = x[2]
        elif near(x[1], ymax):
            y[0] = x[0]
            y[1] = x[1] - (ymax - ymin)
            y[2] = x[2]
        elif near(x[2], zmax):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2]- (zmax - zmin)

class Corner(SubDomain):
    def __init__(self, ddict):
        SubDomain.__init__(self)
        self.xmax, self.ymax, self.zmax = ddict['xmax'], ddict['ymax'], ddict['zmax']

    def inside(self, x, on_boundary):
        return near(x[0], self.xmax) and near(x[1], self.ymax) and near(x[2], zmax)

def assign_particle_values(x, u_exact):
    if comm.Get_rank() == 0:
        s=np.asarray([u_exact(x[i,:]) for i in range(len(x))], dtype = np.float_)
    else:
        s = None
    return s

################# USER INPUT #############
geometry = {'xmin': -1., 'ymin': -1., 'zmin': -1, 'xmax': 1., 'ymax': 1., 'zmax': 1.}
#geometry = {'xmin': -np.pi, 'ymin': -np.pi., 'zmin': -np.pi, 'xmax': np.pi, 'ymax': np.pi, 'zmax': np.pi}

# Mesh resolution
nx, ny, nz = 48, 48, 48

# Particle resolution
pres = 200

# Time stepping
Tend = 7.5
dt = Constant(1.25e-2)

# Viscosity
nu = Constant(1. /(np.pi * float(1600))) #Constant(1e-3)

# Stokes related
k       = 1
kbar    = k

alpha = Constant(6*k*k)
beta  = Constant(0.)

# Theta value to compute v*
theta_init = 1.0
theta_next = 0.5
theta_L    = Constant(theta_init)

# Theta value for updating particles
theta_p    = 0.5

# Taylor-Green wave mode
mode  = 1.

# Directory for output
outdir_base = './../results/TaylorGreen_3D_l2_nx48/'
output_table = outdir_base+'output_table.txt'
store_step  = 8
###################################

U_exact = (' U*sin(mode*pi*(x[0])) * cos(mode*pi*(x[1])) * cos(mode*pi*(x[2]))',
           '-U*cos(mode*pi*(x[0])) * sin(mode*pi*(x[1])) * cos(mode*pi*(x[2]))',
           ' 0.')
u_exact = Expression(U_exact, degree = 7, U=float(1.), nu=float(nu), mode = mode)

f = Constant((0.,0.,0.))

# Create mesh
xmin, ymin, zmin = geometry['xmin'], geometry['ymin'], geometry['zmin']
xmax, ymax, zmax = geometry['xmax'], geometry['ymax'], geometry['zmax']

mesh = BoxMesh(MPI.comm_world, Point(xmin,ymin,zmin), Point(xmax,ymax,zmax), nx, ny, nz)
bmesh= BoundaryMesh(mesh, 'exterior')
pbc  = PeriodicBoundary(geometry)

# Set output/xdmf output
if comm.rank == 0:
    if not os.path.exists(outdir_base):
        os.makedirs(outdir_base)
        
    with open(output_table, "w") as write_file:
        write_file.write("%-12s %-20s %-20s \n" % ("Time level", "Kinetic energy", "Vorticity"))
        
xdmf_u = XDMFFile(mesh.mpi_comm(), outdir_base+"u.xdmf")
xdmf_p = XDMFFile(mesh.mpi_comm(), outdir_base+"p.xdmf")
xdmf_curl = XDMFFile(mesh.mpi_comm(), outdir_base+"curl.xdmf")

# Required elements
W_E_2   = VectorElement("DG", mesh.ufl_cell(), k)
T_E_2   = VectorElement("DG", mesh.ufl_cell(), 0)
Wbar_E_2= VectorElement("DGT", mesh.ufl_cell(), kbar)
Wbar_E_2_H12 = VectorElement("CG", mesh.ufl_cell(), kbar)["facet"]

Q_E    = FiniteElement("DG", mesh.ufl_cell(), k-1)
Qbar_E = FiniteElement("DGT", mesh.ufl_cell(), k)

# Function spaces for projection
W_2     = FunctionSpace(mesh,W_E_2)
T_2     = FunctionSpace(mesh,T_E_2)
Wbar_2  = FunctionSpace(mesh,Wbar_E_2, constrained_domain = pbc)
Wbar_2_H12 = FunctionSpace(mesh, Wbar_E_2_H12, constrained_domain = pbc)

# Function spaces for Stokes
mixedL = FunctionSpace(mesh, MixedElement([W_E_2,Q_E]))
mixedG = FunctionSpace(mesh, MixedElement([Wbar_E_2_H12,Qbar_E]), constrained_domain = pbc)

# Define functions
u0_a    = Function(W_2)
u_l2    = Function(W_2)
ustar   = Function(W_2)
duh0    = Function(W_2)
duh00   = Function(W_2)

ubar0_a = Function(Wbar_2_H12)
ubar_a  = Function(Wbar_2)
Udiv    = Function(W_2)

Uh    = Function(mixedL)
Uhbar = Function(mixedG)
U0    = Function(mixedL)
Uhbar0= Function(mixedG)
lamb  = Function(T_2)

u0_a.assign(u_exact)
ubar0_a.assign(u_exact)
Udiv.assign(u_exact)

curl_func = Function(W_2)

# Store zero level
curl_func.assign(project(curl(u0_a), W_2) )

e_kin = 0.5 * assemble(dot(u0_a, u0_a) * dx)
enstrophy = 0.5 * assemble(dot(curl_func, curl_func) * dx )

if comm.rank == 0:
    with open(output_table, "a") as write_file:
        write_file.write("%-12.5g %-20.6g %-20.6g \n" % (float(0.), float(e_kin), float(enstrophy)))


# Initialize particles
if comm.Get_rank() == 0:
    x    =  RandomBox(Point(xmin, ymin, zmin), Point(xmax, ymax, zmax)).generate([pres, pres, pres])
    s    =  assign_particle_values(x, u_exact)
else:
    x = None
    s = None
x = comm.bcast(x, root=0)
s = comm.bcast(s, root=0)

lims = np.array( [[xmin, xmin, ymin, ymax, zmin, zmax],[xmax, xmax, ymin, ymax, zmin, zmax],
                  [xmin, xmax, ymin, ymin, zmin, zmax],[xmin, xmax, ymax, ymax, zmin, zmax],
                  [xmin, xmax, ymin, ymax, zmin, zmin],[xmin, xmax, ymin, ymax, zmax, zmax]] )

# Particle specific momentum is stored at slot 1
# the second slot will be to store old velocities at particle level
property_idx = 1
p   = particles(x, [s,s], mesh)
ap  = advect_rk3(p, W_2, Udiv, bmesh, 'periodic', lims.flatten(), 'none')

if comm.Get_rank() == 0:
    print("Particles initialized")

# Particle management
AD = AddDelete(p, 10, 20, [Udiv, duh0])

# Forms PDE map
#funcspace_dict = {'FuncSpace_local': W_2, 'FuncSpace_lambda': T_2, 'FuncSpace_bar': Wbar_2}
#forms_adv      = FormsPDEMap(mesh, funcspace_dict, beta_map=Constant(1e-5)).forms_theta_nlinear(u0_a, ubar0_a, dt, \
#                    theta_map = Constant(1.0), theta_L = theta_L, duh0 = duh0, duh00 = duh00)
#pde_projection = PDEStaticCondensation(mesh,p, forms_adv['N_a'], forms_adv['G_a'], forms_adv['L_a'],
#                                                                                  forms_adv['H_a'],
#                                                                                   forms_adv['B_a'],
#                                               forms_adv['Q_a'], forms_adv['R_a'], forms_adv['S_a'],
#                                                                                        property_idx)

# Forms Stokes
# Set pressure in corner to zero
bc1 = DirichletBC(mixedG.sub(1), Constant(0), Corner(geometry), "pointwise")
bcs = [bc1]

forms_stokes   = FormsStokes(mesh,mixedL,mixedG, alpha).forms_unsteady(ustar,dt,nu,f)

ssc = StokesStaticCondensation(mesh, forms_stokes['A_S'],forms_stokes['G_S'],
                                                         forms_stokes['B_S'],
                                    forms_stokes['Q_S'], forms_stokes['S_S'], bcs)

lstsq_u = l2projection(p, W_2, 1)

# Prepare time stepping loop
num_steps = np.rint(Tend/float(dt))
step      = 0
t         = 0.

timer = Timer()
timer.start()

while step < num_steps:
    step += 1
    t    += float(dt)
    if comm.rank == 0:
        print('Step number '+str(step))

    # Limit number of particles
    t1 = Timer("[P] advect particles")
    AD.do_sweep()

    # Advect particles
    ap.do_step(float(dt))

    # Do failsafe sweep
    AD.do_sweep_failsafe(5)
    del(t1)

    # Do l2 project
    t1 = Timer("[P] l2 projection")
    lstsq_u.project(ustar.cpp_object())
    del(t1)

    if comm.Get_rank() == 0:
        print("Finished local l2 projection")
    
    # Do constrained projection
    #t1 = Timer("[P] assemble projection")
    #pde_projection.assemble(True, True)
    #del(t1)
    #t1 = Timer("[P] solve projection")
    #pde_projection.solve_problem(ubar_a.cpp_object(), ustar.cpp_object(), lamb.cpp_object(), 'bicgstab', 'hypre_amg') 
    #pde_projection.solve_problem(ubar_a.cpp_object(), ustar.cpp_object(), 'gmres', 'amg')
    #del(t1)

    comm.barrier()
    
    # Solve Stokes
    t1 = Timer("[P] Stokes assemble ")
    ssc.assemble_global_system(True)
    del(t1)
    #comm.barrier()

    #print("Apply BC")
    #for bc in bcs:
    #    ssc.apply_boundary(bc)
    #del(t1)
        
    t1 = Timer("[P] Stokes solve")
    ssc.solve_problem(Uhbar.cpp_object(), Uh.cpp_object(), "gmres", "hypre_amg")
    #ssc.solve_problem(Uhbar.cpp_object(), Uh.cpp_object(), "mumps", "default")
    del(t1)
    
    # Needed for particle advection
    assign(Udiv, Uh.sub(0))

    # Needed for constrained map
    assign(ubar0_a, Uhbar.sub(0))
    assign(u0_a, ustar);  assign(duh00, duh0); assign(duh0, project(Uh.sub(0)-ustar,W_2))

    #p.interpolate(Udiv.cpp_object(), 1)
    #p.increment(Udiv.cpp_object(), ustar, 1)
    p.increment(Udiv.cpp_object(), ustar.cpp_object(), np.array([1,2], dtype = np.uintp), theta_p, step)

    if step == 2:
        theta_L.assign(theta_next)

    # Compute vorticity                                                                                                                                                                              
    curl_func.assign( project(curl(Uh.sub(0)), W_2) )

    if step % store_step is 0:
        xdmf_u.write(Uh.sub(0), t)
        xdmf_p.write(Uh.sub(1), t)

        # Compute vorticity
        xdmf_curl.write(curl_func, t)

    e_kin = 0.5 * assemble(dot(Uh.sub(0), Uh.sub(0)) * dx)
    enstrophy = 0.5 * assemble(dot(curl_func, curl_func) * dx )

    if comm.rank == 0:
        with open(output_table, "a") as write_file:
            write_file.write("%-12.5g %-20.6g %-20.6g \n" % (float(t), float(e_kin), float(enstrophy)))

timer.stop()

# Compute errors
ex = as_vector((1., 0., 0.))
ey = as_vector((0., 1., 0.))
ez = as_vector((0., 0., 1.))

momentum = assemble( (dot(Uh.sub(0), ex) + dot(Uh.sub(0), ey) + dot(Uh.sub(0), ez)) * dx )

if comm.Get_rank() == 0:
    print("Momentum "+str(momentum))
    print("Elapsed time "+str(timer.elapsed()[0]))

list_timings(TimingClear.keep, [TimingType.wall])
time_table = timings(TimingClear.keep, [TimingType.wall])
with open(outdir_base+"timings.log", "w") as out:
    out.write(time_table.str(True))
