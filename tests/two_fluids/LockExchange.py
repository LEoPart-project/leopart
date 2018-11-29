# __author__ = 'Jakob Maljaars <j.m.maljaars@tudelft.nl>'
# __date__   = '2018-08'
# __copyright__ = 'Copyright (C) 2011' + __author__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import (RectangleMesh, FiniteElement, VectorElement, MixedElement, FunctionSpace,
                    Function, SubDomain, Constant, Point, XDMFFile, Expression, MeshFunction,
                    Measure, assign, project, as_vector, assemble, dot, outer, dx, FacetNormal,
                    MPI, Timer, TimingClear, TimingType, timings)
from DolfinParticles import (particles, PDEStaticCondensation, RandomRectangle, advect_rk3,
                             StokesStaticCondensation, BinaryBlock, l2projection, FormsPDEMap,
                             FormsStokes)
from mpi4py import MPI as pyMPI
import numpy as np

comm = pyMPI.COMM_WORLD

'''
    Density driven gravity current.
    For description of test, see Birman et al: The non-Boussinesq lock-exchange problem.
    Part 2. High-resolution simulations (2005) doi:10.1017/S002211200500503
'''


class Boundaries(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


def assign_particle_values(x, u_exact):
    if comm.Get_rank() == 0:
        s = np.asarray([u_exact(x[i, :]) for i in range(len(x))], dtype=np.float_)
    else:
        s = None
    return s


# User input
xmin, xmax = 0., 30.
ymin, ymax = -0.5, 0.5
xmin_rho1 = xmin
xmax_rho1 = 14.
ymin_rho1 = ymin
ymax_rho1 = ymax
nx, ny = 2000, 80
pres = 16800
g = -9.81
Re = 4000.
theta_p = 0.5
theta_L = Constant(1.0)

# Specify body force
f = Constant((0, g))

geometry = {'xmin': xmin_rho1, 'xmax': xmax_rho1, 'ymin': ymin_rho1, 'ymax': ymax_rho1}

gamma = 0.92
rho1 = Constant(1000.)
rho2 = Constant(gamma * float(rho1))

g_prime = abs(g) * (1. - gamma)
ub = np.sqrt(g_prime * (ymax - ymin))
nu = Constant((ub * (ymax - ymin)) / Re)

# Polynomial order
k = 1
kbar = k
alpha = Constant(6.*k*k)

# Time stepping
T_star_end = 16.
tscale = np.sqrt(g_prime / (ymax - ymin))
T_end = T_star_end / tscale
dt = Constant(1.25e-2/tscale)
num_steps = int(T_end // float(dt) + 1)

if comm.rank == 0:
    print('{:=^72}'.format('Computation for gamma '+str(gamma)))
    print('Time scale is '+str(tscale))
    print('End time of simulation set to '+str(T_end))
    print('Time step set to '+str(float(dt)))
    print('Number of steps '+str(num_steps))

# T_end = 16.
# dt = Constant(2.e-2)
# num_steps = int(T_end // float(dt) + 1)

# Directory for output
outdir_base = './../../results/LockExchange_hires/'
# Particle output
fname_list = [outdir_base+'xp.pickle',
              outdir_base+'up.pickle',
              outdir_base+'rhop.pickle']
property_list = [0, 2, 1]
store_step = 20

# Helper vectors
ex = as_vector([1.0, 0.0])
ey = as_vector([0.0, 1.0])

mesh = RectangleMesh(MPI.comm_world, Point(xmin, ymin), Point(xmax, ymax), nx, ny)
n = FacetNormal(mesh)

# xdmf output
xdmf_u = XDMFFile(mesh.mpi_comm(), outdir_base+"u.xdmf")
xdmf_p = XDMFFile(mesh.mpi_comm(), outdir_base+"p.xdmf")
xdmf_rho = XDMFFile(mesh.mpi_comm(), outdir_base+"rho.xdmf")

# Function Spaces density tracking/pressure
T_1 = FunctionSpace(mesh, 'DG', 0)
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
if comm.Get_rank() == 0:
    x = RandomRectangle(Point(xmin, ymin),
                        Point(xmax, ymax)).generate([pres, int(pres * (ymax-ymin) / (xmax-xmin))])
    up = assign_particle_values(x, zero_expression)
    rhop = assign_particle_values(x, initial_density)
else:
    x = None
    up = None
    rhop = None

x = comm.bcast(x, root=0)
up = comm.bcast(up, root=0)
rhop = comm.bcast(rhop, root=0)
# Increment requires dup to be stored, init zero
dup = up

p = particles(x, [rhop, up, dup], mesh)

# Init rho0 field
lstsq_rho = l2projection(p, Q_Rho, 1)
lstsq_rho.project(rho0, float(rho2), float(rho1))

# Initialize advection class
ap = advect_rk3(p, W_2, Udiv, 'closed')

# Set-up boundary conditions (free slip)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
all_bounds = Boundaries()
all_bounds.mark(boundaries, 98)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Set-up density projection
funcspaces_rho = {'FuncSpace_local': Q_Rho, 'FuncSpace_lambda': T_1, 'FuncSpace_bar': Qbar}
forms_rho = FormsPDEMap(mesh, funcspaces_rho).forms_theta_linear(rho0, ubar0_a, dt,
                                                                 theta_map=Constant(1.0),
                                                                 theta_L=Constant(0.),
                                                                 zeta=Constant(20.))
pde_rho = PDEStaticCondensation(mesh, p,
                                forms_rho['N_a'], forms_rho['G_a'], forms_rho['L_a'],
                                forms_rho['H_a'],
                                forms_rho['B_a'],
                                forms_rho['Q_a'], forms_rho['R_a'], forms_rho['S_a'], 1)

# Set-up momentum projection
FuncSpace_u = {'FuncSpace_local': W_2, 'FuncSpace_lambda': T_2, 'FuncSpace_bar': Wbar_2}
forms_u = FormsPDEMap(mesh, FuncSpace_u).forms_theta_nlinear_multiphase(rho, rho0, rho00, rhobar,
                                                                        u0, ubar0_a, dt,
                                                                        theta_map=Constant(1.0),
                                                                        theta_L=theta_L,
                                                                        duh0=duh0, duh00=duh00)
pde_u = PDEStaticCondensation(mesh, p,
                              forms_u['N_a'], forms_u['G_a'], forms_u['L_a'],
                              forms_u['H_a'],
                              forms_u['B_a'],
                              forms_u['Q_a'], forms_u['R_a'], forms_u['S_a'], 2)

# Set-up Stokes Solve
forms_stokes = FormsStokes(mesh, mixedL, mixedG, alpha, ds=ds).forms_multiphase(rho, ustar,
                                                                                dt, rho * nu, f)
ssc = StokesStaticCondensation(mesh,
                               forms_stokes['A_S'], forms_stokes['G_S'],
                               forms_stokes['B_S'],
                               forms_stokes['Q_S'], forms_stokes['S_S'])

# Loop and output
step = 0
t = 0.

# Store tstep 0
xdmf_rho.write(rho0, t)
xdmf_u.write(Uh.sub(0), t)
xdmf_p.write(Uh.sub(1), t)
p.dump2file(mesh, fname_list, property_list, 'wb')
comm.barrier()

timer = Timer("[P] Total time consumed")
timer.start()

while step < num_steps:
    step += 1
    t += float(dt)

    if comm.Get_rank() == 0:
        print("Step "+str(step)+', time = '+str(t))

    # Advect
    t1 = Timer("[P] advect particles")
    ap.do_step(float(dt))
    del(t1)

    # Project density and specific momentum
    t1 = Timer("[P] density projection")
    pde_rho.assemble(True, True)
    pde_rho.solve_problem(rhobar, rho, "mumps", "default")
    del(t1)

    t1 = Timer("[P] momentum projection")
    pde_u.assemble(True, True)
    pde_u.solve_problem(ustar_bar, ustar, "mumps", "default")
    del(t1)

    # Check (global) momentum conservation
    mx_change = assemble((rho * dot(ustar, ex) - dot(rho0 * Uh.sub(0), ex))/dt * dx
                         + dot(outer(rhobar * ustar_bar, ubar0_a) * n, ex) * ds)
    my_change = assemble((rho * dot(ustar, ey) - dot(rho0 * Uh.sub(0), ey))/dt * dx
                         + dot(outer(rhobar * ustar_bar, ubar0_a)*n, ey) * ds)
    mt_change = mx_change + my_change
    if comm.rank == 0:
        print("Momentum change over map "+str(mt_change))

    # Solve Stokes
    t1 = Timer("[P] Stokes assemble ")
    ssc.assemble_global()
    del(t1)

    t1 = Timer("[P] Stokes solve ")
    ssc.solve_problem(Uhbar, Uh, "mumps", "default")
    del(t1)

    t1 = Timer("[P] Update mesh fields")
    # Needed for particle advection
    assign(Udiv, Uh.sub(0))

    # Needed for constrained map
    assign(rho0, rho)
    assign(ubar0_a, Uhbar.sub(0))
    assign(u0, ustar)
    assign(duh00, duh0)
    assign(duh0, project(Uh.sub(0)-ustar, W_2))
    del(t1)

    t1 = Timer("[P] Update particle field")
    p.increment(Udiv, ustar, np.array([2, 3], dtype=np.uintp), theta_p, step)
    del(t1)

    if step == 2:
        theta_L.assign(1.0)

    if step % store_step is 0:
        # Set output, also throw out particle output
        xdmf_rho.write(rho, t)
        xdmf_u.write(Uh.sub(0), t)
        xdmf_p.write(Uh.sub(1), t)

        # Save particle data
        p.dump2file(mesh, fname_list, property_list, 'ab', False)
        comm.barrier()
timer.stop()

xdmf_u.close()
xdmf_rho.close()
xdmf_p.close()

time_table = timings(TimingClear.keep, [TimingType.wall])
with open(outdir_base+"timings"+str(nx)+".log", "w") as out:
    out.write(time_table.str(True))
