from dolfin import (RectangleMesh, FiniteElement, VectorElement, MixedElement, FunctionSpace,
                    Function, SubDomain, Constant, Point, XDMFFile, Expression, MeshFunction,
                    Measure, DirichletBC, assign, project, near,
                    MPI, Timer, TimingClear, TimingType, timings)
from DolfinParticles import (particles, PDEStaticCondensation, RandomRectangle, advect_rk3,
                             StokesStaticCondensation, Sinusoidal, l2projection, FormsPDEMap,
                             FormsStokes)
from mpi4py import MPI as pyMPI
import numpy as np
import shutil as sht


'''
    Standing wave problem in a closed box
'''


comm = pyMPI.COMM_WORLD


class Boundaries(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class Corner(SubDomain):
    def __init__(self, xc, yc):
        SubDomain.__init__(self)
        self.xc, self.yc = xc, yc

    def inside(self, x, on_boundary):
        return near(x[0], self.xc) and near(x[1], self.yc)


def assign_particle_values(x, u_exact):
    if comm.Get_rank() == 0:
        s = np.asarray([u_exact(x[i, :]) for i in range(len(x))], dtype=np.float_)
    else:
        s = None
    return s


xmin, xmax = 0., 8.
ymin, ymax = 0., 8

# Medium
nx, ny = 80, 80
pres = 800
res = 'medium'
dt = Constant(0.01)
store_field_step = 10
store_particle_step = 20

# Hires
# nx, ny = 160, 160
# pres = 1600
# res = 'high'
# dt = Constant(0.005)
# store_field_step = 10
# store_particle_step = 20

mu = Constant(5.)
theta_p = 0.5
theta_L = Constant(1.0)

A = 0.2
g = 9.81
d = 4.
length = xmax - xmin
mode = 2.
phase = -np.pi / 2.
geometry = {'amplitude': A, 'depth': d, 'length': length, 'mode': mode, 'phase': phase}
km = mode * np.pi / length

# Specify body force
f = Constant((0, -9.81))

rho1 = Constant(1000.)
rho2 = Constant(1.)

# Polynomial order
k = 1
kbar = k
alpha = Constant(6.*k*k)

# Time stepping
T_end = 20
num_steps = int(T_end // float(dt) + 1)


# Directory for output
outdir_base = "./../../results/StandingWave_Periodic_mu" + str(float(mu)) + \
              "_theta"+str(float(theta_p))+"_res_"+res+"/"

# Particle output
fname_list = [outdir_base+'xp.pickle',
              outdir_base+'up.pickle',
              outdir_base+'rhop.pickle']
property_list = [0, 2, 1]

mesh = RectangleMesh(MPI.comm_world, Point(xmin, ymin), Point(xmax, ymax), nx, ny)

# Set-up boundary conditions (free slip)
allbounds = Boundaries()
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
allbounds.mark(boundaries, 98)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

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
initial_density = Sinusoidal(geometry, float(rho1), float(rho2), degree=1)
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

# Initialize advection class, set all boundaries to "open"
ap = advect_rk3(p, W_2, Udiv, "open")

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
                                                                                dt, mu, f)
ssc = StokesStaticCondensation(mesh,
                               forms_stokes['A_S'], forms_stokes['G_S'],
                               forms_stokes['G_ST'], forms_stokes['B_S'],
                               forms_stokes['Q_S'], forms_stokes['S_S'])

# Set pressure in upper left corner to zero
bc1 = DirichletBC(mixedG.sub(1), Constant(0), Corner(xmin, ymax), "pointwise")
bcs = [bc1]

lstsq_u = l2projection(p, W_2, 2)

# Loop and output
step = 0
t = 0.

# Store at step 0
xdmf_rho.write(rho0, t)
xdmf_u.write(Uh.sub(0), t)
xdmf_p.write(Uh.sub(1), t)

p.dump2file(mesh, fname_list, property_list, 'wb')

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

    try:
        pde_u.solve_problem(ustar_bar, ustar, "mumps", "default")
    except Exception:
        # FIXME: work-around
        lstsq_u.project(ustar)
    del(t1)

    # Solve Stokes
    t1 = Timer("[P] Stokes assemble ")
    ssc.assemble_global()
    for bc in bcs:
        ssc.apply_boundary(bc)
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

    if step % store_field_step is 0:
        # Set output, also throw out particle output
        xdmf_rho.write(rho, t)
        xdmf_u.write(Uh.sub(0), t)
        xdmf_p.write(Uh.sub(1), t)

    if step % store_particle_step is 0:
        # Save particle data
        p.dump2file(mesh, fname_list, property_list, 'ab', False)
        comm.barrier()

xdmf_u.close()
xdmf_rho.close()
xdmf_p.close()

time_table = timings(TimingClear.keep, [TimingType.wall])
with open(outdir_base+"timings"+str(nx)+".log", "w") as out:
    out.write(time_table.str(True))

if comm.rank == 0:
    sht.copy2('./StandingWave.py', outdir_base)
