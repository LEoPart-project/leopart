from dolfin import (RectangleMesh, FiniteElement, VectorElement, MixedElement, FunctionSpace,
                    Function, SubDomain, Constant, Point, XDMFFile, Expression, MeshFunction,
                    Measure, DirichletBC, assign, project, near,
                    MPI, Timer, TimingClear, TimingType, timings, VectorFunctionSpace, ALE,
                    BoundaryMesh, FacetNormal, outer, dot, assemble,
                    dx, as_vector)
from DolfinParticles import (particles, PDEStaticCondensation, RandomRectangle, advect_rk3,
                             StokesStaticCondensation, l2projection, FormsPDEMap,
                             FormsStokes, PeriodicPiston, BinaryBlock, AddDelete)
from mpi4py import MPI as pyMPI
import numpy as np
import shutil as sht


'''
    Wave flume, monochromatic wave train is generated
    by a moving boundary
'''


comm = pyMPI.COMM_WORLD


class Boundaries(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class Left(SubDomain):
    def __init__(self, xmin):
        SubDomain.__init__(self)
        self.xmin = xmin

    def inside(self, x, on_boundary):
        return near(x[0], self.xmin)


class Right(SubDomain):
    def __init__(self, xmax):
        SubDomain.__init__(self)
        self.xmax = xmax

    def inside(self, x, on_boundary):
        return near(x[0], self.xmax)


class Bottom(SubDomain):
    def __init__(self, ymin):
        SubDomain.__init__(self)
        self.ymin = ymin

    def inside(self, x, on_boundary):
        return near(x[1], ymin)


class Top(SubDomain):
    def __init__(self, ymax):
        SubDomain.__init__(self)
        self.ymax = ymax

    def inside(self, x, on_boundary):
        return near(x[1], ymax)


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


def strip_meshfunc(mesh_function, marker):
    """
    This functions takes a mesh function and strips
    it into parts according to the specified marker value.
    It returns the stripped mesh_function.array()
    Working for size_t markers only!
    """
    sub2bound = np.where(mesh_function.array() == marker)[0].astype(dtype=np.uintp)
    return sub2bound


xmin, xmax = 0., 40.
ymin, ymax = 0., 4.

# Medium
nx, ny = 400, 80
pres = 3000
res = 'medium'
dt = Constant(0.025)
store_field_step = 4
store_particle_step = 40
store_probe_step = 2

# Hires
# nx, ny = 800, 160
# pres = 7000
# res = 'high'
# dt = Constant(0.0125)
# store_field_step = 8
# store_particle_step = 80
# store_probe_step = 4

mu = Constant(1e-1)
theta_p = 0.5
theta_L = Constant(1.0)

# Target wave to be generated
d = 2.5
omega = 3.83
H = 0.2
g = 9.81
x0 = xmin
xL = 5.

# Class for computing mesh velocity
dU = PeriodicPiston(x0, xL, float(dt), 0.,
                    omega, d, H, g, Tramp=0., degree=1)

# Geometry initial condition
geometry = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': d}

# Specify body force
f = Constant((0, -g))

method = 'l2'
rho1 = Constant(1000.)
rho2 = Constant(1.)

# Polynomial order
k = 1
kbar = k
alpha = Constant(6.*k*k)

# Time stepping
T_end = 30.
num_steps = int(T_end // float(dt) + 1)

# Directory for output
outdir_base = "./../../results/WaveTrain_"\
              "theta"+str(float(theta_p))+"_res_"+res+"_mu"+str(float(mu))+"_l2/"

meta_data = outdir_base+"meta_data.txt"
conservation_data = outdir_base+"conservation_data.txt"

# Particle output
fname_list = [outdir_base+'xp.pickle',
              outdir_base+'up.pickle',
              outdir_base+'rhop.pickle']
property_list = [0, 2, 1]

xmin_probe1, xmax_probe1 = 4.8, 5.2
xmin_probe2, xmax_probe2 = 7.3, 7.7
xmin_probe3, xmax_probe3 = 9.8, 10.2

fname_probe1 = [outdir_base+'xp_probe05.pickle',
                outdir_base+'up_probe05.pickle',
                outdir_base+'rhop_probe05.pickle']

fname_probe2 = [outdir_base+'xp_probe75.pickle',
                outdir_base+'up_probe75.pickle',
                outdir_base+'rhop_probe75.pickle']

fname_probe3 = [outdir_base+'xp_probe10.pickle',
                outdir_base+'up_probe10.pickle',
                outdir_base+'rhop_probe10.pickle']

mesh = RectangleMesh(MPI.comm_world, Point(xmin, ymin), Point(xmax, ymax), nx, ny)
# bmesh = BoundaryMesh(mesh, 'exterior')

# Helper vectors
ex = as_vector([1.0, 0.0])
ey = as_vector([0.0, 1.0])
n = FacetNormal(mesh)

# Set-up boundary conditions (free slip)
allbounds = Boundaries()
btop = Top(ymax)
bbottom = Bottom(ymin)
bleft = Left(xmin)
bright = Right(xmax)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
bbottom.mark(boundaries, 98)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Mark facets, needed for particle boundaries
# facet_domains = MeshFunction("size_t", bmesh, bmesh.topology().dim())
# facet_domains.set_all(0)
# btop.mark(facet_domains, 1)
# bleft.mark(facet_domains, 2)
# tbound = strip_meshfunc(facet_domains, 1)
# rbound = strip_meshfunc(facet_domains, 0)
# lbound = strip_meshfunc(facet_domains, 2)

# xdmf output
xdmf_u = XDMFFile(mesh.mpi_comm(), outdir_base+"u.xdmf")
xdmf_p = XDMFFile(mesh.mpi_comm(), outdir_base+"p.xdmf")
xdmf_rho = XDMFFile(mesh.mpi_comm(), outdir_base+"rho.xdmf")

# Function Spaces density tracking/pressure
T_1 = FunctionSpace(mesh, 'DG', 0)
Q_E_Rho = FiniteElement("DG", mesh.ufl_cell(), 1)

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

# Mesh motion
Wcg = VectorFunctionSpace(mesh, 'CG', 1)

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

# Mesh velocity
umesh = Function(Wcg)
u_ALE = ubar0_a - umesh

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

# Initialize advection class, set all boundaries to "open"
# ap = advect_rk3(p, W_2, Udiv, bmesh, "closed", np.hstack([rbound, lbound]), "open", tbound)
# Workaround for boundary mesh in parallel
ap = advect_rk3(p, W_2, Udiv, "open")

# Particle management
AD = AddDelete(p, 20, 35, [rho, Udiv, duh0], [1], [float(rho2), float(rho1)])

# Set-up density projection
funcspaces_rho = {'FuncSpace_local': Q_Rho, 'FuncSpace_lambda': T_1, 'FuncSpace_bar': Qbar}
forms_rho = FormsPDEMap(mesh, funcspaces_rho).forms_theta_linear(rho0, u_ALE, dt,
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
                                                                        u0, u_ALE, dt,
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
u_D = Constant(0.0)
bcL = DirichletBC(mixedG.sub(0).sub(0), u_D, bleft)
bcR = DirichletBC(mixedG.sub(0).sub(0), Constant(0.0), bright)
bcP = DirichletBC(mixedG.sub(1), Constant(0), Corner(xmin, ymax), "pointwise")
bcs = [bcL, bcR, bcP]

lstsq_u = l2projection(p, W_2, 2)

# Loop and output
step = 0
t = 0.

# Store at step 0
assign(rho, rho0)
xdmf_rho.write_checkpoint(rho, "rho", t)
xdmf_u.write(Uh.sub(0), t)
xdmf_p.write(Uh.sub(1), t)

p.dump2file(mesh, fname_list, property_list, 'wb')
p.particle_probe(mesh, xmin_probe1, xmax_probe1, fname_probe1, property_list, 'wb')
p.particle_probe(mesh, xmin_probe2, xmax_probe2, fname_probe2, property_list, 'wb')
p.particle_probe(mesh, xmin_probe3, xmax_probe3, fname_probe3, property_list, 'wb')

# Save metadata
num_cells_t = mesh.num_entities_global(2)
num_particles = len(x)

with open(meta_data, "w") as write_file:
    write_file.write("%-12s %-15s %-20s \n" %
                     ("Time step", "Number of cells", "Number of particles"))
    write_file.write("%-12.5g %-15d %-20d \n" % (float(dt), num_cells_t, num_particles))

# Table for mass and momentum conservation errors
with open(conservation_data, "w") as write_file:
    write_file.write("%-12s %-20s %-20s \n" %
                     ("Time", "Mass conservation", "Momentum conservation"))

timer = Timer("[P] Total time consumed")
timer.start()

while step < num_steps:
    step += 1
    t += float(dt)

    if comm.Get_rank() == 0:
        print("Step "+str(step)+', time = '+str(t))

    # Advect
    t1 = Timer("[P] advect particles")
    if step > 1:
        AD.do_sweep()

    ap.do_step(float(dt))
    AD.do_sweep_failsafe(4)
    del(t1)

    # Compute mass and momentum on old configuration
    old_area = assemble(rho0*dx)
    momentum_oldconfig_x = assemble(dot(rho0 * Uh.sub(0), ex) * dx)
    momentum_oldconfig_y = assemble(dot(rho0 * Uh.sub(0), ey) * dx)

    # Assemble RHS on old mesh configuration
    pde_rho.assemble_state_rhs()
    pde_u.assemble_state_rhs()

    # Move mesh
    if step > 1:
        umesh.assign(project(dU, Wcg))
        ALE.move(mesh, project(dU * dt, Wcg))
        p.relocate()

    dU.mesh_velocity()
    u_D.assign(Constant(dU.du))

    # Project density and specific momentum
    t1 = Timer("[P] density projection")
    pde_rho.assemble(True, False)
    pde_rho.solve_problem(rhobar, rho, "superlu_dist", "default")
    del(t1)

    t1 = Timer("[P] momentum projection")
    pde_u.assemble(True, False)
    try: 
        pde_u.solve_problem(ustar_bar, ustar, "superlu_dist", "default")
    except Exception:
        # FIXME: work-around
        print("Solve by l2")
        lstsq_u.project(ustar)    
    del(t1)

    # Compute mass and boundary flux
    new_area = assemble(rho*dx)
    bflux = assemble(dt * dot(u_ALE, n) * rhobar * ds)

    momentum_newconfig_x = assemble(rho * dot(ustar, ex) * dx +
                                    dt * dot(outer(rhobar * ustar_bar, u_ALE) * n, ex) * ds)
    momentum_newconfig_y = assemble(rho * dot(ustar, ey) * dx +
                                    dt * dot(outer(rhobar * ustar_bar, u_ALE) * n, ey) * ds)

    # Solve Stokes
    t1 = Timer("[P] Stokes assemble ")
    ssc.assemble_global()
    for bc in bcs:
        ssc.apply_boundary(bc)
    del(t1)

    t1 = Timer("[P] Stokes solve ")
    ssc.solve_problem(Uhbar, Uh, "superlu_dist", "default")
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
        xdmf_rho.write_checkpoint(rho, "rho", t, append=True)
        xdmf_u.write(Uh.sub(0), t)
        xdmf_p.write(Uh.sub(1), t)

        num_particles = p.number_of_particles(mesh)

        if comm.Get_rank() == 0:
            mass_change = new_area - old_area + bflux
            momentum_change_x = momentum_newconfig_x - momentum_oldconfig_x
            momentum_change_y = momentum_newconfig_y - momentum_oldconfig_y
            momentum_change = momentum_change_x + momentum_change_y

            # Write to file
            with open(meta_data, "a") as write_file:
                write_file.write("%-12.5g %-15d %-20d \n" %
                                 (t, num_cells_t, num_particles))

            with open(conservation_data, "a") as write_file:
                write_file.write("%-12.5g %-20.3g %-20.3g \n" %
                                 (t, mass_change, momentum_change))

            print('Mass balance '+str(mass_change))
            print("Momentum change over map"+str(momentum_change))

    if step % store_particle_step is 0:
        # Save particle data
        p.dump2file(mesh, fname_list, property_list, 'ab', False)

    # Save around probes
    if step % store_probe_step is 0:
        p.particle_probe(mesh, xmin_probe1, xmax_probe1, fname_probe1, property_list, 'ab')
        p.particle_probe(mesh, xmin_probe2, xmax_probe2, fname_probe2, property_list, 'ab')
        p.particle_probe(mesh, xmin_probe3, xmax_probe3, fname_probe3, property_list, 'ab')

xdmf_u.close()
xdmf_rho.close()
xdmf_p.close()

time_table = timings(TimingClear.keep, [TimingType.wall])
with open(outdir_base+"timings"+str(nx)+".log", "w") as out:
    out.write(time_table.str(True))

if comm.rank == 0:
    sht.copy2('./WaveTrain_l2.py', outdir_base)
