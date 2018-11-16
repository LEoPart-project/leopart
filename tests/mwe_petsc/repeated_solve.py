"""
    MWE for petsc-issue
    fails for resolution_level = 4 
    and running
    mpirun -np 32 python3 repeated_solve.py
"""

from dolfin import (Mesh, FiniteElement, Constant, VectorFunctionSpace, Function, FunctionSpace,
                    Expression, Point, DirichletBC, assign, sqrt, dot, assemble, dx,
                    refine, Timer, TimingType, TimingClear, list_timings)
from mpi4py import MPI as pyMPI
import numpy as np

# Load from package
from DolfinParticles import (particles, PDEStaticCondensation, RandomCircle,
                             FormsPDEMap, GaussianPulse)

# set_log_level(10)
comm = pyMPI.COMM_WORLD

# Geometric properties
x0, y0 = 0., 0.
xc, yc = -0.15, 0.
r = 0.5
sigma = Constant(0.1)

# Mesh/particle properties, use safe number of particles
resolution_level = 4
nx = pow(2, resolution_level)
pres = 160 * pow(2, resolution_level)

# Polynomial order
k = 2
lm = 0
kbar = k

# Set advective transport to 0
Uh = 0.

# Timestepping info
Tend = 2.
dt = Constant(0.08/(pow(2, resolution_level)))

if comm.Get_rank() == 0:
    print("Starting computation with grid resolution "+str(nx))

# Compute num steps till completion
num_steps = np.rint(Tend/float(dt))

# Generate mesh
mesh = Mesh('./../../meshes/circle_0.xml')
n = nx
while (n > 1):
    mesh = refine(mesh)
    n /= 2

# Velocity and initial condition
V = VectorFunctionSpace(mesh, 'DG', 3)
uh = Function(V)
uh.assign(Expression(('-Uh*x[1]', 'Uh*x[0]'), Uh=Uh, degree=3))

psi0_expression = GaussianPulse(center=(xc, yc), sigma=float(sigma),
                                U=[Uh, Uh], time=0., height=1., degree=3)

# Generate particles
if comm.Get_rank() == 0:
    x = RandomCircle(Point(x0, y0), r).generate([pres, pres])
    s = np.zeros((len(x), 1), dtype=np.float_)
else:
    x = None
    s = None

x = comm.bcast(x, root=0)
s = comm.bcast(s, root=0)

# Initialize particles with position x and scalar property s at the mesh
p = particles(x, [s], mesh)
property_idx = 1  # Scalar quantity is stored at slot 1

# Define the variational (projection problem)
W_e = FiniteElement("DG", mesh.ufl_cell(), k)
T_e = FiniteElement("DG", mesh.ufl_cell(), lm)
Wbar_e = FiniteElement("DGT", mesh.ufl_cell(), k)

W = FunctionSpace(mesh, W_e)
T = FunctionSpace(mesh, T_e)
Wbar = FunctionSpace(mesh, Wbar_e)

psi_h, psi0_h = Function(W), Function(W)
lambda_h = Function(T)
psibar_h = Function(Wbar)

# Boundary conditions
bc = DirichletBC(Wbar, Constant(0.), "on_boundary")

# Initialize forms
FuncSpace_adv = {'FuncSpace_local': W, 'FuncSpace_lambda': T, 'FuncSpace_bar': Wbar}
forms_pde = FormsPDEMap(mesh, FuncSpace_adv).forms_theta_linear(psi0_h, uh,
                                                                dt, Constant(1.0))
pde_projection = PDEStaticCondensation(mesh, p,
                                       forms_pde['N_a'], forms_pde['G_a'], forms_pde['L_a'],
                                       forms_pde['H_a'],
                                       forms_pde['B_a'],
                                       forms_pde['Q_a'], forms_pde['R_a'], forms_pde['S_a'],
                                       [bc], property_idx)

# Set initial condition at mesh and particles
psi0_h.interpolate(psi0_expression)
p.interpolate(psi0_h, property_idx)

# Just assemble once
pde_projection.assemble(True, True)

step = 0
t = 0.
timer = Timer()

# Just repeatedly solve same system
timer.start()
while step < num_steps:
    step += 1
    t += float(dt)

    if comm.rank == 0:
        print("Step  "+str(step))
    
    t1 = Timer("[P] Solve PDE constrained projection")
    pde_projection.solve_problem(psibar_h, psi_h, 'mumps', 'default')
    del(t1)
    
timer.stop()

# Compute error (we should remain at initial condition)
psi0_expression.t = step * float(dt)
l2_error = sqrt(assemble(dot(psi_h - psi0_expression, psi_h - psi0_expression)*dx))

if comm.Get_rank() == 0:
    print("l2 error "+str(l2_error))

list_timings(TimingClear.keep, [TimingType.wall])
