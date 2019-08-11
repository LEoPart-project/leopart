import numpy as np
from dolfin import *
from leopart import *
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD

# Buoyant layer thickness
db = 0.2
# Aspect ratio
lmbda = Constant(0.9142)
xmin, xmax = 0.0, float(lmbda)
ymin, ymax = 0.0, 1.0

class StepFunction(UserExpression):

    def eval_cell(self, values, x, cell):
        c = Cell(mesh, cell.index)
        if c.midpoint()[1] > db + 0.02*np.cos(np.pi*x[0]/float(lmbda)):
            values[0] = 0.0
        else:
            values[0] = 1.0


lims = np.array([[xmin, xmin, ymin, ymax], [xmax, xmax, ymin, ymax],
                 [xmin, xmax, ymin, ymin], [xmin, xmax, ymax, ymax]])
lim_dict = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}

mesh = RectangleMesh.create(
    comm, [Point(0.0, 0.0), Point(float(lmbda), 1.0)],
    [80, 80], CellType.Type.triangle, "left/right")

# Shift the mesh to line up with the initial step function condition
# scale = db * (1.0 - db)
# shift = Expression(("0.0", "x[1]*(H - x[1])/S*A*cos(pi/L*x[0])"),
#                    A=0.02, L=lmbda, H=1.0, S=scale, degree=4)
#
# V = VectorFunctionSpace(mesh, "CG", 1)
# displacement = interpolate(shift, V)
# ALE.move(mesh, displacement)

# Entrainment functional measures
de = 1
cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
CompiledSubDomain("x[1] > db - DOLFIN_EPS", db=db).mark(cf, de)
dx = Measure("dx", subdomain_data=cf)

# Setup particles
pres = 300
x = RegularRectangle(Point(xmin, ymin), Point(xmax, ymax)).generate([pres, pres])
s = np.zeros((len(x), 1), dtype=np.float_)

# Interpolate initial function onto particles, index slot 1
property_idx = 1
ptcls = particles(x, [s], mesh)

# Define the variational (projection problem)
k = 1
W_e = FiniteElement("DG", mesh.ufl_cell(), k)
T_e = FiniteElement("DG", mesh.ufl_cell(), 0)
Wbar_e = FiniteElement("DGT", mesh.ufl_cell(), k)

# Composition field space
Wh = FunctionSpace(mesh, W_e)
Th = FunctionSpace(mesh, T_e)
Wbarh = FunctionSpace(mesh, Wbar_e)

gamma = interpolate(StepFunction(), Wh)
gamma0 = interpolate(StepFunction(), Wh)

ptcls.interpolate(gamma, property_idx)

lambda_h = Function(Th)
psibar_h = Function(Wbarh)


# Elements for Stokes
W_e_2 = VectorElement("DG", mesh.ufl_cell(), k)
T_e_2 = VectorElement("DG", mesh.ufl_cell(), 0)
Wbar_e_2 = VectorElement("DGT", mesh.ufl_cell(), k)
Wbar_e_2_H12 = VectorElement("CG", mesh.ufl_cell(), k)["facet"]

Q_E = FiniteElement("DG", mesh.ufl_cell(), k-1)
Qbar_E = FiniteElement("DGT", mesh.ufl_cell(), k)

W_2 = FunctionSpace(mesh, W_e_2)
u_vec = Function(W_2)

t = Constant(0.0)
dt = Constant(1e-2)
# Initialise advection forms
FuncSpace_adv = {'FuncSpace_local': Wh, 'FuncSpace_lambda': Th, 'FuncSpace_bar': Wbarh}
forms_pde = FormsPDEMap(mesh, FuncSpace_adv).forms_theta_linear(gamma0, u_vec,
                                                                dt, Constant(1.0), zeta=Constant(25.0))
pde_projection = PDEStaticCondensation(mesh, ptcls,
                                       forms_pde['N_a'], forms_pde['G_a'], forms_pde['L_a'],
                                       forms_pde['H_a'],
                                       forms_pde['B_a'],
                                       forms_pde['Q_a'], forms_pde['R_a'], forms_pde['S_a'],
                                       [DirichletBC(Wbarh, Constant(1.0), "near(x[1], 0.0)")], property_idx)

# Function spaces for Stokes
mixedL = FunctionSpace(mesh, MixedElement([W_e_2, Q_E]))
mixedG = FunctionSpace(mesh, MixedElement([Wbar_e_2_H12, Qbar_E]))

U0, Uh = Function(mixedL), Function(mixedL)
Uhbar = Function(mixedG)

# BCs
bcs = [DirichletBC(mixedG.sub(0), Constant((0, 0)), "near(x[1], 0.0) or near(x[1], 1.0)"),
       DirichletBC(mixedG.sub(0).sub(0), Constant(0), CompiledSubDomain("near(x[0], 0.0) or near(x[0], lmbda)", lmbda=lmbda)),]

# Forms Stokes
alpha = Constant(6*k*k)
beta = Constant(0.)
Rb = Constant(1.0)
eta = Constant(1.0)
forms_stokes = FormsStokes(mesh, mixedL, mixedG, alpha).forms_steady(eta, Rb*gamma*Constant((0, 1)))

ssc = StokesStaticCondensation(mesh,
                               forms_stokes['A_S'], forms_stokes['G_S'],
                               forms_stokes['B_S'],
                               forms_stokes['Q_S'], forms_stokes['S_S'])

# Particle advector
C_CFL = 0.2
hmin = MPI.min(comm, mesh.hmin())
ap = advect_rk3(ptcls, u_vec.function_space(), u_vec, "closed")

def output_functionals(fname, vals, append=True):
    if comm.rank == 0:
        with open(fname, "a" if append else "w") as fi:
            fi.write(",".join(map(lambda v: "%.6e" % v, vals)) + "\n")

points_list = list(Point(*pp) for pp in ptcls.positions())
particles_values = ptcls.get_property(property_idx)
XDMFFile("./pts/step%.4d.xdmf" % 0).write(points_list, particles_values)
# Time loop
XDMFFile("gamma.xdmf").write_checkpoint(gamma, "gamma", float(t), append=False)
conservation0 = assemble(gamma * dx)

velocity_assigner = FunctionAssigner(u_vec.function_space(), mixedL.sub(0))
gamma_assigner = FunctionAssigner(gamma0.function_space(), gamma.function_space())

for j in range(5000):
    info("step %d, dt %.3e" % (j, float(dt)))
    t.assign(float(t) + float(dt))
    if float(t) > 2010.0:
        break

    # Solve Stokes
    time = Timer("ZZZ Stokes assemble")
    ssc.assemble_global_system(True)
    del time
    time = Timer("ZZZ Stokes solve")
    for bc in bcs:
        ssc.apply_boundary(bc)
    ssc.solve_problem(Uhbar.cpp_object(), Uh.cpp_object(), "mumps", "default")
    del time

    velocity_assigner.assign(u_vec, Uh.sub(0))
    max_u_vec = u_vec.vector().norm("linf")
    dt.assign(C_CFL * hmin / max_u_vec)

    urms = (1.0/lmbda *assemble(dot(u_vec, u_vec)*dx))**0.5
    conservation = assemble(gamma * dx) / conservation0
    entrainment = assemble(1.0/(lmbda*Constant(db)) * gamma * dx(de))
    output_functionals("data.dat", [float(t), float(dt), urms, conservation, entrainment],
                       append=j!=0)

    time = Timer("ZZZ Do_step")
    ap.do_step(float(dt))
    del time

    time = Timer("ZZZ PDE project assemble")
    pde_projection.assemble(True, True)
    del time

    time = Timer("ZZZ PDE project solve")
    pde_projection.solve_problem(psibar_h.cpp_object(), gamma.cpp_object(),
                                 lambda_h.cpp_object(), 'mumps', 'default')
    del time

    gamma_assigner.assign(gamma0, gamma)

    points_list = list(Point(*pp) for pp in ptcls.positions())
    particles_values = ptcls.get_property(property_idx)
    info("num particles: %d" % MPI.sum(comm, len(points_list)))
    XDMFFile("./pts/step%.4d.xdmf" % (j+1)).write(points_list, particles_values)
    XDMFFile("gamma.xdmf").write_checkpoint(gamma, "gamma", float(t), append=True)
    quit()

list_timings(TimingClear.clear, [TimingType.wall])