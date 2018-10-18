from dolfin import *
import numpy as np
import time 
from mpi4py import MPI as pyMPI
from DolfinParticles import StokesStaticCondensation

np.set_printoptions(precision=1, linewidth = 150)
comm = pyMPI.COMM_WORLD
#my_rank = pyMPI.rank(comm)

#parameters["krylov_solver"]["absolute_tolerance"]  = 10E-8
#parameters["krylov_solver"]["relative_tolerance"]  = 10E-8
#parameters["krylov_solver"]["maximum_iterations"]  = 1000
#parameters["lu_solver"]["reuse_factorization"] = True

xmin = 0.; xmax = 1.
ymin = 0.; ymax = 1.
nx = 40 ; ny = 40
k  = 3
nu = Constant(1.E-0) 
dt = Constant(2.5e-2)
num_steps = 40
theta0 = 1.0    # Initial theta value
theta1 = 1.0  # Theta after step_change_theta
theta  = Constant(theta0)
step_change_theta = 1 # After step_change_theta we change from theta0 to theta1

outdir = './UnsteadyStokes_HDG_sc/'
u_out = File(outdir+'u.pvd')
p_out = File(outdir+'p.pvd')
ubar_out = File(outdir+'ubar.pvd')
pbar_out = File(outdir+'pbar.pvd')

# Define boundary
def Gamma(x, on_boundary):  return on_boundary
def Corner(x, on_boundary):
    return near(x[0], 1.0) and near(x[1], 1.0)

# Short-cut function for evaluating sum_{K} \int_{K} (integrand) ds
def facet_integral(integrand):
    return integrand('-')*dS + integrand('+')*dS + integrand*ds

class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and\
                (not ((near(x[0], 0) and near(x[1], 1)) or 
                (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
        elif near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.

# Exact solution
#P7e = VectorElement("Lagrange", "triangle", 8)
#P2e = FiniteElement("Lagrange", "triangle", 3)

mesh = RectangleMesh(Point(xmin,ymin), Point(xmax,ymax), nx,nx)   


# The 'unsteady version' of the benchmark in the 2012 paper by Labeur&Wells
u_exact  = Expression(("sin(t) * x[0]*x[0]*(1.0 - x[0])*(1.0 - x[0])*(2.0*x[1] \
                      - 6.0*x[1]*x[1] + 4.0*x[1]*x[1]*x[1])", \
                      "-sin(t)* x[1]*x[1]*(1.0 - x[1])*(1.0 - x[1])*(2.0*x[0] \
                      - 6.0*x[0]*x[0] + 4.0*x[0]*x[0]*x[0])"), t =0, degree = 7, domain = mesh) #element = P7e)
p_exact  = Expression("sin(t) * x[0]*(1.0 - x[0])", t = 0, degree = 7, domain = mesh ) #element = P2e)
du_exact = Expression(("cos(t) * x[0]*x[0]*(1.0 - x[0])*(1.0 - x[0])*(2.0*x[1] \
                      - 6.0*x[1]*x[1] + 4.0*x[1]*x[1]*x[1])", \
                      "-cos(t)* x[1]*x[1]*(1.0 - x[1])*(1.0 - x[1])*(2.0*x[0] \
                      - 6.0*x[0]*x[0] + 4.0*x[0]*x[0]*x[0])"), t =0, degree = 7, domain = mesh ) #element = P7e)

ux_exact = Expression((" x[0]*x[0]*(1.0 - x[0])*(1.0 - x[0])*(2.0*x[1] \
                      - 6.0*x[1]*x[1] + 4.0*x[1]*x[1]*x[1])", \
                      "-x[1]*x[1]*(1.0 - x[1])*(1.0 - x[1])*(2.0*x[0] \
                      - 6.0*x[0]*x[0] + 4.0*x[0]*x[0]*x[0])"), degree = 7, domain = mesh) #element = P7e)

px_exact = Expression("x[0]*(1.0 - x[0])", degree = 7, domain = mesh ) #element = P2e)

sin_ext = Expression("sin(t)", t=0, degree = 7, domain = mesh ) #element = P2e)

f  = du_exact + sin_ext * div( px_exact*Identity(2) - 2*sym(grad(ux_exact)))

Vhigh = VectorFunctionSpace(mesh,"DG",7)
Phigh = FunctionSpace(mesh,"DG",7)

# New syntax:
V   = VectorElement("DG", mesh.ufl_cell(), k)
Q   = FiniteElement("DG", mesh.ufl_cell(), k-1)
#Vbar= VectorElement("CG", mesh.ufl_cell(), k)['facet']
#Qbar= FiniteElement("CG", mesh.ufl_cell(), k)['facet'] 
#Vbar= VectorElement("CG", mesh.ufl_cell(), k)['facet']
Vbar= VectorElement("DGT", mesh.ufl_cell(), k)
Qbar= FiniteElement("DGT", mesh.ufl_cell(),k)

mixedL = FunctionSpace(mesh, MixedElement([V,Q]))
mixedG = FunctionSpace(mesh, MixedElement([Vbar,Qbar]))

v, q = TestFunctions(mixedL)
u, p = TrialFunctions(mixedL)

vbar, qbar = TestFunctions(mixedG)
ubar, pbar = TrialFunctions(mixedG)

Uh    = Function(mixedL)
Uhbar = Function(mixedG)
U0    = Function(mixedL)
Uhbar0= Function(mixedG)
u0, p0  = split(U0)
ubar0, pbar0= split(Uhbar0)

# Penalty
alpha = Constant(6*k*k)
beta  = Constant(0.)

# Mesh related
n  = FacetNormal(mesh)
he = CellSize(mesh)

# Just to test scaling
#p       = Constant(1000)*pt
#pbar    = Constant(1000)*pbart


pI = p*Identity(V.cell().topological_dimension())
pbI= pbar*Identity(V.cell().topological_dimension())

p0I = p0*Identity(V.cell().topological_dimension())
pb0I= pbar0*Identity(V.cell().topological_dimension())

#
# Get it in block form: 
#
# | A   G | 
# |       | V = R  
# | G^T B | 

# Theta scheme
AB = dot(u,v)/dt * dx \
     + theta * inner( 2*nu*sym(grad(u)),grad(v) )*dx \
     + theta * facet_integral( dot(-2*nu*sym(grad(u))*n + (2*nu*alpha/he)*u,v) ) \
     + theta*facet_integral( dot(-2*nu*u,sym(grad(v))*n) ) \
     - theta * inner(pI,grad(v))*dx
BtF= -dot(q,div(u))*dx - facet_integral(beta*he/(nu+1)*dot(p,q))
A  = AB + BtF
A  = Form(A)

# Upper right block G
CD= theta   * facet_integral(-alpha/he*2*nu*inner( ubar,v ) ) \
    + theta * facet_integral( 2*nu*inner(ubar, sym(grad(v))*n) ) \
    + theta * facet_integral(dot(pbI*n,v))   
H = facet_integral(beta*he/(nu+1)*dot(pbar,q))

G = CD + H
G = Form(G)

CDT = theta * facet_integral(- alpha/he*2*nu*inner( vbar, u ) ) \
      + theta * facet_integral( 2*nu*inner(vbar, sym(grad(u))*n) ) \
      + facet_integral( qbar * dot(u,n)) 
HT   = theta * facet_integral(beta*he/(nu+1)*dot(p,qbar))

GT   = CDT + HT
GT   = Form(GT)

# Lower right block B
KL = theta * facet_integral( alpha/he * 2*nu*dot(ubar,vbar)) - theta * facet_integral( dot(pbar*n,vbar) )
#LtP= -facet_integral(dot(ubar,n)*qbar) -facet_integral( beta*he/(nu+1) * pbar * qbar ) 
LtP= -facet_integral(dot(ubar,n)*qbar) - facet_integral( beta*he/(nu+1) * pbar * qbar ) 
B = KL + LtP
B = Form(B)

#Righthandside
Q = dot(f,v)*dx + dot(u0,v)/dt * dx \
    - (1-theta) * inner( 2*nu*sym(grad(u0)),grad(v) )*dx \
    - (1-theta) * facet_integral( dot(-2*nu*sym(grad(u0))*n + (2*nu*alpha/he)*u0,v) ) \
    - (1-theta) * facet_integral( dot(-2*nu*u0,sym(grad(v))*n) ) \
    + (1-theta) * inner(p0I,grad(v))*dx \
    - (1-theta) * facet_integral(-alpha/he*2*nu*inner( ubar0,v ) ) \
    - (1-theta) * facet_integral( 2*nu*inner(ubar0, sym(grad(v))*n) ) \
    - (1-theta) * facet_integral(dot(pb0I*n,v))   
S = facet_integral( dot( Constant((0,0)), vbar) ) #\
    #- (1-theta) * facet_integral(-alpha/he*2*nu*inner( u0,vbar ) ) \
    #+ (1-theta) * facet_integral( dot(pb0I*n,vbar) ) 
    #- (1-theta) * facet_integral( 2*nu*inner(u0, sym(grad(vbar))*n) ) \
    #- (1-theta) * facet_integral( alpha/he * 2*nu*dot(ubar0,vbar)) \
    #+ (1-theta) * facet_integral( dot(pb0I*n,vbar) ) \
    #+ (1-theta) * facet_integral(dot(ubar0,n)*qbar)  \
    #+ (1-theta) * facet_integral( beta*he/(nu+1) * pbar0 * qbar ) 

Q = Form(Q)
S = Form(S)

# Then the boundary conditions
bc0 = DirichletBC(mixedG.sub(0), Constant((0,0)), Gamma)
bc1 = DirichletBC(mixedG.sub(1), Constant(0), Corner, "pointwise")
bcs = [bc0,bc1]

Ael   = [];
invAel= [];
Gel   = [];
Bel   = [];

t = 0.
step = 0

tic = time.time()
solvertime = 0.

reassemble_lhs = True

ssc = StokesStaticCondensation(mesh, A, G, GT, B, Q, S)
#ssc.assemble_global_lhs()
for step in range(num_steps):
    step += 1
    t += float(dt)
    if comm.Get_rank() == 0: print "Step", step, "Time", t
    
    # Set time level in exact solution
    u_exact.t = t 
    p_exact.t = t
    
    # Set time level for body force terms (\theta dependent!)
    #theta = Constant(1.0)
    
    du_exact.t= t - (1-float(theta))*float(dt)
    sin_ext.t = t - (1-float(theta))*float(dt)
    
    #ssc.assemble_global_system(reassemble_lhs)
    #reassemble_lhs = False
    ssc.assemble_global_lhs()
    ssc.assemble_global_rhs()
    for bc in bcs:
        ssc.apply_boundary(bc)
    #[bc.apply(Uhbar.vector()) for bc in bcs ]    

    #if step == 1:
        #ssc.solve_problem(Uhbar, Uh)
    #else:
    
    ssc.solve_problem(Uhbar, Uh)
    
    assign(U0, Uh)
    assign(Uhbar0, Uhbar)
    if step == 1:
        theta.assign(theta1)
    
    udiv_e = sqrt(assemble(div(Uh.sub(0)) * div(Uh.sub(0))*dx))
    
            
toc = time.time()
comm.barrier()
if comm.Get_rank() == 0: print 'Elapsed time for SC procedure ', toc - tic

u_ex_h = interpolate(u_exact, Vhigh)
p_ex_h = interpolate(p_exact, Phigh)

u_error = sqrt(assemble(dot(Uh.sub(0) - u_ex_h, Uh.sub(0) - u_ex_h)*dx) )
p_error = sqrt(assemble(dot(Uh.sub(1) - p_ex_h, Uh.sub(1) - p_ex_h)*dx) )

if comm.Get_rank() == 0:
    print u_error
    print p_error
    print udiv_e
    
u_out = File(outdir+'u.pvd')
p_out = File(outdir+'p.pvd')
u_out   << Uh.split()[0]
p_out   << Uh.split()[1]