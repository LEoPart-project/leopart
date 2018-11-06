# __author__ = 'Jakob Maljaars <j.m.maljaars@tudelft.nl>'
# __date__   = '2018-08'
# __copyright__ = 'Copyright (C) 2011' + __author__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

"""
    Testing the advection of a slotted disk (with sharp discontinuities)
    , using a bounded l2 projection.
    Since no global solve is involved, this
    test is well-suited for assessing the scaling in parallel.
    Note: conservation properties are lost with this approach.
"""

from dolfin import *
from mpi4py import MPI as pyMPI
import numpy as np

# Load from package
from DolfinParticles import (particles, advect_rk3,
                        l2projection, RandomCircle)

comm = pyMPI.COMM_WORLD

# TODO: consider placing in InitialConditions
class SlottedDisk(UserExpression):
    def __init__(self,radius, center, width, depth, lb = 0., ub = 1., **kwargs):
        self.r      = radius
        self.width  = width
        self.depth  = depth
        self.center = center
        self.lb     = lb
        self.ub     = ub
        super().__init__(self, **kwargs)

    def eval(self, value, x):
        xc = self.center[0]
        yc = self.center[1]

        if  ((x[0] - xc)**2 + (x[1] - yc)**2 <=self.r**2) \
            and not ( (xc - self.width) <= x[0] <=  (xc + self.width)  and  x[1] >= yc + self.depth):
            value[0] = self.ub
        else:
            value[0] = self.lb

    def value_shape(self):
        return ()

def assign_particle_values(x, u_exact):
    if comm.Get_rank() == 0:
        s=np.asarray([u_exact(x[i,:]) for i in range(len(x))], dtype = np.float_).reshape(len(x), 1)
    else:
        s = None
    return s

# Domain properties
x0,y0   = 0., 0.        # Center of domain
xc,yc   = -0.15, 0.     # Center of Gaussian
r       = .5            # Radius of domain
rdisk   = 0.2           # Radius of slotted disk
rwidth  = 0.05          # Width of slot
lb      = -1.           # Lower value in slotted disk
ub      = 3.            # Upper value in slotted disk

# Mesh/particle resolution
nx  = 64
pres= 800

# Polynomial order for bounded l2 map
k   = 1

# Magnitude solid body rotation .
Uh = np.pi

# Timestepping
Tend = 2.
dt = Constant(0.02)
num_steps = np.rint(Tend/float(dt))

# Output directory
store_step = 1
outdir = './../results/SlottedDisk_Rotation/'
outfile= File(outdir+'psi_h.pvd')

# Mesh
mesh = Mesh('circle.xml')
mesh = refine(mesh)
mesh = refine(mesh)
mesh = refine(mesh)
bmesh  = BoundaryMesh(mesh,'exterior')

# Set slotted disk
psi0_expr = SlottedDisk(radius = rdisk, center = [xc, yc], width = rwidth, depth = 0.,
                                        degree = 3, lb = lb, ub = ub)

# Function space and velocity field
W = FunctionSpace(mesh, 'DG', k)
psi_h = Function(W)

V   = VectorFunctionSpace(mesh,'DG', 3)
uh  = Function(V)
uh.assign( Expression( ('-Uh*x[1]','Uh*x[0]'),Uh = Uh, degree=3) )

# Generate particles
if comm.Get_rank() == 0:
    x    =  RandomCircle(Point(x0, y0), r).generate([pres, pres])
    s    =  assign_particle_values(x, psi0_expr)
else:
    x = None
    s = None

x = comm.bcast(x, root=0)
s = comm.bcast(s, root=0)

p = particles(x, [s], mesh)
# Initialize advection class, use RK3 scheme
ap  = advect_rk3(p, V, uh, bmesh, 'closed', 'none')
# Init projection
lstsq_psi = l2projection(p,W,1)

# Do projection to get initial field
lstsq_psi.project(psi_h.cpp_object(), lb, ub)
outfile << psi_h

step = 0
area_0   = assemble(psi_h*dx)
timer    = Timer()

timer.start()
while step < num_steps:
    step += 1

    if comm.Get_rank() == 0:
        print("Step "+str(step))

    ap.do_step(float(dt))
    lstsq_psi.project(psi_h.cpp_object(),lb, ub)

    if step % store_step is 0:
        outfile << psi_h

timer.stop()

area_end = assemble(psi_h*dx)
if comm.Get_rank() == 0:
    print('Num cells '+str(mesh.num_entities_global(2)))
    print('Num particles '+str(len(x)))
    print('Elapsed time '+str(timer.elapsed()[0]) )
    print('Area error '+str(abs(area_end-area_0)) )
