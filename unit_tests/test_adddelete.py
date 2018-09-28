from dolfin import *
from DolfinParticles import particles, AddDelete, RandomRectangle, l2projection
import matplotlib.pyplot as plt
from mpi4py import MPI as pyMPI
import numpy as np

comm = pyMPI.COMM_WORLD

# TODO: clean-up

def assign_particle_values(x, u_exact):
    if comm.Get_rank() == 0:
        s=np.asarray([u_exact(x[i,:]) for i in range(len(x))], dtype = np.float_)
    else:
        s = None
    return s

interpolate_expression = Expression('x[0]', degree = 1)
#interpolate_expression2 = Expression('x[0]**2', degree = 1)


mesh = UnitSquareMesh(5,5)
V    = FunctionSpace(mesh,"DG", 1)
v, vrec = Function(V), Function(V)
v.assign(interpolate_expression)

np_min = 3
np_max = 5

# Initialize particles
x = RandomRectangle(Point(0.0, 0.0), Point(1.,1.)).generate([15, 15])
s = assign_particle_values(x, interpolate_expression)

p = particles(x, [s], mesh)
AD = AddDelete(p, np_min, np_max, [v])
#AD.do_sweep_failsafe(3)
AD.do_sweep_weighted()

print(p.positions(mesh))
print(p.return_property(mesh,1))



property_idx = 1
lstsq_rho = l2projection(p,V,property_idx)
lstsq_rho.project(v.cpp_object())

outfile = File('adddelete_projected.pvd')
outfile << v

#v.assign(interpolate_expression2)

#V    = VectorFunctionSpace(mesh, 'CG',1)
#v    = Function(V)
#vlist = [v,v]

#x = RandomRectangle(Point(0.0, 0.0), Point(1.,1.)).generate([12, 12])

#p = particles(x, [x,x], mesh)
#xp_old = p.positions(mesh)

##print(vlist)
#AD = AddDelete(p, 6, 15, vlist)
##vlist2 = [v, v]
##AD2 = AddDelete(p, 1,6, vlist2)
#AD.do_sweep()

#xp_new = p.positions(mesh)
#print(len(xp_new))

#plt.scatter(xp_old[:,0], xp_old[:,1], c = 'r', edgecolor = None, rasterized = True)
#plt.scatter(xp_new[:,0], xp_new[:,1], c = 'b', edgecolor = None, rasterized = True)
#plt.savefig('positions.png')


#AD2 = AddDelete(p, 1,6, vlist)


