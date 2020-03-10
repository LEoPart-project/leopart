from dolfin import (BoxMesh, Point, Timer, list_timings, TimingClear, TimingType, MPI)
from leopart import particles, RandomBox, RandomCell
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD

# Geometry and mesh resolution
xmin, xmax = -1, 1
ymin, ymax = -1, 1
zmin, zmax = -1, 1

(nx, ny, nz) = (64, 64, 64)
mesh = BoxMesh(MPI.comm_world, Point(xmin, ymin, zmin), Point(xmax, ymax, zmax), nx, ny, nz)

# Initialize particles with RandomBox
pres = 384
t1 = Timer("[B] create particles with RandomBox")
x = RandomBox(Point(xmin, ymin, zmin), Point(xmax, ymax, zmax)).generate([pres] * 3)
del(t1)
t1 = Timer("[B] instantiate particles from RandomBox")
p = particles(x, [], mesh)
del(t1)
p_num_rbox = p.number_of_particles()
del(p, x)

# Initialize particles with RandomCell
ppc = 36
t1 = Timer("[C] create particles with RandomCell")
x = RandomCell(mesh).generate(ppc)
del(t1)
t1 = Timer("[C] instantiate particles from RandomCell ")
p = particles(x, [], mesh)
del(t1)

p_num_rcell = p.number_of_particles()

del(p)
del(x)


list_timings(TimingClear.keep, [TimingType.wall])

if comm.rank == 0:
    print("Particles in RandomBox " + str(p_num_rbox))
    print("Particles in RandomCell " + str(p_num_rcell))
