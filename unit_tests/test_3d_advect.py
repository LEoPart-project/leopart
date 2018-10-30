from dolfin import *
from mpi4py import MPI as pyMPI
import numpy as np
import os
from mshr import *
import matplotlib.pyplot as plt 

# Load from package
from DolfinParticles import (particles, advect_rk3, advect_rk2, advect_particles,
                     RandomBox, RegularBox)

# 
comm = pyMPI.COMM_WORLD

def decorate_periodic_tests(my_func):
    def wrapper():
        xmin, ymin, zmin = 0., 0., 0.
        xmax, ymax, zmax = 1., 1., 1.
        
        mesh = UnitCubeMesh(10, 10, 10)
        bmesh  = BoundaryMesh(mesh,'exterior')

        lims = np.array([[xmin, xmin, ymin, ymax, zmin, zmax],[xmax, xmax, ymin, ymax, zmin, zmax],
                        [xmin, xmax, ymin, ymin, zmin, zmax],[xmin, xmax, ymax, ymax, zmin, zmax],
                        [xmin, xmax, ymin, ymax, zmin, zmin],[xmin, xmax, ymin, ymax, zmax, zmax]
                        ])
        
        vexpr = Constant((1.,1.,1.))
        V = VectorFunctionSpace(mesh,"CG", 1)
        
        x = RandomBox(Point(0.,0.,0.), Point(1.,1.,1.)).generate([10,10,10])
        x = comm.bcast(x, root=0)
        dt= 0.05
        
        xp0, xpE = my_func(mesh,bmesh,lims,V, vexpr,x,dt)
        
        xp0_root = comm.gather( xp0, root = 0)
        xpE_root = comm.gather( xpE, root = 0)
        
        assert len(xp0) == len(xpE)
        
        if comm.Get_rank() == 0: 
            
            
            xp0_root = np.float32( np.vstack(xp0_root) )
            xpE_root = np.float32( np.vstack(xpE_root) )
                        
            # Sort on x positions
            xp0_root = xp0_root[xp0_root[:,0].argsort(),:]
            xpE_root = xpE_root[xpE_root[:,0].argsort(),:]
            
            error = np.linalg.norm(xp0_root - xpE_root)
            if error > 1e-10:
                raise Exception("Error too high in function "+my_func.__name__)        
        return    
    return wrapper 

@decorate_periodic_tests
def advect_particle_periodic(mesh,bmesh,lims,V, vexpr,x,dt):
    v = Function(V)
    v.assign(vexpr)
    
    p = particles(x, [x*0, x**2], mesh)
    ap= advect_particles(p, V, v, bmesh, 'periodic', lims.flatten())
    
    xp_0 = p.positions(mesh)
    t  = 0.
    while t<1.-1e-12:
        ap.do_step(dt)
        t += dt
    xp_end = p.positions(mesh)
    return xp_0, xp_end

@decorate_periodic_tests
def advect_particle_periodic_rk2(mesh,bmesh,lims,V, vexpr,x,dt):
    v = Function(V)
    v.assign(vexpr)
    
    p = particles(x, [x*0, x**2], mesh)
    ap= advect_rk2(p, V, v, bmesh, 'periodic', lims.flatten())
    
    xp_0 = p.positions(mesh)
    t  = 0.
    while t<1.-1e-12:
        ap.do_step(dt)
        t += dt
    xp_end = p.positions(mesh)
    return xp_0, xp_end

@decorate_periodic_tests
def advect_particle_periodic_rk3(mesh,bmesh,lims,V, vexpr,x,dt):
    v = Function(V)
    v.assign(vexpr)
    
    p = particles(x, [x[:,0]*0, x**2], mesh)
    ap= advect_rk2(p, V, v, bmesh, 'periodic', lims.flatten())
    
    xp_0 = p.positions(mesh)
    t  = 0.
    while t<1.-1e-12:
        ap.do_step(dt)
        t += dt
    xp_end = p.positions(mesh)
    return xp_0, xp_end

def main():
    # TODO: one particle tests (convergence)

    # Periodic tests
    if comm.Get_rank() == 0:
        print ('{:=^72}'.format('Run periodic particle tests'))
    advect_particle_periodic()
    advect_particle_periodic_rk2()
    advect_particle_periodic_rk3()
    if comm.Get_rank() == 0:
        print ('{:=^72}'.format('Passed periodic particle tests'))

if __name__ == "__main__":
    main()