from dolfin import *
from mpi4py import MPI as pyMPI
import numpy as np
import os
from mshr import *
import matplotlib.pyplot as plt 

# Load from package
from DolfinParticles import (particles, advect_rk3, advect_rk2, advect_particles, l2projection,
                     PDEStaticCondensation, RandomBox, RegularBox,
                     FormsPDEMap, GaussianPulse, AddDelete)

# 
comm = pyMPI.COMM_WORLD

def assign_particle_values(x, u_exact):
    if comm.Get_rank() == 0:
        s=np.asarray([u_exact(x[i,:]) for i in range(len(x))], dtype = np.float_)
    else:
        s = None
    return s

class Ball(Expression):
    def __init__(self,radius, center, **kwargs):
        assert len(center)==3
        self.r = radius
        self.center = center
        if 'lb' in kwargs:
            self.lb = kwargs['lb']
        else:
            self.lb = 0.
        
        if 'ub' in kwargs: 
            self.ub = kwargs['ub']
        else:
            self.ub = 1.
    
    def eval(self, value, x):
        (xc, yc, zc) = self.center
        
        if (x[0] - xc)**2 + (x[1] - yc)**2 + (x[2] - zc)**2 <= self.r**2:
            value[0] = self.ub
        else:
            value[0] = self.lb

    def value_shape(self):
        return ()

def decorate_projection_test(my_projection_test):
    def wrapper(polynomial_order, interpolate_expression, **kwargs):
        xmin, ymin, zmin = 0., 0., 0.
        xmax, ymax, zmax = 1., 1., 1.
        nx = 25

        property_idx = 1

        mesh = BoxMesh(Point(xmin,ymin,zmin), Point(xmax,ymax,zmax), nx,nx,nx)

        if len(interpolate_expression.ufl_shape) == 0:
            V = FunctionSpace(mesh,"DG", polynomial_order)
        elif len(interpolate_expression.ufl_shape) == 1:
            V = VectorFunctionSpace(mesh,"DG", polynomial_order)

        v_exact = Function(V)
        v_exact.assign(interpolate_expression)

        #x = RandomBox(Point(0.,0.,0.), Point(1.,1.,1.)).generate([70,70,70])
        if my_projection_test.__name__ == "l2projection_test":
            x = RandomBox(Point(0.,0.,0.), Point(1.,1.,1.)).generate([4,4,4])
        elif my_projection_test.__name__ == "l2projection_bounded_test":
            x = RegularBox(Point(0.,0.,0.), Point(1.,1.,1.)).generate([125,125,125])
            
        s = assign_particle_values(x, interpolate_expression)

        x = comm.bcast(x, root=0)
        s = comm.bcast(s, root=0)

        # Just make a complicated particle, possibly with scalars and vectors mixed 
        p = particles(x, [s], mesh)
                                                       
        if my_projection_test.__name__ == "l2projection_test":
            # Do AddDelete sweep
            AD = AddDelete(p, 13, 15, [v_exact])
            AD.do_sweep()
            
            vh = my_projection_test(mesh,V,p, property_idx, **kwargs)
            
            error_sq = abs(assemble( dot(v_exact - vh, v_exact - vh)*dx ))
            if comm.Get_rank() == 0:
                if error_sq > 1e-13:
                    raise Exception("Function should be reconstructed exactly")

        if my_projection_test.__name__ == "l2projection_bounded_test":
            # AddDelete swee not yet working for bounded test --> Need fix in [lb, ub] list
            vh = my_projection_test(mesh,V,p, property_idx, **kwargs)           
            if np.any(vh.vector().array() > kwargs["ub"] + 1e-6) or \
                np.any(vh.vector().array() < kwargs["lb"] - 1e-6) :
                raise Exception("Violated bounds in box constrained projection")    
        return
    return wrapper

@decorate_projection_test
def l2projection_test(mesh, V, p, property_idx):
    phih = Function(V)
    lstsq_phi = l2projection(p,V,property_idx)
    lstsq_phi.project(phih)
    
    return phih

@decorate_projection_test
def l2projection_bounded_test(mesh, V, p, property_idx, **kwargs):
    if 'lb' in kwargs:
        lb = kwargs['lb']
    else:
        raise Exception('Lowerbound needs to be provided!')
    
    if 'ub' in kwargs:
        ub = kwargs['ub']
    else:
        raise Exception('Lowerbound needs to be provided!')
    
    phih = Function(V)
    lstsq_rho = l2projection(p,V,property_idx)
    lstsq_rho.project(phih, lb, ub)
    return phih

def main():
    polynomial_order = 2   
    # Scalar expression
    interpolate_expression = Expression("pow(x[0],2) + pow(x[1],2)", degree =3)
    
    # TODO: Vector expression
   
    # Scalar project
    if comm.Get_rank() == 0:
        print ('{:=^72}'.format('Run scalar project'))
    l2projection_test(polynomial_order, interpolate_expression)
    if comm.Get_rank() == 0:
        print ('{:=^72}'.format('Passed scalar project'))
    
    # TODO: vector 
    
    # Discontinuous scalar expression
    lb = -3.; ub = -1. 
    interpolate_discontinuous = Ball(radius = 0.15, center = [0.5, 0.5, 0.5], degree = 3, lb = lb, ub = ub)
    
    # Bounded projection
    if comm.Get_rank() == 0:
        print ('{:=^72}'.format('Run bounded projection'))     
    l2projection_bounded_test(polynomial_order, interpolate_discontinuous, lb = lb, ub = ub)
    if comm.Get_rank() == 0:
        print ('{:=^72}'.format('Passed bounded projection')) 
    
if __name__ == "__main__":
    main()    



