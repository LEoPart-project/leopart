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
    # TODO: one particle tests 
    
    
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

#class PeriodicBoundary(SubDomain):
    ## Left boundary is "target domain" G
    #def inside(self, x, on_boundary):
        ## return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        #return bool(( near(x[0], 0) or near(x[1], 0) or near(x[2],0) ) and\
                #(not ((near(x[0], 0) and near(x[1], 1)) or 
                #(near(x[0], 1) and near(x[1], 0)))) and on_boundary)

    #def map(self, x, y):
        #if near(x[0], 1) and near(x[1], 1):
            #y[0] = x[0] - 1.
            #y[1] = x[1] - 1.
        #elif near(x[0], 1):
            #y[0] = x[0] - 1.
            #y[1] = x[1]
        #else:   # near(x[1], 1)
            #y[0] = x[0]
            #y[1] = x[1] - 1.


#def assign_particle_values(x, u_exact):
    #if comm.Get_rank() == 0:
        #s=np.asarray([u_exact(x[i,:]) for i in range(len(x))], dtype = np.float_)
    #else:
        #s = None
    #return s

#class Ball(Expression):
    #def __init__(self,radius, center, **kwargs):
        #assert len(center)==3
        #self.r = radius
        #self.center = center
        #if 'lb' in kwargs:
            #self.lb = kwargs['lb']
        #else:
            #self.lb = 0.
        
        #if 'ub' in kwargs: 
            #self.ub = kwargs['ub']
        #else:
            #self.ub = 1.
    
    #def eval(self, value, x):
        #(xc, yc, zc) = self.center
        
        #if (x[0] - xc)**2 + (x[1] - yc)**2 + (x[2] - zc)**2 <= self.r**2:
            #value[0] = self.ub
        #else:
            #value[0] = self.lb

    #def value_shape(self):
        #return ()

#nx,ny,nz = 5,5,5

#mesh = UnitCubeMesh(nx,ny,nz)
#bmesh= BoundaryMesh(mesh, 'exterior')

#V    = VectorFunctionSpace(mesh, 'DG', 1)
#v    = Function(V)

#vexpr = Constant((1., 1., 1.))
#dt    = Constant(0.1)
#t     = 0.
#v.assign(vexpr)
##facet_domains = CellFunction("size_t" , bmesh)
##facet_domains.set_all(0)
##btop  = Top()
##btop.mark(facet_domains,1)
##tbound = strip_meshfunc(facet_domains, 1)


#x = RandomBox(Point(0.,0.,0.), Point(1.,1.,1.)).generate([10,10,10])
##print x
##x = np.array([[0.65, 0.65, 0.15],
              ##[0.55, 0.45, 0.35],
              ##[0.55, 0.45, 0.35]])
#p = particles(x, [x], mesh)
#xp0 = p.positions(mesh)

## Sort on x positions
#idx = xp0[:,0].argsort()
#xp0 = xp0[idx,:]

##quit()

#xmin, ymin, zmin = 0. , 0., 0.
#xmax, ymax, zmax = 1. , 1., 1.

#lims = np.array([[xmin, xmin, ymin, ymax, zmin, zmax],[xmax, xmax, ymin, ymax, zmin, zmax],
                 #[xmin, xmax, ymin, ymin, zmin, zmax],[xmin, xmax, ymax, ymax, zmin, zmax],
                 #[xmin, xmax, ymin, ymax, zmin, zmin],[xmin, xmax, ymin, ymax, zmax, zmax]
                 #])

#ap= advect_rk3(p, V, v, bmesh, 'periodic', lims.flatten())

#while t<1.-1e-12:
    #ap.do_step(float(dt))
    ##print p.positions(mesh)
    #t += float(dt)


##for step in range(4):
    ##ap.do_step(float(dt))

#xpn = p.positions(mesh)
#idx = xpn[:,0].argsort()
#xpn = xpn[idx,:]
##print xpn
#error = np.linalg.norm(xp0 - xpn)
#print error


#Psi = FunctionSpace(mesh, 'CG', 1)
#psi_h0 = Function(Psi)
#psi_hn = Function(Psi)

#psi_exp = Expression('x[0]', degree = 1);
#psi_h0.assign(psi_exp)

#if comm.Get_rank() == 0:
    #x = RegularBox(Point(0.,0.,0.), Point(1.,1.,1.)).generate([1,1,1])
    #s =  np.zeros(len(x), dtype = np.float_)
#else:
    #x = None
    #s = None
    
#x = comm.bcast(x, root=0)
#s = comm.bcast(s, root=0)

#p   = particles(x, [s], mesh)
#property_idx = 1

##print p.positions(mesh)

#p.interpolate(psi_h0, property_idx)
#AD = AddDelete(p, 4, 5, [psi_h0])
#AD.do_sweep()

##print p.positions(mesh)
#print len(p.positions(mesh))
##quit()
## Then to reconstruct

#outfile_o = File("initial_field.pvd")
#outfile_o << psi_h0

## l2 projection
#lstsq_psi = l2projection(p,Psi,property_idx)
#lstsq_psi.project(psi_hn)

#outfile_n = File("reconstructed_field.pvd")
#outfile_n << psi_hn

#quit()

#def decorate_projection_test(my_projection_test):
    #def wrapper(polynomial_order, interpolate_expression, **kwargs):
        #xmin, ymin, zmin = 0., 0., 0.
        #xmax, ymax, zmax = 1., 1., 1.
        #nx = 50

        #property_idx = 1

        #mesh = BoxMesh(Point(xmin,ymin,zmin), Point(xmax,ymax,zmax), nx,nx,nx)

        #if len(interpolate_expression.ufl_shape) == 0:
            #V = FunctionSpace(mesh,"DG", polynomial_order)
        #elif len(interpolate_expression.ufl_shape) == 1:
            #V = VectorFunctionSpace(mesh,"DG", polynomial_order)

        #v_exact = Function(V)
        #v_exact.assign(interpolate_expression)

        ##x = RandomBox(Point(0.,0.,0.), Point(1.,1.,1.)).generate([70,70,70])
        #x = RegularBox(Point(0.,0.,0.), Point(1.,1.,1.)).generate([1,1,1])
        #s = assign_particle_values(x, interpolate_expression)

        #x = comm.bcast(x, root=0)
        #s = comm.bcast(s, root=0)

        ## Just make a complicated particle, possibly with scalars and vectors mixed 
        #p = particles(x, [s], mesh)
                                
        #AD = AddDelete(p, 6, 10, [v_exact])
        #AD.do_sweep()
                
        #vh = my_projection_test(mesh,V,p, property_idx, **kwargs)
        
        #if my_projection_test.__name__ == "l2projection_test":
            #error_sq = abs(assemble( dot(v_exact - vh, v_exact - vh)*dx ))
            #if comm.Get_rank() == 0:
                #if error_sq > 1e-13:
                    #raise Exception("Function should be reconstructed exactly")

        #if my_projection_test.__name__ == "l2projection_bounded_test":
            #if np.any(vh.vector().array() > kwargs["ub"] + 1e-10) or \
                #np.any(vh.vector().array() < kwargs["lb"] - 1e-10) :
                #raise Exception("Violated bounds in box constrained projection")    
        #return
    #return wrapper

#@decorate_projection_test
#def l2projection_test(mesh, V, p, property_idx):
    #phih = Function(V)
    #lstsq_phi = l2projection(p,V,property_idx)
    #lstsq_phi.project(phih)
    
    #outfile = File("initial_field.pvd")
    #outfile << phih
    #return phih

#@decorate_projection_test
#def l2projection_bounded_test(mesh, V, p, property_idx, **kwargs):
    #if 'lb' in kwargs:
        #lb = kwargs['lb']
    #else:
        #raise Exception('Lowerbound needs to be provided!')
    
    #if 'ub' in kwargs:
        #ub = kwargs['ub']
    #else:
        #raise Exception('Lowerbound needs to be provided!')
    
    #phih = Function(V)
    #lstsq_rho = l2projection(p,V,property_idx)
    #lstsq_rho.project(phih, lb, ub)
    
    #outfile = File("initial_field.pvd")
    #outfile << phih
    #return phih

#def main():
    #polynomial_order = 1   
    ## Scalar expression
    ##interpolate_expression = Expression("pow(x[0],1) + pow(x[1],1)", degree =3)
    #interpolate_expression = Expression("x[0] + x[1]", degree =1)
    ##interpolate_expression = Expression("pow(x[0],2)+pow(x[1],2)", degree =3)
    ## Vector expression
    
    ## Scalar project
    #if comm.Get_rank() == 0:
        #print ('{:=^72}'.format('Run scalar project'))
    #l2projection_test(polynomial_order, interpolate_expression)
    #if comm.Get_rank() == 0:
        #print ('{:=^72}'.format('Passed scalar project'))
    
    ## TODO: vector 
    
    ### Discontinuous scalar expression
    ##lb = -3.; ub = -1. 
    ##interpolate_discontinuous = Ball(radius = 0.15, center = [0.5, 0.5, 0.5], degree = 3, lb = lb, ub = ub)
    
    ### Bounded projection
    ##if comm.Get_rank() == 0:
        ##print ('{:=^72}'.format('Run bounded projection'))    
        
    ##l2projection_bounded_test(polynomial_order, interpolate_discontinuous, lb = lb, ub = ub)
    
    
#if __name__ == "__main__":
    #main()    



