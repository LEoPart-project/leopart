# __author__ = 'Jakob Maljaars <j.m.maljaars@tudelft.nl>'
# __date__   = '2018-08-02'
# __copyright__ = 'Copyright (C) 2011' + __author__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

"""
Unit tests for the least squares and PDE-constrained projection. For further reading
"""

from dolfin import *
from DolfinParticles import particles, l2projection, PDEStaticCondensation, FormsPDEMap, RandomRectangle
import numpy as np
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD

def assign_particle_values(x, u_exact):
    if comm.Get_rank() == 0:
        s=np.asarray([u_exact(x[i,:]) for i in range(len(x))], dtype = np.float_)
    else:
        s = None
    return s

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

def decorate_projection_test(my_projection_test):
    # I want to provide the  function space
    def wrapper(polynomial_order, interpolate_expression, **kwargs):
        xmin = 0.; xmax = 1.
        ymin = 0.; ymax = 1.
        
        property_idx = 5
    
        mesh = RectangleMesh(Point(xmin,ymin),Point(xmax, ymax), 40,40)
        bmesh  = BoundaryMesh(mesh,'exterior')
        
        if len(interpolate_expression.ufl_shape) == 0:
            V = FunctionSpace(mesh,"DG", polynomial_order)
        elif len(interpolate_expression.ufl_shape) == 1:
            V = VectorFunctionSpace(mesh,"DG", polynomial_order)
            
        v_exact = Function(V)
        v_exact.interpolate(interpolate_expression)
        
        x = RandomRectangle(Point(xmin, ymin), Point(xmax,ymax)).generate([500, 500])
        s = assign_particle_values(x,interpolate_expression)
        x = comm.bcast(x, root=0)
        s = comm.bcast(s, root=0)
        
        # Just make a complicated particle, possibly with scalars and vectors mixed 
        p = particles(x, [x,s,x,x,s], mesh)
                
        vh = my_projection_test(mesh,bmesh,V,p, property_idx, **kwargs)
                
        if my_projection_test.__name__ == "l2projection_test":
            error_sq = abs(assemble( dot(v_exact - vh, v_exact - vh)*dx ))
            if comm.Get_rank() == 0:
                if error_sq > 1e-15:
                    raise Exception("Function should be reconstructed exactly")
        
        if my_projection_test.__name__ == "l2projection_bounded_test":
            if np.any(vh.vector().vec().array > kwargs["ub"] + 1e-12) or \
                np.any(vh.vector().vec().array < kwargs["lb"] - 1e-12) :
                raise Exception("Violated bounds in box constrained projection")    
        return
    return wrapper
          
@decorate_projection_test
def l2projection_test(mesh, bmesh, V,p, property_idx):
    phih = Function(V)
    lstsq_rho = l2projection(p,V,property_idx)
    lstsq_rho.project(phih.cpp_object())
    return phih

@decorate_projection_test
def l2projection_bounded_test(mesh, bmesh, V,p, property_idx, **kwargs):
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
    lstsq_rho.project(phih.cpp_object(), lb, ub)
    return phih

def pde_constrained_test(polynomial_order, interpolate_expression):
    xmin = 0.; xmax = 1.
    ymin = 0.; ymax = 1.
    
    property_idx = 1
    dt           = 1.
    k            = polynomial_order
    
    # Make mesh
    mesh  = RectangleMesh(Point(xmin,ymin),Point(xmax, ymax), 40,40)
    bmesh = BoundaryMesh(mesh,'exterior')
        
    # Make function spaces and functions
    W_e    = FiniteElement("DG", mesh.ufl_cell(), k)
    T_e    = FiniteElement("DG", mesh.ufl_cell(), 0)
    Wbar_e = FiniteElement("DGT", mesh.ufl_cell(), k)
    
    W      = FunctionSpace(mesh, W_e) 
    T      = FunctionSpace(mesh, T_e) 
    Wbar   = FunctionSpace(mesh, Wbar_e) 
    
    psi_h    = Function(W); psi0_h = Function(W)
    lambda_h = Function(T)
    psibar_h = Function(Wbar)
    
    uadvect = Constant((0,0))
    
    # Define particles
    x = RandomRectangle(Point(xmin, ymin), Point(xmax,ymax)).generate([500, 500])
    s = assign_particle_values(x,interpolate_expression)
    x = comm.bcast(x, root=0)
    s = comm.bcast(s, root=0)
    psi0_h.assign(interpolate_expression)
    
    # Just make a complicated particle, possibly with scalars and vectors mixed 
    p = particles(x, [s], mesh)
    p.interpolate(psi0_h.cpp_object(), 1)
    
    # Initialize forms
    FuncSpace_adv = {'FuncSpace_local': W, 'FuncSpace_lambda': T, 'FuncSpace_bar': Wbar}
    forms_pde     = FormsPDEMap(mesh, FuncSpace_adv).forms_theta_linear(psi0_h, uadvect, dt, Constant(1.0) )
    pde_projection = PDEStaticCondensation(mesh,p,  forms_pde['N_a'], forms_pde['G_a'], forms_pde['L_a'],
                                                                                        forms_pde['H_a'], 
                                                                                        forms_pde['B_a'],
                                                    forms_pde['Q_a'], forms_pde['R_a'], forms_pde['S_a'],
                                                                                                  [],property_idx)
    # Assemble and solve
    pde_projection.assemble(True, True)
    pde_projection.solve_problem(psibar_h.cpp_object(), psi_h.cpp_object(), lambda_h.cpp_object(),
                                 'none', 'default')
    
    error_psih   = abs(assemble( (psi_h - psi0_h) * (psi_h - psi0_h) *dx ) )   
    error_lamb = abs(assemble( lambda_h * lambda_h *dx ) )
    
    # psi_h should be reconstructed exactly, Lagrange multiplier field should be zero
    if error_psih > 1e-15 or error_lamb > 1e-15:
        raise Exception('Failed in PDE constrained test')
    return
    
def main():
    polynomial_order = 2
    # Scalar expression
    interpolate_expression = Expression("pow(x[0],2)", degree = 3)
    # Vector expression
    interpolate_expression_vec = Expression(("pow(x[0],1)", "pow(x[1],1)"), degree = 3)
    # Discontinuous scalar expression
    lb = -3.; ub = -1. 
    interpolate_discontinuous = SlottedDisk(radius = 0.15, center = [0.5, 0.5], width = 0.05, depth = 0., degree = 3, lb = lb, ub = ub)
    
    # Scalar project
    if comm.Get_rank() == 0:
        print ('{:=^72}'.format('Run scalar project'))
    l2projection_test(2, interpolate_expression)
    if comm.Get_rank() == 0:
        print ('{:=^72}'.format('Passed scalar project'))
    
    # Vector project
    if comm.Get_rank() == 0:
        print ('{:=^72}'.format('Run vector projection'))    
    
    l2projection_test(2, interpolate_expression_vec)
    
    if comm.Get_rank() == 0:
        print ('{:=^72}'.format('Passed vector projection'))
    
    # Bounded projection
    if comm.Get_rank() == 0:
        print ('{:=^72}'.format('Run bounded projection'))    
    
    l2projection_bounded_test(3, interpolate_discontinuous, lb = lb, ub = ub)
    
    if comm.Get_rank() == 0:
        print ('{:=^72}'.format('Passed bounded projection'))  
    
    # PDE constrained test
    if comm.Get_rank() == 0:
        print ('{:=^72}'.format('Run PDE constrained projection')) 
        
    pde_constrained_test(polynomial_order, interpolate_expression)
    
    if comm.Get_rank() == 0:
        print ('{:=^72}'.format('Passed PDE constrained projection')) 
        
if __name__ == "__main__":
    main()        