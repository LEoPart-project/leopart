# __author__ = 'Jakob Maljaars <j.m.maljaars@tudelft.nl>'
# __date__   = '2018-08-02'
# __copyright__ = 'Copyright (C) 2011' + __author__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

"""
Unit tests for the least squares and PDE-constrained projection.
"""

from dolfin import *
from DolfinParticles import particles, l2projection, PDEStaticCondensation, FormsPDEMap, RandomRectangle
import numpy as np
from mpi4py import MPI as pyMPI
import pytest

comm = pyMPI.COMM_WORLD

def assign_particle_values(x, u_exact):
    if comm.Get_rank() == 0:
        s=np.asarray([u_exact(x[i,:]) for i in range(len(x))], dtype = np.float_)
    else:
        s = None
    return s

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

@pytest.mark.parametrize('polynomial_order, in_expression', [(2, "pow(x[0], 2)"),                                                                                                                           
                                                             (2, ("pow(x[0], 2)", "pow(x[1], 2)"))])
def test_l2projection(polynomial_order, in_expression):
    # Test l2 projection for scalar and vector valued expression
    interpolate_expression = Expression(in_expression, degree = 3)
    
    xmin = 0.; xmax = 1.
    ymin = 0.; ymax = 1.

    property_idx = 5

    mesh = RectangleMesh(Point(xmin,ymin),Point(xmax, ymax), 40,40)
    bmesh  = BoundaryMesh(mesh, 'exterior')

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
    
    vh = Function(V)                                                                                                                                                                                     
    lstsq_rho = l2projection(p, V, property_idx) 
    lstsq_rho.project(vh.cpp_object())

    error_sq = abs(assemble( dot(v_exact - vh, v_exact - vh)*dx ))
    assert error_sq < 1e-15

@pytest.mark.parametrize('polynomial_order, lb, ub', [(1, -3., -1.),
                                                      (2, -3., -1.)])    
def test_l2projection_bounded(polynomial_order, lb, ub):
    # Test l2 projection if it stays within bounds given by lb and ub
    interpolate_expression = SlottedDisk(radius = 0.15, center = [0.5, 0.5],
                                         width = 0.05, depth = 0., degree = 3, lb =lb, ub = ub)

    xmin = 0.; xmax = 1.
    ymin = 0.; ymax = 1.

    property_idx = 5

    mesh = RectangleMesh(Point(xmin,ymin),Point(xmax, ymax), 40,40)
    bmesh  = BoundaryMesh(mesh, 'exterior')

    V = FunctionSpace(mesh,"DG", polynomial_order)

    x = RandomRectangle(Point(xmin, ymin), Point(xmax,ymax)).generate([500, 500])
    s = assign_particle_values(x,interpolate_expression)
    x = comm.bcast(x, root=0)
    s = comm.bcast(s, root=0)

    # Just make a complicated particle, possibly with scalars and vectors mixed                                                                                                                            
    p = particles(x, [x,s,x,x,s], mesh)

    vh = Function(V)
    lstsq_rho = l2projection(p, V, property_idx)
    lstsq_rho.project(vh.cpp_object(), lb, ub)

    # Assert if it stays within bounds
    assert np.any(vh.vector().get_local() < ub + 1e-12)
    assert np.any(vh.vector().get_local() > lb - 1e-12 )

@pytest.mark.parametrize('polynomial_order, in_expression', [(2, "pow(x[0], 2)"),
                                                             (3, "pow(x[1], 2)")])
def test_pde_constrained(polynomial_order, in_expression):
    interpolate_expression = Expression(in_expression, degree = 3)
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
    
    error_psih = abs(assemble( (psi_h - psi0_h) * (psi_h - psi0_h) *dx ) )   
    error_lamb = abs(assemble( lambda_h * lambda_h *dx ) )

    assert error_psih < 1e-15
    assert error_lamb < 1e-15
