# __author__ = 'Jakob Maljaars <j.m.maljaars@tudelft.nl>'
# __date__   = '2018-08'
# __copyright__ = 'Copyright (C) 2018' + __author__
# __license__  = 'PLEASE DO NOT SHARE WITHOUT AUTHOR CONSENT'

from dolfin import *
import numpy as np
import time as tm
from mpi4py import MPI as pyMPI
from DolfinParticles import StokesStaticCondensation, FormsStokes
import os

comm = pyMPI.COMM_WORLD

def Gamma(x, on_boundary):  return on_boundary
def Corner(x, on_boundary): return x[0] < DOLFIN_EPS and x[1] < DOLFIN_EPS

# Short-cut function for evaluating sum_{K} \int_{K} (integrand) ds
def facet_integral(integrand):
    return integrand('-')*dS + integrand('+')*dS + integrand*ds
    
def exact_solution(domain):
    P7 = VectorElement("Lagrange", "triangle", degree = 8, dim = 2)
    P2 = FiniteElement("Lagrange", "triangle", 3)
    u_exact = Expression((" x[0]*x[0]*(1.0 - x[0])*(1.0 - x[0])*(2.0*x[1] \
                        - 6.0*x[1]*x[1] + 4.0*x[1]*x[1]*x[1])", \
                        "-x[1]*x[1]*(1.0 - x[1])*(1.0 - x[1])*(2.0*x[0] \
                        - 6.0*x[0]*x[0] + 4.0*x[0]*x[0]*x[0])"), element = P7, domain = domain)
    p_exact = Expression("x[0]*(1.0 - x[0])", element = P2, domain = domain)
    return u_exact, p_exact    

def main():
    # Polynomial order and mesh resolution
    k_list  = [1,2,3,4]
    nx_list = [4, 8, 16, 32, 64, 128]
    
    nu = Constant(1)
    
    # Output file for errors
    outdir = './../results/SteadyStokes/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    if comm.rank == 0:
        error_file = open(outdir+"errors.out","w")
        print >> error_file, 5*'%-25s' % ('polynomial order','# cells',
            'velocity error','pressure error','divergence error')
       
    for k in k_list:
        if comm.Get_rank() == 0:
            print('{:=^72}'.format('Computing for polynomial order '+str(k)))
        
        # Error listst
        error_u     = []
        error_p     = []
        error_div   = []
        
        for nx in nx_list:
            if comm.Get_rank() == 0:
                print('# Resolution '+str(nx))
                
            mesh = UnitSquareMesh(nx,nx)
            
            # Get forcing from exact solutions
            u_exact, p_exact = exact_solution(mesh)
            f = div(p_exact*Identity(2) - 2*nu*sym(grad(u_exact)))
            
            # Define FunctionSpaces and functions
            V   = VectorElement("DG", mesh.ufl_cell(), k)
            Q   = FiniteElement("DG", mesh.ufl_cell(), k-1)
            Vbar= VectorElement("DGT", mesh.ufl_cell(), k)
            Qbar= FiniteElement("DGT", mesh.ufl_cell(),k)
            
            mixedL = FunctionSpace(mesh, MixedElement([V,Q]))
            mixedG = FunctionSpace(mesh, MixedElement([Vbar,Qbar]))
            
            Uh    = Function(mixedL)
            Uhbar = Function(mixedG)
    
            # Set forms
            forms_stokes    = FormsStokes(mesh,mixedL,mixedG,k).forms_steady(nu,f)
    
            # No-slip boundary conditions, set pressure in one of the corners
            bc0 = DirichletBC(mixedG.sub(0), Constant((0,0)), Gamma)
            bc1 = DirichletBC(mixedG.sub(1), Constant(0), Corner, "pointwise")
            bcs = [bc0, bc1]
    
            # Initialize static condensation class
            ssc = StokesStaticCondensation(mesh, forms_stokes['A_S'],forms_stokes['G_S'],
                                                                     forms_stokes['B_S'], 
                                                forms_stokes['Q_S'], forms_stokes['S_S'], bcs)                

            # Assemble global system and incorporates bcs
            ssc.assemble_global_system()
            # Solve using mumps
            ssc.solve_problem(Uhbar, Uh, "mumps")
                      
            # Compute velocity/pressure/local div error
            uh,ph       = Uh.split()
            e_u         = np.sqrt(np.abs(assemble(dot(uh-u_exact, uh-u_exact)*dx)) )
            e_p         = np.sqrt(np.abs(assemble((ph-p_exact) * (ph-p_exact)*dx)) )
            e_d         = np.sqrt(np.abs(assemble(div(uh)*div(uh)*dx)))
            
            if comm.rank == 0:
                error_u.append(e_u); error_p.append(e_p); error_div.append(e_d)
                print('Error in velocity '+str(error_u[-1]  ) )
                print('Error in pressure '+str(error_p[-1]  ) )
                print('Local mass error '+str(error_div[-1] ) )
        
        # Save to file
        if comm.rank == 0:
            for ncell, eu, ep, ed in zip(nx_list, error_u, error_p, error_div):
                error_file = open(outdir+"errors.out","a")
                print >> error_file, '%-25.d %-25.d %-25.4e %-25.4e %-25.4e' % (k, ncell**2, eu, ep, ed)
            error_file = open(outdir+"errors.out","a")
            print >> error_file, '-'*125
                     
if __name__ == "__main__":
    main()




