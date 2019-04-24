# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import (FunctionSpace, VectorFunctionSpace, Function,
                    FiniteElement, RectangleMesh, Measure, SubDomain, MeshFunction, Point,
                    FacetNormal, Constant, ALE, Expression, assign,
                    project, assemble, near, dx, File)
from leopart import (particles, PDEStaticCondensation, FormsPDEMap,
                     RandomRectangle, l2projection, CosineHill, advect_rk3,
                     assign_particle_values)
from mpi4py import MPI as pyMPI
import numpy as np

comm = pyMPI.COMM_WORLD

'''
    Cosine hill advection on moving mesh.
'''


class Left(SubDomain):
    def __init__(self, bvalue):
        SubDomain.__init__(self)
        self.bvalue = bvalue

    def inside(self, x, on_boundary):
        return near(x[0], self.bvalue)


t = 0.
T_end = 2.
dt = Constant(0.025)
num_steps = int(np.rint(T_end / float(dt)))
xmin, ymin = 0., 0.
xmax, ymax = 1., 1.
xc, yc = 0.25, 0.5
nx, ny = 32, 32
pres = 400
k = 1

# Directory for output
outdir_base = './../../results/MovingMesh/'

mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), nx, ny)
n = FacetNormal(mesh)

outfile = File(mesh.mpi_comm(), outdir_base+"psi_h.pvd")

V = VectorFunctionSpace(mesh, 'DG', 2)
Vcg = VectorFunctionSpace(mesh, 'CG', 1)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Create function spaces
Q_E_Rho = FiniteElement("DG", mesh.ufl_cell(), k)
T_1 = FunctionSpace(mesh, 'DG', 0)
Qbar_E = FiniteElement("DGT", mesh.ufl_cell(), k)

Q_Rho = FunctionSpace(mesh, Q_E_Rho)
Qbar = FunctionSpace(mesh, Qbar_E)

phih, phih0 = Function(Q_Rho), Function(Q_Rho)
phibar = Function(Qbar)

# Advective velocity
# Swirling deformation advection (see LeVeque)
ux = 'pow(sin(pi*x[0]), 2) * sin(2*pi*x[1])'
vy = '-pow(sin(pi*x[1]), 2) * sin(2*pi*x[0])'
gt_plus = '0.5 * cos(pi*t)'
gt_min = '-0.5 * cos(pi*t)'

u_expr = Expression((ux+'*'+gt_min, vy+'*'+gt_min), degree=2, t=0.)
u_expre_neg = Expression((ux+'*'+gt_plus, vy+'*'+gt_plus), degree=2, t=0.)

# Mesh velocity
umesh = Function(Vcg)

# Advective velocity
uh = Function(V)
uh.assign(u_expr)

# Total velocity
uadvect = uh - umesh

# Now throw in the particles
x = RandomRectangle(Point(xmin, ymin), Point(xmax, ymax)).generate([pres, pres])
s = assign_particle_values(x, CosineHill(radius=0.25, center=[0.25, 0.5],
                                         amplitude=1.0, degree=1))
p = particles(x, [s], mesh)

# Define projections problem
FuncSpace_adv = {'FuncSpace_local': Q_Rho, 'FuncSpace_lambda': T_1, 'FuncSpace_bar': Qbar}
FormsPDE = FormsPDEMap(mesh, FuncSpace_adv, beta_map=Constant(1e-8))
forms_pde = FormsPDE.forms_theta_linear(phih0, uadvect, dt, Constant(1.0), zeta=Constant(0.),
                                        h=Constant(0.))
pde_projection = PDEStaticCondensation(mesh, p,
                                       forms_pde['N_a'], forms_pde['G_a'], forms_pde['L_a'],
                                       forms_pde['H_a'],
                                       forms_pde['B_a'],
                                       forms_pde['Q_a'], forms_pde['R_a'], forms_pde['S_a'],
                                       [], 1)

ap = advect_rk3(p, V, uh, 'open')

# Initialize the initial condition at mesh by an l2 projection
lstsq_rho = l2projection(p, Q_Rho, 1)
lstsq_rho.project(phih0)
outfile << phih0

for step in range(num_steps):
    u_expr.t = step * float(dt)
    u_expre_neg.t = step * float(dt)

    uh.assign(u_expr)

    # Compute area at old configuration
    old_area = assemble(phih0*dx)

    # Pre-assemble rhs
    pde_projection.assemble_state_rhs()

    # Advect the particles
    ap.do_step(float(dt))

    # Move mesh
    umesh.assign(u_expre_neg)
    ALE.move(mesh, project(umesh * dt, Vcg))

    # Relocate particles as a result of mesh motion
    ap.update_facets_info()
    p.relocate()

    # Assemble left-hand side on new config, but not the right-hand side
    pde_projection.assemble(True, False)
    pde_projection.solve_problem(phibar, phih, 'mumps', 'none')

    # Area on new configuration
    new_area = assemble(phih*dx)

    # Update solution
    assign(phih0, phih)

    # Global mass error, should be machine precision
    print("Mass error "+str(new_area - old_area))

    outfile << phih0
