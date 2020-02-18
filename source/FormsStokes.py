# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import (Form, FacetNormal, CellDiameter, ds, dx, dS,
                    div, dot, sym, grad, inner, TestFunctions,
                    TrialFunctions, Constant, Identity, outer)
import numpy as np


class FormsStokes:
    """
    Initializes the forms for the unsteady Stokes problem
    following Labeur and Wells (2012) and
    Rhebergen and Wells (2016,2017).

    Note that we can easiliy change between
    the two formulations since pressure stabilization
    term required for Labeur and Wells (2012) formulation is
    supported.

    It defines the forms in correspondence with following
    algebraic form:

    |  A   B   C   D  | | Uh    |
    |  B^T F   0   H  | | Ph    |    |Q|
    |                 | |       | =  | |
    |  C^T 0   K   L  | | Uhbar |    |S|
    |  D^T H^T L^T P  | | Phbar |

    With part above blank line indicating the contributions
    from local momentum- and local mass conservation statement
    respectively. Part below blank line indicates the contribution
    from global momentum and global mass conservation statement.
    """

    def __init__(self, mesh, FuncSpaces_L, FuncSpaces_G, alpha,
                 beta_stab=Constant(0.), ds=ds):
        self.mixedL = FuncSpaces_L
        self.mixedG = FuncSpaces_G
        self.n = FacetNormal(mesh)
        self.beta_stab = beta_stab
        self.alpha = alpha
        self.he = CellDiameter(mesh)
        self.ds = ds
        self.gdim = mesh.geometry().dim()
        # TODO: class can be much condensed
        # Note ds(98) will be marked as free-slip boundary

    def forms_steady(self, nu, f):
        '''
        Steady Stokes
        '''

        ufl_forms = self.__ufl_forms(nu, f)
        return self.__fem_forms(ufl_forms['A_S'], ufl_forms['G_S'],
                                ufl_forms['G_ST'], ufl_forms['B_S'],
                                ufl_forms['Q_S'], ufl_forms['S_S'])

    def forms_unsteady(self, ustar, dt, nu, f):
        '''
        Forms for Backward-Euler time integration
        '''

        ufl_forms = self.__ufl_forms(nu, f)

        # Change upper left block and local rhs contribution
        (w, q, wbar, qbar) = self.__test_functions()
        (u, p, ubar, pbar) = self.__trial_functions()

        A = dot(u, w)/dt * dx
        Q = dot(ustar, w)/dt * dx

        ufl_forms['A_S'] += A
        ufl_forms['Q_S'] += Q

        return self.__fem_forms(ufl_forms['A_S'], ufl_forms['G_S'],
                                ufl_forms['G_ST'], ufl_forms['B_S'],
                                ufl_forms['Q_S'], ufl_forms['S_S'])

    def forms_multiphase(self, rho, ustar, dt, mu, f):
        '''
        Forms for Backward-Euler time integration
        two-fluid formulation Stokes
        '''

        ufl_forms = self.__ufl_forms(mu, rho * f)

        (w, q, wbar, qbar) = self.__test_functions()
        (u, p, ubar, pbar) = self.__trial_functions()

        A = rho * dot(u, w)/dt * dx
        Q = rho * dot(ustar, w)/dt * dx

        ufl_forms['A_S'] += A
        ufl_forms['Q_S'] += Q

        return self.__fem_forms(ufl_forms['A_S'], ufl_forms['G_S'],
                                ufl_forms['G_ST'], ufl_forms['B_S'],
                                ufl_forms['Q_S'], ufl_forms['S_S'])

    def facet_integral(self, integrand):
        return integrand('-')*dS + integrand('+')*dS + integrand*ds

    def __ufl_forms(self, nu, f):
        (w, q, wbar, qbar) = self.__test_functions()
        (u, p, ubar, pbar) = self.__trial_functions()

        # Infer geometric dimension
        zero_vec = np.zeros(self.gdim)

        ds = self.ds
        n = self.n
        he = self.he
        alpha = self.alpha
        beta_stab = self.beta_stab
        facet_integral = self.facet_integral

        pI = p*Identity(self.mixedL.sub(1).ufl_cell().topological_dimension())
        pbI = pbar * \
            Identity(self.mixedL.sub(1).ufl_cell().topological_dimension())

        # Upper left block
        # Contribution comes from local momentum balance
        AB = inner(2*nu*sym(grad(u)), grad(w))*dx \
            + facet_integral(dot(-2*nu*sym(grad(u))*n
                                 + (2*nu*alpha/he)*u, w)) \
            + facet_integral(dot(-2*nu*u, sym(grad(w))*n)) \
            - inner(pI, grad(w))*dx
        # Contribution comes from local mass balance
        BtF = -dot(q, div(u))*dx - \
            facet_integral(beta_stab*he/(nu+1)*dot(p, q))
        A_S = AB + BtF

        # Upper right block
        # Contribution from local momentum
        CD = facet_integral(-alpha/he*2*nu*inner(ubar, w)) \
            + facet_integral(2*nu*inner(ubar, sym(grad(w))*n)) \
            + facet_integral(dot(pbI*n, w))
        H = facet_integral(beta_stab*he/(nu+1)*dot(pbar, q))
        G_S = CD + H

        # Transpose block
        CDT = facet_integral(- alpha/he*2*nu*inner(wbar, u)) \
            + facet_integral(2*nu*inner(wbar, sym(grad(u))*n)) \
            + facet_integral(qbar * dot(u, n))
        HT = facet_integral(beta_stab*he/(nu+1)*dot(p, qbar))
        G_ST = CDT + HT

        # Lower right block, penalty on ds(98) approximates free-slip
        KL = facet_integral(alpha/he * 2 * nu*dot(ubar, wbar)) \
            - facet_integral(dot(pbar*n, wbar)) \
            + Constant(1E12)/he * inner(outer(ubar, wbar), outer(n, n)) * ds(98)
        LtP = - facet_integral(dot(ubar, n)*qbar) \
            - facet_integral(beta_stab*he/(nu+1) * pbar * qbar)
        B_S = KL + LtP

        # Righthandside
        Q_S = dot(f, w)*dx
        S_S = facet_integral(dot(Constant(zero_vec), wbar))
        return {'A_S': A_S, 'G_S': G_S, 'G_ST': G_ST,
                'B_S': B_S, 'Q_S': Q_S, 'S_S': S_S}

    def __fem_forms(self, A_S, G_S, G_ST, B_S, Q_S, S_S):
        # Turn into forms
        A_S = Form(A_S)
        G_S = Form(G_S)
        G_ST = Form(G_ST)
        B_S = Form(B_S)
        Q_S = Form(Q_S)
        S_S = Form(S_S)
        return {'A_S': A_S, 'G_S': G_S, 'G_ST': G_ST,
                'B_S': B_S, 'Q_S': Q_S, 'S_S': S_S}

    def __test_functions(self):
        w, q = TestFunctions(self.mixedL)
        wbar, qbar = TestFunctions(self.mixedG)
        return (w, q, wbar, qbar)

    def __trial_functions(self):
        u, p = TrialFunctions(self.mixedL)
        ubar, pbar = TrialFunctions(self.mixedG)
        return (u, p, ubar, pbar)
