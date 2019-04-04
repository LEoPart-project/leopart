# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import (Form, FacetNormal, CellDiameter, ds, dx, dS,
                    dot, grad, TestFunction, TrialFunction, Constant)


class FormsDiffusion:
    """
    Initializes the forms for the diffusion problem
    following Labeur and Wells (2007)
    """

    def __init__(self, mesh, FuncSpace_L, FuncSpace_G, alpha, ds=ds):
        self.FSpace_L = FuncSpace_L
        self.FSpace_G = FuncSpace_G
        self.n = FacetNormal(mesh)
        self.alpha = alpha
        self.he = CellDiameter(mesh)
        self.ds = ds
        self.gdim = mesh.geometry().dim()
        # Note ds(98) is marked as inhomogeneous Neumann

    def forms_steady(self, kappa, f=Constant(0), h=Constant(0)):
        ufl_forms = self.__ufl_forms(kappa, f, h)
        return self.__fem_forms(ufl_forms['A'], ufl_forms['G'],
                                ufl_forms['G_T'], ufl_forms['B'],
                                ufl_forms['Q'], ufl_forms['S'])

    def forms_unsteady(self, phi0, dt, kappa, theta, phibar0=Constant(0),
                       f=Constant(0), h=Constant(0)):
        pass

    def __ufl_forms(self, kappa, f, h, theta=Constant(1.0),
                    phi0=Constant(0), phibar0=Constant(0)):
        (q, qbar) = self.__test_functions()
        (phi, phibar) = self.__trial_functions()

        ds = self.ds
        n = self.n
        he = self.he
        alpha = self.alpha
        facet_integral = self.facet_integral
        beta = - alpha*kappa / he

        A = theta * dot(kappa*grad(phi), grad(q)) * dx \
            - theta * facet_integral(kappa * dot(grad(phi), n) * q
                                     + kappa * dot(phi * n, grad(q))
                                     + beta * phi * q)
        G = theta * facet_integral(beta * phibar * q
                                   + kappa * dot(phibar*n, grad(q)))
        G_T = - theta * facet_integral(beta * phi * qbar
                                       + kappa * dot(qbar*n, grad(phi)))
        B = theta * facet_integral(beta * phibar * qbar)

        # Righthandside
        Q = dot(f, q) * dx \
            - (1-theta) * dot(kappa*grad(phi0), grad(q)) * dx \
            + (1-theta) * facet_integral(kappa * dot(grad(phi0), n) * q
                                         + kappa * dot(phi0 * n, grad(q))
                                         + beta * phi0 * q) \
            - (1-theta) * facet_integral(beta * phibar0 * q
                                         + kappa * dot(phibar0*n, grad(q)))
        S = facet_integral(Constant(0) * qbar) \
            + (1-theta) * facet_integral(beta * phi0 * qbar
                                         + kappa * dot(qbar*n, grad(phi0))) \
            - (1-theta) * facet_integral(beta * phibar0 * qbar) \
            + h * qbar * ds(98)
        return {'A': A, 'G': G, 'G_T': G_T,
                'B': B, 'Q': Q, 'S': S}

    def __fem_forms(self, A, G, G_T, B, Q, S):
        # Turn into forms
        (A, G, G_T, B) = (Form(A), Form(G), Form(G_T), Form(B))
        (Q, S) = (Form(Q), Form(S))
        return {'A': A, 'G': G, 'G_T': G_T,
                'B': B, 'Q': Q, 'S': S}

    def facet_integral(self, integrand):
        return integrand('-')*dS + integrand('+')*dS + integrand*ds

    def __test_functions(self):
        q = TestFunction(self.mixedL)
        qbar = TestFunction(self.mixedG)
        return (q, qbar)

    def __trial_functions(self):
        phi = TrialFunction(self.mixedL)
        phibar = TrialFunction(self.mixedG)
        return (phi, phibar)
