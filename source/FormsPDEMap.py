# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import (FacetNormal, Constant, TestFunction, TrialFunction,
                    inner, outer, dot, grad, dx, ds, dS, Form, conditional, ge)
import numpy as np


class FormsPDEMap:
    def __init__(self, mesh, FuncSpace_dict, beta_map=Constant(1E-6), ds=ds):
        self.W = FuncSpace_dict['FuncSpace_local']
        self.T = FuncSpace_dict['FuncSpace_lambda']
        self.Wbar = FuncSpace_dict['FuncSpace_bar']

        self.n = FacetNormal(mesh)
        self.beta_map = beta_map
        self.ds = ds
        self.gdim = mesh.geometry().dim()

    def forms_theta_linear(self, psih0, uh, dt, theta_map,
                           theta_L=Constant(1.0), dpsi0=Constant(0.), dpsi00=Constant(0.),
                           h=Constant(0.), neumann_idx=99,  zeta=Constant(0)):

        (psi, lamb, psibar) = self.__trial_functions()
        (w, tau, wbar) = self.__test_functions()

        beta_map = self.beta_map
        n = self.n
        facet_integral = self.facet_integral

        psi_star = psih0 + (1-theta_L)*dpsi00 + theta_L * dpsi0

        # LHS contributions
        gamma = conditional(ge(dot(uh, n), 0), 0, 1)

        # Standard formulation
        N_a = facet_integral(beta_map*dot(psi, w)) \
            + zeta * dot(grad(psi), grad(w)) * dx
        G_a = dot(lamb, w)/dt * dx - theta_map * dot(uh, grad(lamb))*w*dx  \
            + theta_map * (1-gamma) * dot(uh, n) * \
            lamb * w * self.ds(neumann_idx)
        L_a = -facet_integral(beta_map * dot(psibar, w))  # \
        H_a = facet_integral(dot(uh, n)*psibar * tau) \
            - dot(uh, n)*psibar * tau * self.ds(neumann_idx)
        B_a = facet_integral(beta_map * dot(psibar, wbar))

        # RHS contributions
        Q_a = dot(Constant(0), w) * dx
        R_a = dot(psi_star, tau)/dt*dx  \
            + (1-theta_map)*dot(uh, grad(tau))*psih0*dx \
            - (1-theta_map) * (1-gamma) * dot(uh, n) * psi_star * tau * self.ds(neumann_idx) \
            - gamma*dot(h, tau) * self.ds(neumann_idx)
        S_a = facet_integral(Constant(0) * wbar)
        return self.__fem_forms(N_a, G_a, L_a, H_a, B_a, Q_a, R_a, S_a)

    def forms_theta_nlinear(self, v0, Ubar0, dt, theta_map=Constant(1.0),
                            theta_L=Constant(1.0), duh0=Constant((0., 0.)), duh00=Constant((0., 0)),
                            h=Constant((0., 0.)), neumann_idx=99):

        # Define trial test functions
        (v, lamb, vbar) = self.__trial_functions()
        (w, tau, wbar) = self.__test_functions()

        (zero_vec, h, duh0, duh00) = self.__check_geometric_dimension(h, duh0, duh00)

        beta_map = self.beta_map
        n = self.n
        facet_integral = self.facet_integral

        # Define v_star
        v_star = v0 + (1-theta_L) * duh00 + theta_L * duh0

        Udiv = v0 + duh0
        outer_v_a = outer(w, Udiv)
        outer_v_a_o = outer(v_star, Udiv)
        outer_ubar_a = outer(vbar, Ubar0)

        # Switch to detect in/outflow boundary
        gamma = conditional(ge(dot(Udiv, n), 0), 0, 1)

        # LHS contribution s
        N_a = facet_integral(beta_map*dot(v, w))
        G_a = dot(lamb, w)/dt * dx - theta_map*inner(outer_v_a, grad(lamb))*dx \
            + theta_map * (1-gamma) * dot(outer_v_a * n,
                                          lamb) * self.ds(neumann_idx)

        L_a = -facet_integral(beta_map * dot(vbar, w))
        H_a = facet_integral(dot(outer_ubar_a*n, tau)) \
            - dot(outer_ubar_a*n, tau) * self.ds(neumann_idx)
        B_a = facet_integral(beta_map * dot(vbar, wbar))

        # RHS contributions
        Q_a = dot(Constant(zero_vec), w) * dx
        R_a = dot(v_star, tau)/dt * dx \
            + (1-theta_map)*inner(outer_v_a_o, grad(tau))*dx \
            - gamma * dot(h, tau) * self.ds(neumann_idx)
        S_a = facet_integral(dot(Constant(zero_vec), wbar))

        return self.__fem_forms(N_a, G_a, L_a, H_a, B_a, Q_a, R_a, S_a)

    def forms_theta_nlinear_np(self, v0, v_int, Ubar0, dt, theta_map=Constant(1.0),
                               theta_L=Constant(1.0), duh0=Constant((0., 0.)),
                               duh00=Constant((0., 0)), h=Constant((0., 0.)),
                               neumann_idx=99):

        '''
        No particles in mass matrix
        '''

        # Define trial test functions
        (v, lamb, vbar) = self.__trial_functions()
        (w, tau, wbar) = self.__test_functions()

        (zero_vec, h, duh0, duh00) = self.__check_geometric_dimension(h, duh0, duh00)

        beta_map = self.beta_map
        n = self.n
        facet_integral = self.facet_integral

        # Define v_star
        v_star = v0 + (1-theta_L) * duh00 + theta_L * duh0

        Udiv = v0 + duh0
        outer_v_a = outer(w, Udiv)
        outer_v_a_o = outer(v_star, Udiv)
        outer_ubar_a = outer(vbar, Ubar0)

        # Switch to detect in/outflow boundary
        gamma = conditional(ge(dot(Udiv, n), 0), 0, 1)

        # LHS contribution s
        N_a = dot(v, w) * dx + facet_integral(beta_map*dot(v, w))
        G_a = dot(lamb, w)/dt * dx - theta_map*inner(outer_v_a, grad(lamb))*dx \
            + theta_map * (1-gamma) * dot(outer_v_a * n,
                                          lamb) * self.ds(neumann_idx)

        L_a = -facet_integral(beta_map * dot(vbar, w))
        H_a = facet_integral(dot(outer_ubar_a*n, tau)) \
            - dot(outer_ubar_a*n, tau) * self.ds(neumann_idx)
        B_a = facet_integral(beta_map * dot(vbar, wbar))

        # RHS contributions
        Q_a = dot(v_int, w) * dx
        R_a = dot(v_star, tau)/dt * dx \
            + (1-theta_map)*inner(outer_v_a_o, grad(tau))*dx \
            - gamma * dot(h, tau) * self.ds(neumann_idx)
        S_a = facet_integral(dot(Constant(zero_vec), wbar))

        return self.__fem_forms(N_a, G_a, L_a, H_a, B_a, Q_a, R_a, S_a)

    def forms_theta_nlinear_multiphase(self, rho, rho0, rho00, rhobar, v0, Ubar0, dt, theta_map,
                                       theta_L=Constant(1.0), duh0=Constant((0., 0.)),
                                       duh00=Constant((0., 0)),
                                       h=Constant((0., 0.)), neumann_idx=99):

        (v, lamb, vbar) = self.__trial_functions()
        (w, tau, wbar) = self.__test_functions()

        (zero_vec, h, duh0, duh00) = self.__check_geometric_dimension(h, duh0, duh00)

        beta_map = self.beta_map
        n = self.n
        facet_integral = self.facet_integral

        # FIXME: To be deprecated
        rhov_star = rho0*(v0 + theta_L * duh0) + rho00 * (1-theta_L) * duh00

        Udiv = v0 + duh0
        outer_v_a = outer(rho * w, Udiv)
        outer_v_a_o = outer(rho0 * Udiv, Udiv)
        outer_ubar_a = outer(rhobar * vbar, Ubar0)

        # Switch to detect in/outflow boundary
        gamma = conditional(ge(dot(Udiv, n), 0), 0, 1)

        # LHS contribution
        N_a = facet_integral(beta_map*dot(v, w))
        G_a = dot(lamb, rho * w)/dt * dx - theta_map*inner(outer_v_a, grad(lamb))*dx \
            + theta_map * (1-gamma) * dot(outer_v_a * n,
                                          lamb) * self.ds(neumann_idx)

        L_a = -facet_integral(beta_map * dot(vbar, w))
        H_a = facet_integral(dot(outer_ubar_a*n, tau)) \
            - dot(outer_ubar_a*n, tau) * self.ds(neumann_idx)
        B_a = facet_integral(beta_map * dot(vbar, wbar))

        # RHS contribution
        Q_a = dot(zero_vec, w) * dx
        R_a = dot(rhov_star, tau)/dt*dx  \
            + (1-theta_map)*inner(outer_v_a_o, grad(tau))*dx \
            + gamma * dot(h, tau) * self.ds(neumann_idx)
        S_a = facet_integral(dot(zero_vec, wbar))
        return self.__fem_forms(N_a, G_a, L_a, H_a, B_a, Q_a, R_a, S_a)

    # Short-cut function for evaluating sum_{K} \int_{K} (integrand) ds
    def facet_integral(self, integrand):
        return integrand('-')*dS + integrand('+')*dS + integrand*ds

    def __fem_forms(self, N_a, G_a, L_a, H_a, B_a, Q_a, R_a, S_a):
        # Turn into forms
        N_a = Form(N_a)
        G_a = Form(G_a)
        L_a = Form(L_a)
        H_a = Form(H_a)
        B_a = Form(B_a)
        Q_a = Form(Q_a)
        R_a = Form(R_a)
        S_a = Form(S_a)
        return {'N_a': N_a, 'G_a': G_a, 'L_a': L_a, 'H_a': H_a, 'B_a': B_a,
                'Q_a': Q_a, 'R_a': R_a, 'S_a': S_a}

    def __test_functions(self):
        w = TestFunction(self.W)
        tau = TestFunction(self.T)
        wbar = TestFunction(self.Wbar)
        return (w, tau, wbar)

    def __trial_functions(self):
        psi = TrialFunction(self.W)
        lamb = TrialFunction(self.T)
        psibar = TrialFunction(self.Wbar)
        return (psi, lamb, psibar)

    def __check_geometric_dimension(self, h, duh0, duh00):
        # Get size of constants ligned up in 3D
        zero_vec = Constant(np.zeros(self.gdim))
        if self.gdim > 2:
            if h.value_size() <= 2:
                h = zero_vec
            if duh0.value_size() <= 2:
                duh0 = zero_vec
            if duh00.value_size() <= 2:
                duh00 = zero_vec
        return (zero_vec, h, duh0, duh00)

    # FIXME: to be implemented and tested
    # def forms_imex2_linear(self, alpha_dict, beta_dict, psih0, phih00, uh, dt, gamma):

        # (psi, lamb, psibar) = self.__trial_functions()
        # (w, tau, wbar) = self.__test_functions()

        # Unpack alpha/beta
        # alpha_0 = alpha_dict['alpha_0']
        # alpha_1 = alpha_dict['alpha_1']
        # beta_0 = beta_dict['beta_0']
        # beta_1 = beta_dict['beta_1']

        # Tikhonov regularization term and facet normal
        # beta_map = self.beta_map
        # n = self.n
        # facet_integral = self.facet_integral

        # LHS contributions
        # N_a = facet_integral(beta_map*dot(psi, w))
        # G_a = gamma * dot(lamb, w)/dt * dx
        # L_a = -facet_integral(beta_map * dot(psibar, w))
        # H_a = facet_integral(dot(uh, n)*psibar * tau)
        # B_a = facet_integral(beta_map * dot(psibar, wbar))

        # RHS contributions
        # Q_a = dot(Constant(0), w) * dx
        # R_a = alpha_0 * dot(psih0, tau)/dt*dx  \
        # + alpha_1 * dot(phih00, tau)/dt*dx \
        # + beta_0 * dot(uh, grad(tau))*psih0*dx \
        # + beta_1 * dot(uh, grad(tau))*phih00*dx
        # S_a = dot(Constant(0), wbar) * dx
        # return self.__fem_forms(N_a, G_a, L_a, H_a, B_a, Q_a, R_a, S_a)

    # def forms_imex2_nlinear(self, alpha_dict, beta_dict, u0_a, u00_a, uhbar0_a, dt, gamma):
        # Define trial test functions
        # (v, lamb, vbar) = self.__trial_functions()
        # (w, tau, wbar) = self.__test_functions()

        # Unpack alpha/beta
        # alpha_0 = alpha_dict['alpha_0']
        # alpha_1 = alpha_dict['alpha_1']
        # beta_0 = beta_dict['beta_0']
        # beta_1 = beta_dict['beta_1']

        # Tikhonov regularization term and facet normal
        # beta_map = self.beta_map
        # n = self.n
        # facet_integral = self.facet_integral

        # Advection map
        # outer_v_a = outer(w, u0_a)
        # outer_u_a_o = outer(u0_a, u0_a)
        # outer_u_a_oo = outer(u00_a, u00_a)
        # outer_ubar_a = outer(vbar, uhbar0_a)

        # LHS contribution
        # N_a = facet_integral(beta_map*dot(v, w))
        # G_a = gamma * dot(v, tau)/dt * dx
        # L_a = -facet_integral(beta_map * dot(vbar, w))
        # H_a = facet_integral(dot(outer_ubar_a*n, tau))
        # B_a = facet_integral(beta_map * dot(vbar, wbar))

        # RHS contribution
        # Q_a = dot(Constant((0, 0)), w) * dx
        # R_a = alpha_0 * dot(u0_a, tau)/dt*dx  \
        # + alpha_1 * dot(u00_a, tau)/dt*dx \
        # + beta_0 * inner(outer_u_a_o, grad(tau))*dx \
        # + beta_1 * inner(outer_u_a_oo, grad(tau))*dx
        # S_a = dot(Constant((0, 0)), wbar) * dx
        # return self.__fem_forms(N_a, G_a, L_a, H_a, B_a, Q_a, R_a, S_a)
