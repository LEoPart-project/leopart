# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import (UserExpression)
from math import (cosh)
import numpy as np


__all__ = ['SolitonPiston', 'PeriodicPiston']

"""
    Contains mesh velocity classes.
    TODO: consider to condense in mother-child classes
"""


class SolitonPiston(UserExpression):
    """
    Piston movement for generating solitary wave, see
    Goring (YEAR)
    """

    def __init__(self, x0, xL, dt, t, g, h0, A, **kwargs):
        """
        x0 --> initial location of piston boundary
        xL --> location where mesh velocity is 0
        dt --> time step
        t --> initial time
        g --> gravity
        h0 --> dept
        A --> Target height
        """

        assert x0 < xL
        assert isinstance(dt, float)

        self.t, self.dt = t, dt
        self.x0, self.xL = x0, xL
        c = np.sqrt(g*(h0+A))
        beta = 2.*np.sqrt(0.75*A/h0**3)
        self.tau = 4./(beta*c)*(3.8+A/h0)
        self.Sg = 2*np.sqrt(A*h0/3)
        self.du = 0
        super().__init__(self, **kwargs)

    def mesh_velocity(self):
        self.du = self.__moving_bound()
        self.slope = self.du / (self.x0 - self.xL)

        # Compute for next step
        self.x0 += self.du * self.dt
        self.t += self.dt

    def eval(self, value, x):
        if x[0] > self.xL:
            value[0] = 0
        else:
            value[0] = self.slope * (x[0] - self.xL)
        value[1] = 0.

    def value_shape(self):
        return(2,)

    # Only this method is specific for class
    def __moving_bound(self):
        u_bc = 7.6/self.tau * self.Sg * (1./cosh(7.6*(self.t/self.tau-0.5)))**2
        return u_bc


class PeriodicPiston(UserExpression):
    """
    Periodic piston movement, see Eqs. (6-7) in https://doi.org/10.1016/j.coastaleng.2016.03.005
    which is derived from Ursell (1970) https://doi.org/10.1017/S0022112060000037
    """

    def __init__(self, x0, xL, dt, t, omega, d, H, g, Tramp=0, **kwargs):
        self.t, self.dt = t, dt
        self.x0, self.xL = x0, xL
        self.omega = omega

        assert x0 < xL
        assert isinstance(dt, float)

        # Compute wave number via the dispersion relationship
        k0 = omega**2 / g
        # Iterate towards k0
        kt = 0.
        it = 0
        while abs(kt-k0) > 1e-8 and it < 1000:
            kt = k0
            k0 = omega**2 / (g * np.tanh(k0 * d))
            it += 1

        # Paddle stroke
        C = 4 * pow(np.sinh(k0 * d), 2) / (2 * k0 * d + np.sinh(2 * k0 * d))
        self.A = H / (2 * C) * omega

        self.Tramp = Tramp
        super().__init__(self, **kwargs)

    def mesh_velocity(self):
        self.du = self.__moving_bound()
        self.slope = self.du / (self.x0 - self.xL)

        # Compute for next step
        self.x0 += self.du * self.dt
        self.t += self.dt

    def eval(self, value, x):
        if x[0] > self.xL:
            value[0] = 0
        else:
            value[0] = self.slope * (x[0] - self.xL)
        value[1] = 0.

    def value_shape(self):
        return(2,)

    def __moving_bound(self):
        if self.t < self.Tramp and self.Tramp > self.dt:
            u_bc = self.A * np.cos(self.omega * self.t) * self.t/self.Tramp
        else:
            u_bc = self.A * np.cos(self.omega * self.t)
        return u_bc
