# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import UserExpression, pi
import numpy as np

__all__ = ["BinaryBlock", "GaussianPulse", "SlottedDisk", "SineHump", "CosineHill"]


class BinaryBlock(UserExpression):
    """
    Overloaded Expression initializing a binary valued block based
    on a given geometry dictionary with keys xmin/max, and ymin/max.
    """

    def __init__(self, geometry, value_inside, value_outside, **kwargs):
        """
        Instantiate BinaryBlock class

        Parameters
        ----------
        geometry: dict
            Dictionary specifying the geometry of the binary block. Required keys are:
                - xmin: min x-coordinate of binary block
                - xmax: max x-coordinate of binary block
                - ymin: min y-coordinate of binary block
                - ymax: max y-coordinate of binary block
        value_inside: float, int
            Value inside domain [xmin, xmax] x [ymin, ymax]
        value_outside: float, int
            Value outside domain
        kwargs
        """

        # TODO: generalize for 3D
        self.geometry = geometry
        self.value_inside = value_inside
        self.value_outside = value_outside
        super().__init__(self, **kwargs)

    def eval(self, value, x):
        """
        Overload the eval method
        """
        if (self.geometry["xmin"] <= x[0] <= self.geometry["xmax"]) and (
            self.geometry["ymin"] <= x[1] <= self.geometry["ymax"]
        ):
            value[0] = self.value_inside
        else:
            value[0] = self.value_outside

    def value_shape(self):
        return ()


class GaussianPulse(UserExpression):
    """
    Overloaded expression for initializing a Gaussian pulse with variance sigma
    and centered at center.
    """

    def __init__(self, center, sigma, U, time=0, height=1.0, **kwargs):
        """
        Instantiate GaussianPulse class

        Parameters
        ----------
        center
        sigma
        U
        time
        height
        kwargs
        """
        self.center = center
        self.sigma = sigma
        self.height = height
        self.U = U
        self.t = time
        super().__init__(self, **kwargs)

    def eval(self, value, x):
        """
        Overload the eval method
        """
        xc = self.center[0]
        yc = self.center[1]
        U, t, sigma = self.U, self.t, self.sigma
        value[0] = self.height * np.exp(
            -(
                pow(x[0] * np.cos(U[0] * t) + x[1] * np.sin(U[1] * t) - xc, 2)
                + pow(-x[0] * np.sin(U[0] * t) + x[1] * np.cos(U[1] * t) - yc, 2)
            )
            / (2 * pow(sigma, 2))
        )

    def value_shape(self):
        return ()


class SlottedDisk(UserExpression):
    def __init__(self, radius, center, width, depth, lb=0.0, ub=1.0, **kwargs):
        self.r = radius
        self.width = width
        self.depth = depth
        self.center = center
        self.lb = lb
        self.ub = ub
        super().__init__(self, **kwargs)

    def eval(self, value, x):
        xc = self.center[0]
        yc = self.center[1]

        if ((x[0] - xc) ** 2 + (x[1] - yc) ** 2 <= self.r ** 2) and not (
            (xc - self.width) <= x[0] <= (xc + self.width) and x[1] >= yc + self.depth
        ):
            value[0] = self.ub
        else:
            value[0] = self.lb

    def value_shape(self):
        return ()


class SineHump(UserExpression):
    def __init__(self, center, U, time, **kwargs):
        self.center = center
        self.U = U
        self.t = time
        super().__init__(self, **kwargs)

    def eval(self, value, x):
        xc = self.center[0]
        yc = self.center[1]
        U, t = self.U, self.t
        value[0] = np.sin(2 * pi * (x[0] - xc - U[0] * t)) * np.sin(2 * pi * (x[1] - yc - U[1] * t))


class CosineHill(UserExpression):
    def __init__(self, radius, center, amplitude, **kwargs):
        self.r = radius
        self.center = center
        self.amplitude = amplitude
        # TODO: make time dependent (again?)
        # self.U = U
        # self.t = time
        super().__init__(self, **kwargs)

    def eval(self, value, x):
        xc, yc = self.center[0], self.center[1]
        # TODO: make time dependent (again)?
        # U, t = self.U, self.t
        r = min(np.sqrt(pow(x[0] - xc, 2) + pow(x[1] - yc, 2)), self.r) / self.r
        value[0] = self.amplitude * (1 + np.cos(np.pi * r))

    def value_shape(self):
        return ()
