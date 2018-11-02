# __author__ = 'Jakob Maljaars <j.m.maljaars@tudelft.nl>'
# __date__   = '2018-08'
# __copyright__ = 'Copyright (C) 2011' + __author__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import UserExpression, pi
import numpy as np

__all__ = ['GaussianPulse', 'SineHump']


class GaussianPulse(UserExpression):
    """
    Overloaded expression for initializing a Gaussian pulse with variance sigma
    and centered at center.
    """

    def __init__(self, center, sigma, U, time=0, height=1.0, **kwargs):
        self.center = center
        self.sigma = sigma
        self.height = height
        self.U = U
        self.t = time
        super().__init__(self, **kwargs)

    def eval(self, value, x):
        xc = self.center[0]
        yc = self.center[1]
        U, t, sigma = self.U, self.t, self.sigma
        # value[0] = self.height * np.exp(-(pow(x[0] - xc - U[0]* x[1] * self.t,2)
        #                   +pow(x[1]-yc - U[1]* x[0] * self.t,2))/(2*pow(self.sigma,2)))

        value[0] = self.height * np.exp(-(pow(x[0]*np.cos(U[0]*t)
                                              + x[1]*np.sin(U[1]*t) - xc, 2)
                                          + pow(-x[0]*np.sin(U[0]*t)
                                                + x[1]*np.cos(U[1]*t)-yc, 2))/(2*pow(sigma, 2)))

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
        value[0] = np.sin(2*pi*(x[0] - xc - U[0]*t)) * \
            np.sin(2*pi*(x[1]-yc - U[1]*t))
