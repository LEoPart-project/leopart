# flake8: noqa

from .ParticleFun import (particles, advect_particles, advect_rk2, advect_rk3, l2projection,
                          StokesStaticCondensation, PDEStaticCondensation, AddDelete)
from .FormsPDEMap import FormsPDEMap
from .FormsStokes import FormsStokes
from .InitialConditions import GaussianPulse, SineHump
from .ParticleGenerator import (RandomRectangle, RandomCircle, RegularRectangle,
                                RandomBox, RegularBox)
