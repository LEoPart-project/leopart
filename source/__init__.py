# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# flake8: noqa

from .ParticleFun import (particles, advect_particles, advect_rk2, advect_rk3, l2projection,
                          StokesStaticCondensation, PDEStaticCondensation, AddDelete)
from .FormsPDEMap import FormsPDEMap
from .FormsStokes import FormsStokes
from .InitialConditions import (BinaryBlock, GaussianPulse, SineHump, CosineHill)
from .ParticleGenerator import (RandomRectangle, RandomCircle, RegularRectangle,
                                RandomBox, RegularBox)
from .utils import assign_particle_values
