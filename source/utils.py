# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
from mpi4py import MPI as pyMPI
from dolfin import (Expression, UserExpression, Constant)

__all__ = ['assign_particle_values']

comm = pyMPI.COMM_WORLD

'''
Utility functions for LEoPart
'''


def assign_particle_values(xp, expression):
    # xp: numpy array of particle locations
    # expression: dolfin Expression or UserExpression
    # TODO: accept Function to evaluate
    # TODO: evaluate in parallel

    assert isinstance(expression, (Expression, UserExpression, Constant))
    if comm.rank == 0:
        pval = np.asarray([expression(xp[i, :]) for i in range(len(xp))], dtype=np.float_)
    else:
        pval = None
    pval = comm.bcast(pval, root=0)
    return pval
