# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import dolfin.cpp as cpp
from mpi4py import MPI as pyMPI
import pickle
import os

"""
    Wrapper for the CPP functionalities
"""

__all__ = ['particles', 'advect_particles', 'advect_rk2', 'advect_rk3', 'l2projection',
           'StokesStaticCondensation', 'PDEStaticCondensation', 'AddDelete']

from .cpp import particle_wrapper as compiled_module

comm = pyMPI.COMM_WORLD


class particles(compiled_module.particles):
    def __init__(self, xp, particle_properties, mesh):

        gdim = mesh.geometry().dim()

        particle_template = [gdim]
        for p in particle_properties:
            if len(p.shape) == 1:
                particle_template.append(1)
            else:
                particle_template.append(p.shape[1])

        p_array = xp
        for p_property in particle_properties:
            # Assert if correct size
            assert p_property.shape[0] == xp.shape[0], \
                "Incorrect particle property shape"
            if len(p_property.shape) == 1:
                p_array = np.append(p_array, np.array([p_property]).T, axis=1)
            else:
                p_array = np.append(p_array, p_property, axis=1)

        compiled_module.particles.__init__(self, p_array, particle_template,
                                           mesh)
        self.ptemplate = particle_template
        return

    def interpolate(self, *args):
        a = list(args)
        if not isinstance(a[0], cpp.function.Function):
            a[0] = a[0]._cpp_object
        super().interpolate(*tuple(a))

    def increment(self, *args):
        a = list(args)
        if not isinstance(a[0], cpp.function.Function):
            a[0] = a[0]._cpp_object
        if not isinstance(a[1], cpp.function.Function):
            a[1] = a[1]._cpp_object
        super().increment(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)

    def return_property(self, mesh, index):
        pproperty = np.asarray(self.get_property(index))
        if self.ptemplate[index] > 1:
            pproperty = pproperty.reshape((-1, self.ptemplate[index]))
        return pproperty

    def number_of_particles(self):
        xp_root = comm.gather(self.positions(), root=0)
        if comm.rank == 0:
            xp_root = np.float16(np.vstack(xp_root))
            num_particles = len(xp_root)
        else:
            num_particles = None
        num_particles = comm.bcast(num_particles, root=0)
        return num_particles

    def dump2file(self, mesh, fname_list, property_list, mode, clean_old=False):
        if isinstance(fname_list, str) and isinstance(property_list, int):
            fname_list = [fname_list]
            property_list = [property_list]

        assert isinstance(fname_list, list) and isinstance(property_list, list), ("Wrong dump2file"
                                                                                  " request")
        assert len(fname_list) == len(property_list), ('Property list and index list must '
                                                       'have same length')

        # Remove files if clean_old = True
        if clean_old:
            for fname in fname_list:
                try:
                    os.remove(fname)
                except OSError:
                    pass

        for (property_idx, fname) in zip(property_list, fname_list):
            property_root = comm.gather(self.return_property(mesh, property_idx).T, root=0)
            if comm.Get_rank() == 0:
                with open(fname, mode) as f:
                    property_root = np.float16(np.hstack(property_root).T)
                    pickle.dump(property_root, f)
        return


class advect_particles(compiled_module.advect_particles):
    def __init__(self, *args):
        a = list(args)
        a[1] = a[1]._cpp_object
        a[2] = a[2]._cpp_object
        super().__init__(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)


class advect_rk2(compiled_module.advect_rk2):
    def __init__(self, *args):
        a = list(args)
        a[1] = a[1]._cpp_object
        a[2] = a[2]._cpp_object
        super().__init__(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)


class advect_rk3(compiled_module.advect_rk3):
    def __init__(self, *args):
        a = list(args)
        a[1] = a[1]._cpp_object
        a[2] = a[2]._cpp_object
        super().__init__(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)


class l2projection(compiled_module.l2projection):
    def __init__(self, *args):
        a = list(args)
        a[1] = a[1]._cpp_object
        super().__init__(*tuple(a))

    def project(self, *args):
        a = list(args)
        if not isinstance(a[0], cpp.function.Function):
            a[0] = a[0]._cpp_object
        super().project(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)


class StokesStaticCondensation(compiled_module.StokesStaticCondensation):
    def solve_problem(self, *args):
        a = list(args)
        for i, arg in enumerate(a):
            # Check because number of functions is either 2 or 3
            if not isinstance(arg, str):
                if not isinstance(arg, cpp.function.Function):
                    a[i] = a[i]._cpp_object
        super().solve_problem(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)


class PDEStaticCondensation(compiled_module.PDEStaticCondensation):
    def solve_problem(self, *args):
        a = list(args)
        for i, arg in enumerate(a):
            # Check because number of functions is either 2 or 3
            if not isinstance(arg, str):
                if not isinstance(arg, cpp.function.Function):
                    a[i] = a[i]._cpp_object
        super().solve_problem(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)


class AddDelete(compiled_module.AddDelete):
    def __init__(self, *args):
        a = list(args)
        for i, func in enumerate(a[3]):
            a[3][i] = func._cpp_object
        super().__init__(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)
