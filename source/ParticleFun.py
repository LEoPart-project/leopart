# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import dolfin
import dolfin.cpp as cpp
from mpi4py import MPI as pyMPI
import pickle
import os

"""
    Wrapper for the CPP functionalities
"""

__all__ = [
    "particles",
    "advect_particles",
    "advect_rk2",
    "advect_rk3",
    "l2projection",
    "StokesStaticCondensation",
    "PDEStaticCondensation",
    "AddDelete",
]

from leopart.cpp import particle_wrapper as compiled_module

comm = pyMPI.COMM_WORLD


class particles(compiled_module.particles):
    """
    Python interface to cpp::particles.h
    """

    def __init__(self, xp, particle_properties, mesh):
        """
        Initialize particles.

        Parameters
        ----------
        xp: np.ndarray
            Particle coordinates
        particle_properties: list
            List of np.ndarrays with particle properties.
        mesh: dolfin.Mesh
            The mesh on which the particles will be generated.
        """

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
            assert p_property.shape[0] == xp.shape[0], "Incorrect particle property shape"
            if len(p_property.shape) == 1:
                p_array = np.append(p_array, np.array([p_property]).T, axis=1)
            else:
                p_array = np.append(p_array, p_property, axis=1)

        compiled_module.particles.__init__(self, p_array, particle_template, mesh)
        self.ptemplate = particle_template

    def interpolate(self, *args):
        """
        Interpolate field to particles. Example usage for updating the first property
        of particles. Note that first slot is always reserved for particle coordinates!

        .. code-block:: python

            p.interpolate(psi_h , 1)

        Parameters
        ----------
        psi_h: dolfin.Function
            Function which is used to interpolate
        idx: int
            Integer value indicating which particle property should be updated.

        """
        a = list(args)
        if not isinstance(a[0], cpp.function.Function):
            a[0] = a[0]._cpp_object
        super().interpolate(*tuple(a))

    def increment(self, *args):
        """
        Increment particle at particle slot by an incrementatl change
        in the field, much like the FLIP approach proposed by Brackbill

        The code to update a property psi_p at the first slot with a
        weighted increment from the current time step and an increment
        from the previous time step, can for example be implemented as:

        .. code-block:: python

            #  Particle
            p=particles(xp,[psi_p , dpsi_p_dt], msh)

            #  Incremental update with  theta =0.5, step=2
            p.increment(psih_new , psih_old ,[1, 2], theta , step

        Parameters
        ----------
        psih_new: dolfin.Function
            Function at new timestep
        psih_old: dolfin.Function
            Function at old time step
        slots: list
            Which particle slots to use? list[0] is always the quantity
            that will be updated
        theta: float, optional
            Use weighted update from current increment and previous increment/
            theta = 1: only use current increment
            theta = 0.5: average of previous increment and current increment
        step: int
            Which step are you at? The theta=0.5 increment only works from step >=2

        """

        a = list(args)
        if not isinstance(a[0], cpp.function.Function):
            a[0] = a[0]._cpp_object
        if not isinstance(a[1], cpp.function.Function):
            a[1] = a[1]._cpp_object
        super().increment(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)

    def return_property(self, mesh, index):
        """
        Return particle property by index.

        **FIXME**: mesh input argument seems redundant.

        Parameters
        ----------
        mesh: dolfin.Mesh
            Mesh
        index: int
            Integer index indicating which particle property should be returned.

        Returns
        -------
        np.array
            Numpy array which stores the particle property.
        """

        pproperty = np.asarray(self.get_property(index))
        if self.ptemplate[index] > 1:
            pproperty = pproperty.reshape((-1, self.ptemplate[index]))
        return pproperty

    def number_of_particles(self):
        """
        Get total number of particles

        Returns
        -------
        int:
            Global number of particles
        """
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

        assert isinstance(fname_list, list) and isinstance(property_list, list), (
            "Wrong dump2file" " request"
        )
        assert len(fname_list) == len(property_list), (
            "Property list and index list must " "have same length"
        )

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


def _parse_advect_particles_args(args):
    args = list(args)
    args[1] = args[1]._cpp_object
    if isinstance(args[2], dolfin.Function):
        uh_cpp = args[2]._cpp_object
        def _default_velocity_return(step, dt):
            return uh_cpp
        args[2] = _default_velocity_return
    return args



class advect_particles(compiled_module.advect_particles):
    """
    Particle advection with Euler method
    """

    def __init__(self, *args):
        """
        Initialize class

        Parameters
        ----------
        p: particles
            Particles instance
        V: dolfin.FunctionSpace
            FunctionSpace for the particle advection
            # TODO: can be derived from Function
        v: dolfin.Function
            Dolfin Function that will be used for the
            advection
        bc: string
            Boundary type. Any of "closed", "open" or "periodic"
        lims: np.array, optional
            Optional array for specifying the connected boundary parts
            in case of periodic bc's
        """
        a = _parse_advect_particles_args(args)
        super().__init__(*tuple(a))

    def do_step(self, *args):
        """
        Advect the particles over a timestep

        Parameters
        ----------
        dt: float
            Timestep
        """
        super().do_step(*args)

    def __call__(self, *args):
        return self.eval(*args)


class advect_rk2(compiled_module.advect_rk2):
    """
    Particle advection with RK2 method
    """

    def __init__(self, *args):
        """
        Initialize class

        Parameters
        ----------
        p: particles
            Particles instance
        V: dolfin.FunctionSpace
            FunctionSpace for the particle advection
            # TODO: can be derived from Function
        v: dolfin.Function
            Dolfin Function that will be used for the
            advection
        bc: string
            Boundary type. Any of "closed", "open" or "periodic"
        lims: np.array, optional
            Optional array for specifying the connected boundary parts
            in case of periodic bc's
        """
        a = _parse_advect_particles_args(args)
        super().__init__(*tuple(a))

    def do_step(self, *args):
        """
        Advect the particles over a timestep

        Parameters
        ----------
        dt: float
            Timestep
        """
        super().do_step(*args)

    def __call__(self, *args):
        return self.eval(*args)


class advect_rk3(compiled_module.advect_rk3):
    """
    RK3 advection
    """

    def __init__(self, *args):
        a = _parse_advect_particles_args(args)
        super().__init__(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)


class advect_rk4(compiled_module.advect_rk4):
    def __init__(self, *args):
        """
        Initialize class

        Parameters
        ----------
        p: particles
            Particles instance
        V: dolfin.FunctionSpace
            FunctionSpace for the particle advection
            # TODO: can be derived from Function
        v: dolfin.Function
            Dolfin Function that will be used for the
            advection
        bc: string
            Boundary type. Any of "closed", "open" or "periodic"
        lims: np.array, optional
            Optional array for specifying the connected boundary parts
            in case of periodic bc's
        """
        a = _parse_advect_particles_args(args)
        super().__init__(*tuple(a))

    def do_step(self, *args):
        """
        Advect the particles over a timestep

        Parameters
        ----------
        dt: float
            Timestep
        """
        super().do_step(*args)

    def __call__(self, *args):
        return self.eval(*args)


class l2projection(compiled_module.l2projection):
    """
    Class for handling the l2 projection from particle
    properties onto a FE function space.
    """

    def __init__(self, *args):
        """
        Initialize class

        Parameters
        ----------
        p: particles
            Particles object
        V: dolfin.FunctionSpace
            FunctionSpace that will be used for the
            projection
        property_idx: int
            Which particle property to project?
        """
        a = list(args)
        a[1] = a[1]._cpp_object
        super().__init__(*tuple(a))

    def project(self, *args):
        """
        Project particle property onto discontinuous
        FE function space

        Parameters
        ----------
        vh: dolfin.Function
            dolfin.Function into which particle properties
            are projected. Must match the specified
            FunctionSpace
        lb: float, optional
            Lowerbound which will activate a box-constrained
            projection. Should come in pairs with the upperbound
            ub.
        ub: float, optional
            Upperbound, for box-constrained projection.
            Should come in pairs with lowerbound lb
        """

        a = list(args)
        if not isinstance(a[0], cpp.function.Function):
            a[0] = a[0]._cpp_object
        super().project(*tuple(a))

    def project_cg(self, *args):
        """
        Project particle property onto continuous
        FE function space

        **NOTE**: this method is a bit a bonus and
        certainly could be improved

        Parameters
        ----------
        A: dolfin.Form
            bilinear form for the rhs
        f: dolfin.Form
            linear form for the rhs
        u: dolfin.Function
            dolfin.Function on which particle properties
            are projected.
        """
        super.project_cg(self, *args)

    def __call__(self, *args):
        return self.eval(*args)


class StokesStaticCondensation(compiled_module.StokesStaticCondensation):
    """
    Class for solving the HDG Stokes problem.
    Class interfaces the cpp StokesStaticCondensation class
    """

    def __init__(self, *args):
        """
        Parameters
        ----------
        args
        """
        super().__init__(*args)

    def solve_problem(self, *args):
        """
        Solve the Stokes problem

        Parameters
        ----------
        args
        """
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
    """
    Class for projecting the particle properties onto a discontinuous
    mesh function via a PDE-constrained projection in order to ensure
    conservation properties.

    Class interfaces PDEStaticCondensation
    """

    def solve_problem(self, *args):
        """
        Solve the PDE-constrained projection

        Parameters
        ----------
        args
        """
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
    """
    Class for adding/deleting particles
    """

    def __init__(self, *args):
        """
        Initialize class

        Parameters
        ----------
        args
        """
        a = list(args)
        for i, func in enumerate(a[3]):
            a[3][i] = func._cpp_object
        super().__init__(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)
