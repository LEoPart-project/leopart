import numpy as np
from mpi4py import MPI as pyMPI

# __author__ = 'Jakob Maljaars <j.m.maljaars@tudelft.nl>'
# __date__   = '2018-08'
# __copyright__ = 'Copyright (C) 2011' + __author__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

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

    def __call__(self, *args):
        return self.eval(*args)

    def return_property(self, mesh, index):
        pproperty = np.asarray(self.get_property(index))
        if self.ptemplate[index] > 1:
            pproperty = pproperty.reshape((-1, self.ptemplate[index]))
        return pproperty

    def number_of_particles(self, mesh):
        xp_root = comm.gather(self.positions(mesh), root=0)
        if comm.Get_rank() == 0:
            xp_root = np.float16(np.vstack(xp_root))
            print("Number of particles is "+str(len(xp_root)))
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

    def __call__(self, *args):
        return self.eval(*args)


class StokesStaticCondensation(compiled_module.StokesStaticCondensation):
    def __call__(self, *args):
        return self.eval(*args)


class PDEStaticCondensation(compiled_module.PDEStaticCondensation):
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
