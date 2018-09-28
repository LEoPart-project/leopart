from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI

# __author__ = 'Jakob Maljaars <j.m.maljaars@tudelft.nl>'
# __date__   = '2018-08'
# __copyright__ = 'Copyright (C) 2011' + __author__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

"""
    Wrapper for the CPP functionalities
"""

__all__ = ['particles', 'advect_particles', 'advect_rk2', 'advect_rk3', 'l2projection', 'PDEStaticCondensation', 'AddDelete']

from .cpp import particle_wrapper as compiled_module

comm = pyMPI.COMM_WORLD

class particles(compiled_module.particles):
    def __init__(self,xp,particle_properties, mesh):
        gdim = mesh.geometry().dim()
        particle_template = [gdim]
        num_particles = xp.shape[0]
        p_array = xp.flatten()

        for p_property in particle_properties:
            # Assert if correct size
            assert p_property.shape[0] % num_particles == 0, "Incorrect pproperty shape"

            # Check if scalar/n-d vector
            try:
                pdim = p_property.shape[1]
            except:
                pdim = int(1)

            particle_template.append(pdim)
            p_array = np.append( p_array,p_property.flatten() )

        p_array = np.asarray(p_array, dtype = np.float_)
        particle_template = np.asarray(particle_template,dtype=np.intc)

        compiled_module.particles.__init__(self, p_array, particle_template,
                                           num_particles, mesh)
        self.ptemplate = particle_template
        return

    def __call__(self, *args):
        return self.eval(*args)

    def return_property(self,mesh, index):
        pproperty = np.asarray(self.get_property(index))
        if self.ptemplate[index] > 1:
            pproperty = pproperty.reshape((-1, self.ptemplate[index]))
        return pproperty

    def positions(self,mesh):
        Ndim = mesh.geometry().dim()
        xp = np.asarray(self.get_positions())
        xp = xp.reshape((-1, Ndim))
        return xp
    
    def number_of_particles(self,mesh):
        xp_root = comm.gather( self.positions(mesh), root = 0)
        if comm.Get_rank() == 0: 
            xp_root = np.float16( np.vstack(xp_root) )
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

class PDEStaticCondensation(compiled_module.PDEStaticCondensation):
    def __call__(self, *args):
        return self.eval(*args)

class AddDelete(compiled_module.AddDelete):
    def __init__(self, *args):
        a = list(args)
        for i,func in enumerate(a[3]):
            a[3][i] = func._cpp_object
        super().__init__(*tuple(a))
    
    def __call__(self, *args):
        return self.eval(*args)