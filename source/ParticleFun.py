# __author__ = 'Jakob Maljaars <j.m.maljaars@tudelft.nl>'
# __date__   = '2018-08'
# __copyright__ = 'Copyright (C) 2011' + __author__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

"""
    SWIG wrapper for the CPP functionalities
"""

from dolfin import *
import os, inspect
import numpy as np

__all__ = ['particles', 'advect_particles', 'advect_rk2', 'advect_rk3', 'l2projection', 'PDEStaticCondensation']

# Compile C++ code
def strip_essential_code(filenames):
    code = ""
    for name in filenames:
        f = open(name, 'r').read()
        code += f
    return code

dolfin_folder = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), "../cpp"))
sources =['particles.cpp', 'advect_particles.cpp', 
          'l2projection.cpp', 'pdestaticcondensation.cpp', 'formutils.cpp']
headers = map(lambda x: os.path.join(dolfin_folder, x),['utils.h','particle.h','particles.h','advect_particles.h',
                                                        'l2projection.h', 'pdestaticcondensation.h', 'formutils.h'] )
code = strip_essential_code(headers)
  
include_dirs=[".", os.path.abspath(dolfin_folder)]
library_dirs=[".", "/usr/local/lib"]
libraries=[]

compiled_module = compile_extension_module(code=code, source_directory=os.path.abspath(dolfin_folder),
                                           sources=sources, include_dirs=include_dirs,
                                           library_dirs=library_dirs, libraries=[])

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
        
        compiled_module.particles.__init__(self,p_array, particle_template, 
                                           num_particles, mesh)
        self.ptemplate = particle_template
        return
    
    def __call__(self, *args):
        return self.eval(*args)
   
    def return_property(self,mesh, index):
        pproperty = self.get_property(index)
        if self.ptemplate[index] > 1:
            pproperty = pproperty.reshape((-1, self.ptemplate[index]))
        return pproperty
   
   
    def positions(self,mesh):
        Ndim = mesh.geometry().dim()
        xp = self.get_positions()
        xp = xp.reshape((-1, Ndim))
        return xp
    
class advect_particles(compiled_module.advect_particles):
    def __call__(self, *args):
        return self.eval(*args)     
    
class advect_rk2(compiled_module.advect_rk2):
    def __call__(self, *args):
        return self.eval(*args) 

class advect_rk3(compiled_module.advect_rk3):
    def __call__(self, *args):
        return self.eval(*args) 
    
class l2projection(compiled_module.l2projection):
    def __call__(self, *args):
        return self.eval(*args)    

class PDEStaticCondensation(compiled_module.PDEStaticCondensation):
    def __call__(self, *args):
        return self.eval(*args)
        
