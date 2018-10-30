import warnings

__all__ = ['ParticleFun', 'FormsPDEMap', 'FormsStokes',
           'InitialConditions', 'ParticleGenerator']

try:
    from ParticleFun import *
except:
    warnings.warn("ParticleFun not installed")   

try:
    from FormsPDEMap import FormsPDEMap
except:
    warnings.warn("FormsPDEMap not installed")
    
try:
    from InitialConditions import *
except:
    warnings.warn("InitialConditions not installed")

try:
    from ParticleGenerator import *
except:
    warnings.warn("ParticleGenerator not installed")
 
try:
    from FormsStokes import *
except:
    warnings.warn("FormsStokes not installed") 
