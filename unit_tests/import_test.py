from dolfin import *
from DolfinParticles import AddDelete


mesh = UnitSquareMesh(1,1)

V = FunctionSpace(mesh, 'CG', 1)
v = Function(V)
#vlist = [v.cpp_object(), v.cpp_object()]
vlist = [v, v]
print(dir(v))
AD = AddDelete(vlist)

#import LEoPART as LP

#from LEoPART import GaussianPulse, SineHump
#from LEoPART import *



#print dir(LP)
#psi0_expression = GaussianPulse( center = (0, 0), sigma = float(1.), 
                                        #U = [0.5, 0.5], time = 0., height = 1., degree = 3 )