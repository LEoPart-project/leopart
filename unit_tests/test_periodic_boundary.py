from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
    Set-up class for PeriodicBoundary in 3D box
"""

class PeriodicBoundary(SubDomain):
    def __init__(self, ddict):
        SubDomain.__init__(self)
        self.xmin, self.xmax = ddict['xmin'], ddict['xmax']
        self.ymin, self.ymax = ddict['ymin'], ddict['ymax']
        self.zmin, self.zmax = ddict['zmin'], ddict['zmax']
          
    def inside(self, x, on_boundary):
        xmin, xmax, ymin, ymax, zmin, zmax = self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax
        #xmin, ymin, zmin, xmax, ymax, zmax = 0. , 0., 0., 1., 1., 1.
        return  bool( ( near(x[0], xmin) or  near(x[1], ymin) or near(x[2], zmin)) and \
                    (not (( near(x[0], xmin) and near(x[1], ymax) ) or \
                          ( near(x[0], xmin) and near(x[2], zmax) ) or \
                         #( near(x[0], xmax) and near(x[2], zmax) ) or \
                          ( near(x[1], ymin) and near(x[2], zmax) ) or \
                          ( near(x[1], ymax) and near(x[2], zmin) ) or \
                          ( near(x[0], xmax) and near(x[2], zmin) ) or \
                          ( near(x[0], xmax) and near(x[1], ymin) )    )) )    
    
    def map(self, x, y):
        xmin, xmax, ymin, ymax, zmin, zmax = self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax
        # Is this corner projection really needded? 
        if near(x[0], xmax) and near(x[1], ymax) and near(x[2], zmax):
            y[0] = x[0] - (xmax - xmin)
            y[1] = x[1] - (ymax - ymin)
            y[2] = x[2] - (zmax - zmin)
        ## This handles the corners
        #elif near(x[0], xmax) and near(x[1], ymin) and near(x[2], zmax):
            #y[0] = x[0] - (xmax - xmin)
            #y[1] = x[1] 
            #y[2] = x[2] - (zmax - zmin)
        #elif near(x[0], xmax) and near(x[1], ymax) and near(x[2], zmin):    
            #y[0] = x[0] - (xmax - xmin)
            #y[1] = x[1] - (ymax - ymin)
            #y[2] = x[2] 
        #elif near(x[0], xmin) and near(x[1], ymax) and near(x[2], zmax):
            #y[0] = x[0] 
            #y[1] = x[1] - (ymax - ymin)
            #y[2] = x[2] - (zmax - zmin)
        ## This handles the outer edges
        elif near(x[0], xmax) and near(x[2], zmax):  
            y[0] = x[0] - (xmax - xmin)
            y[1] = x[1] 
            y[2] = x[2] - (zmax - zmin)
        elif near(x[0], xmax) and near(x[1], ymax):
            y[0] = x[0] - (xmax - xmin)
            y[1] = x[1] - (ymax - ymin)
            y[2] = x[2] 
        elif near(x[1], ymax) and near(x[2], zmax):    
            y[0] = x[0] 
            y[1] = x[1] - (ymax - ymin)
            y[2] = x[2] - (zmax - zmin)   
        elif near(x[0], xmax):
            y[0] = x[0] - (xmax - xmin)
            y[1] = x[1]
            y[2] = x[2]
        elif near(x[1], ymax):
            y[0] = x[0] 
            y[1] = x[1] - (ymax - ymin)
            y[2] = x[2]
        elif near(x[2], zmax):
            y[0] = x[0] 
            y[1] = x[1] 
            y[2] = x[2]- (zmax - zmin)

#mesh = UnitCubeMesh(2,2,2)

geometry = {'xmin': 0., 'ymin': 0., 'zmin': 0., 'xmax': 1., 'ymax': 1., 'zmax': 1.}
xmin, ymin, zmin = geometry['xmin'], geometry['ymin'], geometry['zmin']
xmax, ymax, zmax = geometry['xmax'], geometry['ymax'], geometry['zmax']

nx, ny, nz = 24,24,24

mesh = BoxMesh(Point(xmin,ymin,zmin), Point(xmax,ymax,zmax), nx, ny, nz)
bmesh= BoundaryMesh(mesh, 'exterior')
pbc  = PeriodicBoundary(geometry)



V = FunctionSpace(mesh, 'CG', 1, constrained_domain = pbc)
v = Function(V)
print("Number of dofs")
print(len(v.vector().array()))

print("Dof coordinates")
print(np.reshape( V.tabulate_dof_coordinates(), (-1,3)))



