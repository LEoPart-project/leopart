# __author__ = 'Chris Richardson <chris@bpi.cam.ac.uk>'
# __date__   = '2018-11-23'
# __copyright__ = 'Copyright (C) 2018' + __author__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

"""
    Unit tests for mesh based random point generator
"""

from dolfin import RectangleMesh, BoxMesh, Point
from DolfinParticles import MeshGenerator


def test_mesh_generator_2d():
    """Basic functionality test."""
    mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), 5, 5)
    for x in mesh.coordinates():
        x[0] += 0.5*x[1]

    w = MeshGenerator(mesh)
    pts = w.generate(3)
    assert len(pts) == mesh.num_cells()*3


def test_mesh_generator_3d():
    """Basic functionality test."""
    mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0), 5, 5, 5)
    for x in mesh.coordinates():
        x[0] += 0.5*x[1] + 0.2*x[2]

    w = MeshGenerator(mesh)
    pts = w.generate(3)
    assert len(pts) == mesh.num_cells()*3
