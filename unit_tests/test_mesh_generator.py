# __author__ = 'Chris Richardson <chris@bpi.cam.ac.uk>'
# __date__   = '2018-11-23'
# __copyright__ = 'Copyright (C) 2018' + __author__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

"""
    Unit tests for mesh based random point generator
"""

from dolfin import RectangleMesh, BoxMesh, Point, Expression
from leopart import particles, MeshGenerator, assign_particle_values
import numpy as np


def test_mesh_generator_2d():
    """Basic functionality test."""
    mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), 5, 5)
    for x in mesh.coordinates():
        x[0] += 0.5*x[1]

    w = MeshGenerator(mesh)
    pts = w.generate(3)
    assert len(pts) == mesh.num_cells()*3

    interpolate_expression = Expression("x[0] + x[1]", degree=1)
    s = assign_particle_values(pts, interpolate_expression, on_root=False)
    p = particles(pts, [s], mesh)

    assert np.linalg.norm(np.sum(pts, axis=1) - s) <= 1e-15
    assert np.linalg.norm(pts-p.positions()) <= 1e-15


def test_mesh_generator_3d():
    """Basic functionality test."""
    mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0), 5, 5, 5)
    for x in mesh.coordinates():
        x[0] += 0.5*x[1] + 0.2*x[2]

    w = MeshGenerator(mesh)
    pts = w.generate(3)
    assert len(pts) == mesh.num_cells()*3

    interpolate_expression = Expression("x[0] + x[1] + x[2]", degree=1)
    s = assign_particle_values(pts, interpolate_expression, on_root=False)
    p = particles(pts, [s], mesh)

    assert np.linalg.norm(np.sum(pts, axis=1) - s) <= 1e-15
    assert pts.shape == p.positions().shape
