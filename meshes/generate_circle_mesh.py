# -*- coding: utf-8 -*-
import pygmsh
import meshio

geom = pygmsh.built_in.Geometry()
lcar_list = [0.1, 0.05, 0.025, 0.0125, 0.00625]

for i, lcar in enumerate(lcar_list):
    geom.add_circle(
        [0.0, 0.0, 0.0],
        0.5,
        lcar=lcar,
        num_sections=4,
        # If compound==False, the section borders have to be points of the
        # discretization. If using a compound circle, they don't; gmsh can
        # choose by itself where to point the circle points.
        compound=True,
    )

    points, cells, _, _, _ = pygmsh.generate_mesh(geom)

    # Prune topology and save 2D
    cells = {'triangle': cells['triangle']}
    points = points[:, :2]

    meshio.write_points_cells("circle_"+str(i)+".xml", points, cells)
