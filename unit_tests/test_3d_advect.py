from dolfin import *
from mpi4py import MPI as pyMPI
import numpy as np
import os
import pytest

# Load from package
from DolfinParticles import (particles, advect_rk3, advect_rk2, advect_particles,
                     RandomBox, RegularBox)

#
comm = pyMPI.COMM_WORLD

@pytest.mark.parametrize('advection_scheme', ['euler', 'rk2', 'rk3'])
def test_advect_periodic(advection_scheme):
    xmin, ymin, zmin = 0., 0., 0.
    xmax, ymax, zmax = 1., 1., 1.

    mesh = UnitCubeMesh(10, 10, 10)

    lims = np.array([[xmin, xmin, ymin, ymax, zmin, zmax],[xmax, xmax, ymin, ymax, zmin, zmax],
                    [xmin, xmax, ymin, ymin, zmin, zmax],[xmin, xmax, ymax, ymax, zmin, zmax],
                    [xmin, xmax, ymin, ymax, zmin, zmin],[xmin, xmax, ymin, ymax, zmax, zmax]
                    ])

    vexpr = Constant((1.,1.,1.))
    V = VectorFunctionSpace(mesh,"CG", 1)
    v = Function(V)
    v.assign(vexpr)

    x = RandomBox(Point(0.,0.,0.), Point(1.,1.,1.)).generate([10,10,10])
    x = comm.bcast(x, root=0)
    dt= 0.05

    p = particles(x, [x*0, x**2], mesh)

    if advection_scheme == 'euler':
        ap= advect_particles(p, V, v, 'periodic', lims.flatten(), 'none')
    elif advection_scheme == 'rk2':
        ap= advect_rk2(p, V, v, 'periodic', lims.flatten(), 'none')
    elif advection_scheme == 'rk3':
        ap= advect_rk3(p, V, v, 'periodic', lims.flatten(), 'none')

    xp0 = p.positions()
    t  = 0.
    while t<1.-1e-12:
        ap.do_step(dt)
        t += dt
    xpE = p.positions()

    xp0_root = comm.gather( xp0, root = 0)
    xpE_root = comm.gather( xpE, root = 0)

    assert len(xp0) == len(xpE)

    if comm.Get_rank() == 0:
        xp0_root = np.float32( np.vstack(xp0_root) )
        xpE_root = np.float32( np.vstack(xpE_root) )

        # Sort on x positions
        xp0_root = xp0_root[xp0_root[:,0].argsort(),:]
        xpE_root = xpE_root[xpE_root[:,0].argsort(),:]

        error = np.linalg.norm(xp0_root - xpE_root)
        assert error < 1e-10
