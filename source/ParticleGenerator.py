# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
from math import sqrt
from itertools import product
from mpi4py import MPI as pyMPI

__all__ = ['RandomRectangle', 'RandomCircle', 'RandomBox', 'RandomSphere',
           'RegularRectangle', 'RegularBox']

comm = pyMPI.COMM_WORLD


'''
Classes for particle generation, either on regular lattice (RegularRectangle),
or randomly placed (RandomRectangle/RandomCircle)
'''


class RandomGenerator(object):

    '''
    Fill object by random points.
    '''

    def __init__(self, domain, rule):
        '''
        Domain specifies bounding box for the shape and is used to generate
        points. The rule filter points of inside the bounding box that are
        axctually inside the shape.
        '''
        assert isinstance(domain, list)
        self.domain = domain
        self.rule = rule
        self.dim = len(domain)
        self.rank = comm.Get_rank()

    def generate(self, N, method='full'):
        'Genererate points.'
        assert len(N) == self.dim
        assert method in ['full', 'tensor']
        np.random.seed(10)
        if self.rank == 0:
            # Generate random points for all coordinates
            if method == 'full':
                n_points = np.product(N)
                points = np.random.rand(n_points, self.dim)
                for i, (a, b) in enumerate(self.domain):
                    points[:, i] = a + points[:, i]*(b-a)
            # Create points by tensor product of intervals
            else:
                # Values from [0, 1) used to create points between
                # a, b - boundary
                # points in each of the directiosn
                shifts_i = np.array([np.random.rand(n) for n in N])
                # Create candidates for each directions
                points_i = (a+shifts_i[i]*(b-a)
                            for i, (a, b) in enumerate(self.domain))
                # Cartesian product of directions yield n-d points
                points = (np.array(point) for point in product(*points_i))

            # Use rule to see which points are inside
            points_inside = np.array(list(filter(self.rule, points)))
        else:
            points_inside = None

        # Broadcast to other processes
        points_inside = comm.bcast(points_inside, root=0)
        return points_inside

    # TODO generate in parallel


class RandomRectangle(RandomGenerator):
    def __init__(self, ll, ur):
        # a is lower left, b is upper right
        ax, ay = ll.x(), ll.y()
        bx, by = ur.x(), ur.y()
        assert ax < bx and ay < by
        RandomGenerator.__init__(self, [[ax, bx], [ay, by]], lambda x: True)


class RandomCircle(RandomGenerator):
    def __init__(self, center, radius):
        assert radius > 0
        domain = [[center[0]-radius, center[0]+radius],
                  [center[1]-radius, center[1]+radius]]
        RandomGenerator.__init__(self, domain,
                                 lambda x: sqrt((x[0]-center[0])**2 +
                                                (x[1]-center[1])**2) < radius
                                 )


class RandomBox(RandomGenerator):
    def __init__(self, ll, ur):
        # a is lower left, b is upper right
        ax, ay, az = ll.x(), ll.y(), ll.z()
        bx, by, bz = ur.x(), ur.y(), ur.z()
        assert ax < bx and ay < by and az < bz
        domain = [[ax, bx], [ay, by], [az, bz]]
        RandomGenerator.__init__(self, domain, lambda x: True)


class RandomSphere(RandomGenerator):
    def __init__(self, center, radius):
        assert len(center) == 3
        assert radius > 0
        domain = [[center[0]-radius, center[0]+radius],
                  [center[1]-radius, center[1]+radius],
                  [center[2]-radius, center[2]+radius]]
        RandomGenerator.__init__(self, domain,
                                 lambda x: sqrt((x[0]-center[0])**2 +
                                                (x[1]-center[1])**2 +
                                                (x[2]-center[1])**2) < radius
                                 )


class RegularRectangle(RandomGenerator):
    def __init__(self, ll, ur):
        # ll is Point(lower left coordinate), ur is Point(upper right coordinate)
        ax, ay = ll.x(), ll.y()
        bx, by = ur.x(), ur.y()
        assert ax < bx and ay < by
        RandomGenerator.__init__(self, [[ax, bx], [ay, by]], lambda x: True)

    def generate(self, N, method='open'):
        'Genererate points.'
        assert len(N) == self.dim

        if self.rank == 0:
            if method == 'closed':
                endpoint = True
            elif method == 'half open':
                endpoint = False
            elif method == 'open':
                endpoint = True
                new_domain = []
                for i, (a, b) in enumerate(self.domain):
                    delta = 0.5 * (b-a)/float(N[i])
                    a += delta
                    b -= delta
                    new_domain.append([a, b])
                self.domain = new_domain
            else:
                raise Exception('Unknown particle placement method')
            coords = []
            for i, (a, b) in enumerate(self.domain):
                coords.append(np.linspace(a, b, N[i], endpoint=endpoint))

            X, Y = np.meshgrid(coords[0], coords[1])
            points = np.vstack((np.hstack(X), np.hstack(Y))).T
            assert np.product(N) == len(points)
            points_inside = np.array(list(filter(self.rule, points)))
        else:
            points_inside = None
        points_inside = comm.bcast(points_inside, root=0)
        return points_inside


class RegularBox(RandomGenerator):
    def __init__(self, ll, ur):
        # a is lower left, b is upper right
        ax, ay, az = ll.x(), ll.y(), ll.z()
        bx, by, bz = ur.x(), ur.y(), ur.z()
        assert ax < bx and ay < by and az < bz
        domain = [[ax, bx], [ay, by], [az, bz]]
        RandomGenerator.__init__(self, domain, lambda x: True)

    def generate(self, N, method='open'):
        'Genererate points.'
        assert len(N) == self.dim
        if self.rank == 0:
            if method == 'closed':
                endpoint = True
            elif method == 'half open':
                endpoint = False
            elif method == 'open':
                endpoint = True
                new_domain = []
                for i, (a, b) in enumerate(self.domain):
                    delta = 0.5 * (b-a)/float(N[i])
                    a += delta
                    b -= delta
                    new_domain.append([a, b])
                self.domain = new_domain
            else:
                raise Exception('Unknown particle placement method')
            coords = []
            for i, (a, b) in enumerate(self.domain):
                coords.append(np.linspace(a, b, N[i], endpoint=endpoint))

            X, Y, Z = np.meshgrid(coords[0], coords[1], coords[2])
            # Unfold lists
            X_unf = np.hstack(np.hstack(X))
            Y_unf = np.hstack(np.hstack(Y))
            Z_unf = np.hstack(np.hstack(Z))
            points = np.vstack((X_unf, Y_unf, Z_unf)).T
            assert np.product(N) == len(points)
            points_inside = np.array(list(filter(self.rule, points)))
        else:
            points_inside = None
        points_inside = comm.bcast(points_inside, root=0)
        return points_inside
