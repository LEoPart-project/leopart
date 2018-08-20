#!/usr/bin/env python

from distutils.core import setup

# Version number
major = 2017
minor = 1

setup(name = "DolfinParticles",
      version = "%d.%d" % (major, minor),
      description = "LEoPART -- FEniCS functionality for advecting and projecting " \
                    "scattered particle data onto a finite element mesh",
      author = "Jakob Maljaars",
      author_email = "j.m.maljaars@tudelft.nl",
      classifiers = [
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 2.6',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      packages=['DolfinParticles'],
      package_dir={'DolfinParticles': './source'},
      package_data={'DolfinParticles': ["cpp/*.h", "cpp/*.cpp"]},
)
