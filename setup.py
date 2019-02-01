#!/usr/bin/env python3

from distutils.core import setup

# Version number
major = 2017
minor = 1

setup(name = "leopart",
      version = "%d.%d" % (major, minor),
      description = "LEoPart -- FEniCS functionality for advecting and projecting " \
                    "scattered particle data onto a finite element mesh",
      author = "Jakob Maljaars",
      author_email = "j.m.maljaars@tudelft.nl",
      classifiers = [
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.0',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      packages=['leopart'],
      package_dir={'leopart': './source'},
      package_data={'leopart': ["cpp/*.h", "cpp/*.cpp", "cpp/*.so"]},
)
