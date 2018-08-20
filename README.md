## DolfinParticles -- FEniCS functionality for scattered particle advection and projecion

Main purpose of this library is to advect and project scattered particle data onto a mesh in 
a (locally) conservative manner. By making use of a combination of a finite element mesh and 
Lagrangian particles, the library enables to solve hyperbolic conservation laws in an accurate 
and conservative manner, yet free of numerical diffusion. 

Further documentation can be found in a series of papers:
Maljaars et al., A hybridized discontinuous Galerkin framework for high-order particle–mesh 
operator splitting of the incompressible Navier–Stokes equations, JCP (2018)

Maljaars et al., Constrained particle-mesh projections in a hybridized discontinuous Galerkin 
framework with applications to advection-dominated flows, Submitted (2018), prerprint available at

Where the latter paper gives an in-depth description of the PDE-constrained particle-mesh 
projection, which is paramount for obtaining local conservation properties.

As for now, the repo only covers the functionalities for scalar advection problems. Extension 
to the particle-mesh operator splitting strategies as described in the aboved references 
may be implemented in the future.

---

## Dependencies
Requires FEniCS 2016.1.0, although it might work for newer versions.
(www.fenicsproject.org)

Python 2.7

---

## Package contents
The package source can be found in the directory ./source.
The directory ./unit_tests contains some test for checking some essential parts of the code.
Two numerical tests for solving the (linear) scalar advection equation are included in the 
directory ./tests.

## Installation and executing the code
1. Clone the repo via
    
    git clone git clone https://jakob_maljaars@bitbucket.org/jakob_maljaars/dolfinparticles.git

2. IF you want to use FEniCS using Docker, run 
   
    [sudo] docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable:[2016.1.0/YOUR_PREFERED_VERSION]
    
    from the directory where the clone is located. 

3. Inside Docker container, go to /shared directory and install as Python package by running commands
    1. ./setup.py build 
    
    2. [sudo] ./setup.py install
    
4. The code runs in parallel and the tests (found in unit_tests and tests directories) can be executed as, e.g.,
    
    mpirun -np [N] python test_stokes.py

    for running the test using [N] cores.

---

## Contact
Any questions or suggestions? Feel free to contact the authors:
j.m.maljaars _at_ tudelft.nl / jakobmaljaars _at_ gmail.com

## License
Copyright (C) 2018 Maljaars et al.

This software can be redistributed and/or modified under the terms of the GNU Lesser General Public License as published by the Free Software Foundation (<http://www.gnu.org/licenses/>).

The software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.