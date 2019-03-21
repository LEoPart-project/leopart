## DolfinParticles -- FEniCS functionality for scattered particle advection and projection

This library integrates particle functionality into the open-source finite element library
FEniCS (www.fenicsproject.org).

Main purpose of this library is to advect and project scattered particle data onto a mesh in
a (locally) conservative manner. By making use of a combination of a finite element mesh and
Lagrangian particles, the library enables to solve hyperbolic conservation laws in an accurate
and conservative manner, yet free of numerical diffusion.

Further documentation can be found in a series of papers, we would be grateful if you
cite one of these references when using this library:

[2] Maljaars et al., Constrained particle-mesh projections in a hybridized discontinuous Galerkin
framework with applications to advection-dominated flows, Submitted (2018), prerprint available at arXiv

[1] Maljaars et al., A hybridized discontinuous Galerkin framework for high-order particle–mesh
operator splitting of the incompressible Navier–Stokes equations, JCP (2018)

---

## Dependencies
Requires FEniCS 2018.1.0
(www.fenicsproject.org)

Python 3

---

## Package contents
The package source can be found in the directory ./source.
The directory ./unit_tests contains unit tests for checking essential parts of the code.
Various numerical examples, covering linear scalar advection and incompressible Navier-Stokes
equations are included in the ./tests directory.

---

## Documentation
Coming soon.

---

## Automated testing
Circle CI is used to perform automated
testing. Test status is:

[![CircleCI](https://circleci.com/bb/jakob_maljaars/leopart/tree/master.svg?style=svg)](https://circleci.com/bb/jakob_maljaars/leopart/tree/master)


## Installation and executing the code
1. Clone the repo via

    ```
    git clone git clone https://jakob_maljaars@bitbucket.org/jakob_maljaars/dolfinparticles.git
    ```

2. If you want to use FEniCS in Docker, run

    ```
    [sudo] docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable:[2018.1.0.r1]
    ```

    from the directory where the clone is located.

3. Compile the python wrapper:

    ```
    cd ./source/cpp
    cmake .
    make
    cd ../..
    ```

4. Add library to your PYTHONPATH

    ```
    python3 setup.py install --prefix=[YOUR PYTHONPATH]
    ```

The code runs in parallel and test found in unit_tests/tests directory can be executed as

```
mpirun -np [N] python3 [YOUR_TEST].py
```

---

## Applications
The code can among others be used for passive and active particle tracer modeling (on moving meshes)

![Alt text](https://bitbucket.org/jakob_maljaars/dolfinparticles/raw/09207324fcc39dbad388cb3c1893b2dbe95c43e5/figs/moving_mesh.png)

Other applications include mass and momentum conservative density trackin in multi-fluid flows:

![Alt text](https://bitbucket.org/jakob_maljaars/dolfinparticles/raw/09207324fcc39dbad388cb3c1893b2dbe95c43e5/figs/lock_exchange.png)

We encourage users to come up with other applications.

---

## Contact
Any questions or suggestions? Feel free to contact the developers:
j.m.maljaars _at_ tudelft.nl / jakobmaljaars _at_ gmail.com
chris _at_ bpi.cam.ac.uk

## License
Copyright (C) 2018 Maljaars et al.

This software can be redistributed and/or modified under the terms of the GNU Lesser General Public License as published by the Free Software Foundation (<http://www.gnu.org/licenses/>).

The software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
