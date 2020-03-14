## LEoPart -- FEniCS functionality for scattered particle advection and projection
[![CircleCI](https://circleci.com/bb/jakob_maljaars/leopart/tree/master.svg?style=svg)](https://circleci.com/bb/jakob_maljaars/leopart/tree/master)
[![Documentation Status](https://readthedocs.org/projects/leopart/badge/?version=latest)](https://leopart.readthedocs.io/en/latest/?badge=latest)

LEoPart integrates particle functionality into the open-source finite element library [FEniCS](www.fenicsproject.org). LEoPart is so much as to say **L**agrangian-**E**ulerian **o**n **Part**icles,
and might remind you of the animal with particles imprinted on its skin.

Main purpose of this library is to advect and project scattered particle data onto a mesh in an accurate and -optionally- (locally) conservative manner. By blending a finite element mesh and
Lagrangian particles, the library enables to solve hyperbolic conservation laws in an accurate and conservative manner, yet free of numerical diffusion. Alternatively, LEoPart might come in handy
if you just need to trace large numbers of particles on a fixed or moving mesh polygonal or polyhedral mesh.

As a bonus, this library shows how **static condensation** principles can be implemented in a rather efficient way in FEniCS.

For a detailed mathematical/implementation background of LEoPart reference is made to a series of papers. We would be grateful if you
cite one of these references when using LEoPart:

```
@article{maljaars2020,
  title={LEoPart: a particle library for FEniCS},
  author={Maljaars, Jakob M and Richardson, Chris N and Sime, Nathan},
  journal={arXiv preprint arXiv:1912.13375},
  year={2019}
}

@article{Maljaars2019,
  author = {Maljaars, Jakob M. and Labeur, Robert Jan and Trask, Nathaniel and Sulsky, Deborah},
  doi = {10.1016/J.CMA.2019.01.028},
  issn = {0045-7825},
  journal = {Comput. Methods Appl. Mech. Eng.},
  month = {jan},
  pages = {443--465},
  publisher = {North-Holland},
  title = {{Conservative, high-order particle-mesh scheme with applications to advection-dominated flows}},
  volume = {348},
  year = {2019}
}

@article{Maljaars2017,
  author = {Maljaars, Jakob M. and Labeur, Robert Jan and M{\"{o}}ller, Matthias},
  doi = {10.1016/j.jcp.2017.12.036},
  issn = {10902716},
  journal = {J. Comput. Phys.},
  month = {apr},
  pages = {150--172},
  publisher = {Elsevier Inc.},
  title = {{A hybridized discontinuous Galerkin framework for high-order particle--mesh operator splitting of the incompressible Navier--Stokes equations}},
  volume = {358},
  year = {2018}
}
```

---

## Dependencies
Requires FEniCS 2018.1.0 or FEniCS 2019.1.0
www.fenicsproject.org

Python 3

---

## Package contents
The package source can be found in the directory `./source.` The directory `./unit_tests` contains unit tests for checking essential parts of the code.
Various numerical examples, covering linear scalar advection and incompressible Navier-Stokes
equations are included in the ./tests directory.

---

## Documentation
Check out the (beta) documentation at https://leopart.readthedocs.io/en/latest

---

## Old version
**This repo serves as a better, cleaner and faster replacement of the purely Python based library,
which is now hosted at** https://bitbucket.org/jakob_maljaars/leopart_python


## Installation and executing the code
1. Clone the repo via

    ```
    git clone git clone https://jakob_maljaars@bitbucket.org/jakob_maljaars/dolfinparticles.git
    ```
    Using Docker? See Step 2, otherwise go to Step 3
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

## Application examples
Passive and active particle tracer modeling on fixed and moving meshes

![Alt text](https://bitbucket.org/jakob_maljaars/leopart/raw/09207324fcc39dbad388cb3c1893b2dbe95c43e5/figs/moving_mesh.png)

Mass and momentum conservative density tracking in multi-fluid flows:

![Alt text](https://bitbucket.org/jakob_maljaars/leopart/raw/09207324fcc39dbad388cb3c1893b2dbe95c43e5/figs/lock_exchange.png)

Users are encouraged to apply the code to suit their particular problem.

---

## Contact
Any questions or suggestions? Feel free to contact the developers:
j.m.maljaars _at_ tudelft.nl / jakobmaljaars _at_ gmail.com
chris _at_ bpi.cam.ac.uk

## License
Copyright (C) 2018 Maljaars et al.

This software can be redistributed and/or modified under the terms of the GNU Lesser General Public License as published by the Free Software Foundation (<http://www.gnu.org/licenses/>).

The software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
