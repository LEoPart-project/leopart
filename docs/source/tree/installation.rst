Installation
============

Clone LEoPart using :code:`git` into a subdiractory of you location of choice with::

    git clone https://bitbucket.org/jakob_maljaars/leopart/
    cd leopart


To compile and run LEoPart, the following dependencies are required:

* `FEniCS <fenicsproject.org>`_
* `pybind11 <https://github.com/pybind/pybind11>`_
* `CMAKE <https://cmake.org/>`_

A :code:`conda` environment is provided containing all the dependencies. To get this environment up-and-running::

    conda create -f envs/environment.yml
    conda activate leopart

Nexct, Compile the :code:`cpp` source code by running::

    cd source/cpp
    cmake . && make

And install as python package::

    cd ../..
    [sudo] python3 setup.py install

You now should be able to use :code:`leopart` from :code:`python` as:

.. code-block:: python

    import leopart as lp

Or any appropriate import syntax.