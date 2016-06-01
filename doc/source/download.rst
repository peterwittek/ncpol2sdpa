*************************
Download and Installation
*************************
The package is available in the `Python Package Index <https://pypi.python.org/pypi/ncpol2sdpa/>`_. The latest development version is available on `GitHub <https://github.com/peterwittek/ncpol2sdpa>`_.

Dependencies
============
The implementation requires `SymPy <http://sympy.org/>`_ and `Numpy <http://www.numpy.org/>`_. The code is compatible with both Python 2 and 3, but using version 3 incurs a major decrease in performance.

While the default CPython interpreter is sufficient for small to medium-scale problems, execution time becomes excessive for larger problems. The code is compatible with Pypy. Using it yields a 10-20x speedup. If you use Pypy, you will need the `Pypy fork of Numpy <https://bitbucket.org/pypy/numpy>`_.

By default, Ncpol2sdpa does not require a solver, but then it will not be able to solve a generated relaxation either. Install any supported solver and it will be detected automatically.

Optional dependencies include:

- `SDPA <http://sdpa.sourceforge.net/>`_ is a possible target solver.
- `SciPy <http://scipy.org/>`_ yields faster execution with the default CPython interpreter.
- `PICOS <http://picos.zib.de/>`_ is necessary for using the Cvxopt solver and for converting the problem to a PICOS instance.
- `MOSEK <http://www.mosek.com/>`_ Python module is necessary to work with the MOSEK solver.
- `CVXPY <http://cvxpy.org/>`_ is required for converting the problem to or by solving it by CVXPY.
- `Cvxopt <http://cvxopt.org/>`_ is required by both Chompack and PICOS.
- `Chompack <http://chompack.readthedocs.io/>`_ improves the sparsity of the chordal graph extension.

Installation
============
Follow the standard procedure for installing Python modules:

::

    $ pip install ncpol2sdpa

If you use the development version, install it from the source code:

::

    $ git clone https://github.com/peterwittek/ncpol2sdpa.git
    $ cd ncpol2sdpa
    $ python setup.py install
