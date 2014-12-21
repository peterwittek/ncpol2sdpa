*************************
Download and Installation
*************************
The entire package for is available as a `gzipped tar <https://pypi.python.org/packages/source/n/ncpol2sdpa/ncpol2sdpa-1.6.tar.gz>`_ file from the `Python Package Index <https://pypi.python.org/pypi/ncpol2sdpa/>`_, containing the source, documentation, and examples.

The latest development version is available on `GitHub <https://github.com/peterwittek/ncpol2sdpa>`_.

Dependencies
============
The implementation requires `SymPy <http://sympy.org/>`_ and `Numpy <http://www.numpy.org/>`_. The code is compatible with both Python 2 and 3, but using version 3 incurs a major decrease in performance. 

While the default CPython interpreter is sufficient for small to medium-scale problems, execution time becomes excessive for larger problems. The code is compatible with Pypy. Using it yields a 10-20x speedup. If you use Pypy, you will need the `Pypy fork of Numpy <https://bitbucket.org/pypy/numpy>`_.

Optional dependencies include:

  - `SciPy <http://scipy.org/>`_ allows faster execution with the default CPython interpreter, and enables removal of equations and chordal graph extensions.
  - `Chompack <http://chompack.readthedocs.org/en/latest/>`_ improves the sparsity of the chordal graph extension.
  - `PICOS <http://picos.zib.de/>`_ is necessary for converting the problem to a PICOS problem.
  - `MOSEK <http://mosek.com>`_ Python module is necessary to work with the MOSEK converter.
  - `Cvxopt <http://cvxopt.org/>`_ is required by both Chompack and PICOS.

Installation
============
Follow the standard procedure for installing Python modules:

::

    $ pip install ncpol2sdpa --user

If you use the development version, install it from the source code:

::

    $ git clone https://github.com/peterwittek/ncpol2sdpa.git
    $ cd ncpol2sdpa
    $ python setup.py install --user
