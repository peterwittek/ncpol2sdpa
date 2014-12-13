*************************
Download and Installation
*************************
The entire package for is available as a `gzipped tar <https://pypi.python.org/packages/source/n/ncpol2sdpa/ncpol2sdpa-1.5.tar.gz>`_ file from the `Python Package Index <https://pypi.python.org/pypi/ncpol2sdpa/>`_, containing the source, documentation, and examples.

The latest development version is available on `GitHub <https://github.com/peterwittek/ncpol2sdpa>`_.

Dependencies
============
The implementation requires `SymPy <http://sympy.org/>`_ and `SciPy <http://scipy.org/>`_ in the Python search path. The code is compatible with both Python 2 and 3, but using version 3 incurs a major decrease in performance. 

While the default CPython interpreter is sufficient for small to medium-scale problems, execution time becomes excessive for larger problems. The code is compatible with Pypy. Using it yields a 10-20x speedup. If you use Pypy, SciPy is not necessary, but the removal of equations and the chordal graph extension are not supported, and you will need the `Pypy fork of Numpy <https://bitbucket.org/pypy/numpy>`_.


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
