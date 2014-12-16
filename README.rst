Ncpol2sdpa
==========
Ncpol2sdpa is a tool to convert a polynomial optimization problem of either commutative or noncommutative variables to a sparse semidefinite programming (SDP) problem that can be processed by the `SDPA <http://sdpa.sourceforge.net/>`_ family of solvers, `MOSEK <http://www.mosek.com/>`_, or further processed by `PICOS <http://picos.zib.de/>`_ to solve the problem by `CVXOPT <http://cvxopt.org/>`_ . The optimization problem can be unconstrained or constrained by equalities and inequalities.

The objective is to be able to solve very large scale optimization problems. Example applications include:

- `Ground-state energy problems <http://dx.doi.org/10.1137/090760155>`_: bosonic and `fermionic systems <http://nbviewer.ipython.org/github/peterwittek/ipython-notebooks/blob/master/Comparing_DMRG_ED_and_SDP.ipynb>`_, Pauli spin operators.
- `Maximum quantum violation <http:/dx.doi.org/10.1103/PhysRevLett.98.010401>`_ of `Bell inequalities <http://peterwittek.com/2014/06/quantum-bound-on-the-chsh-inequality-using-sdp/>`_.
- `Nieto-Silleras <http://dx.doi.org/10.1088/1367-2630/16/1/013035>`_ hierarchy for `quantifying randomness <http://peterwittek.com/2014/11/the-nieto-silleras-and-moroder-hierarchies-in-ncpol2sdpa/>`_.
- `Moroder <http://dx.doi.org/10.1103/PhysRevLett.111.030501>`_ hierarchy to enable PPT-style and other additional constraints.
- If using commutative variables, the hierarchy is identical to `Lasserre's <http://dx.doi.org/10.1137/S1052623400366802>`_.

The implementation has an intuitive syntax for entering problems and it scales for a larger number of noncommutative variables using a sparse representation of the SDP problem. 

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


Usage
=====
Documentation is available `online <http://peterwittek.github.io/ncpol2sdpa/>`_. The following code replicates the toy example from Pironio, S.; Navascues, M. & Acin, A. Convergent relaxations of polynomial optimization problems with noncommuting variables SIAM Journal on Optimization, SIAM, 2010, 20, 2157-2180.

::

  from ncpol2sdpa import generate_variables, SdpRelaxation, write_to_sdpa

  # Number of Hermitian variables
  n_vars = 2
  # Level of relaxation
  level = 2

  # Get Hermitian variables
  X = generate_variables(n_vars, hermitian=True)

  # Define the objective function
  obj = X[0] * X[1] + X[1] * X[0]

  # Inequality constraints
  inequalities = [-X[1] ** 2 + X[1] + 0.5]

  # Simple monomial substitutions
  monomial_substitution = {}
  monomial_substitution[X[0] ** 2] = X[0]

  # Obtain SDP relaxation
  sdpRelaxation = SdpRelaxation(X)
  sdpRelaxation.get_relaxation(level, objective=obj, inequalities=inequalities,
                               substitutions=monomial_substitution)
  write_to_sdpa(sdpRelaxation, 'examplenc.dat-s')


Further instances are in the examples folder and also in the manual.

Installation
============
The code is available on PyPI, hence it can be installed by 

``$ sudo pip install ncpol2sdpa``

If you want the latest git version, follow the standard procedure for installing Python modules:

``$ sudo python setup.py install``

Acknowledgment
==============
This work is supported by the European Commission Seventh Framework Programme under Grant Agreement Number FP7-601138 `PERICLES <http://pericles-project.eu/>`_, by the `Red Espanola de Supercomputacion <http://www.bsc.es/RES>`_ grants number FI-2013-1-0008 and  FI-2013-3-0004, and by the `Swedish National Infrastructure for Computing <http://www.snic.se/>`_ project number SNIC 2014/2-7.

More Information
================
For more information refer to the following manuscript:

Wittek, P. (2014). `Ncpol2sdpa -- Sparse Semidefinite Programming Relaxations for Polynomial Optimization Problems of Noncommuting Variables<http://arxiv.org/abs/1308.6029>`_. To Appear in ACM Transactions on Mathematical Software.
