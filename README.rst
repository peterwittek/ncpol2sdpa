Ncpol2sdpa
==========
Ncpol2sdpa is a set of scripts to convert a polynomial optimization problem of either commutative or noncommutative variables to a sparse semidefinite programming (SDP) problem that can be processed by the `SDPA <http://sdpa.sourceforge.net/>`_ family of solvers. The optimization problem can be unconstrained or constrained by equalities and inequalities.

The objective is to be able to solve very large scale optimization problems. For example, a convergent series of lower bounds can be obtained for ground state problems with arbitrary Hamiltonians.

The implementation has an intuitive syntax for entering Hamiltonians and it scales for a larger number of noncommutative variables using a sparse representation of the SDP problem. 

Dependencies
============
The code requires `SymPy <http://sympy.org/>`_>=0.7.2 and `SciPy <http://scipy.org/>`_>=0.13 in the Python search path. The code is compatible with Python 3, but using it incurs a major decrease in performance. Note that since SciPy is now a dependency, Pypy is no longer supported.

Usage
=====
The following code replicates the toy example from Pironio, S.; Navascues, M. & Acin, A. Convergent relaxations of polynomial optimization problems with noncommuting variables SIAM Journal on Optimization, SIAM, 2010, 20, 2157-2180.

::

  from ncpol2sdpa import generate_variables, SdpRelaxation

  # Number of Hermitian variables
  n_vars = 2
  # Order of relaxation
  order = 2

  # Get Hermitian variables
  X = generate_variables(n_vars, hermitian=True)

  # Define the objective function
  obj = X[0] * X[1] + X[1] * X[0]

  # Inequality constraints
  inequalities = [-X[1] ** 2 + X[1] + 0.5]

  # Equality constraints
  equalities = []

  # Simple monomial substitutions
  monomial_substitution = {}
  monomial_substitution[X[0] ** 2] = X[0]

  # Obtain SDP relaxation
  sdpRelaxation = SdpRelaxation(X)
  sdpRelaxation.get_relaxation(obj, inequalities, equalities,
                               monomial_substitution, order)
  sdpRelaxation.write_to_sdpa('examplenc.dat-s')


Further examples are in the examples folder.

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

`http://arxiv.org/abs/1308.6029 <http://arxiv.org/abs/1308.6029>`_
