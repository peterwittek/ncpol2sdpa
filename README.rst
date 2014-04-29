Ncpol2sdpa
==========
Ncpol2sdpa is a set of scripts to convert a noncommutative polynomial optimization problem to a sparse semidefinite programming (SDP) problem that can be processed by the `SDPA <http://sdpa.sourceforge.net/>`_ family of solvers. The optimization problem can be unconstrained or constrained by equalities and inequalities.

The objective is to be able to solve very large scale optimization problems. For example, a convergent series of lower bounds can be obtained for ground state problems with arbitrary Hamiltonians.

The implementation has an intuitive syntax for entering Hamiltonians and it scales for a larger number of noncommutative variables using a sparse representation of the SDP problem. 

Dependencies
============
The code requires `SymPy <http://sympy.org/>`_>=0.7.2 in the Python search path. The code is known to work with Python 2.6.8 and 2.7.5, and also with Pypy 1.8 and 2.0.2. Using Pypy is highly recommended, as execution time is several times faster and memory use is reduced. The code is compatible with Python 3, but using Python 3.3.2 incurs a major decrease in performance; the case is likely to be similar in with other Python 3 versions.

Usage
=====
The following code replicates the toy example from ironio, S.; Navascués, M. & Acín, A. Convergent relaxations of polynomial optimization problems with noncommuting variables SIAM Journal on Optimization, SIAM, 2010, 20, 2157-2180.

``from ncpol2sdpa.ncutils import generate_ncvariables
from ncpol2sdpa.sdprelaxation import SdpRelaxation

#Number of Hermitian variables
n_vars = 2
#Order of relaxation
order = 2

#Get Hermitian variables
X = generate_ncvariables(n_vars)

#Define the objective function
obj = X[0] * X[1] + X[1] * X[0]

# Inequality constraints
inequalities = [ -X[1]**2 + X[1] + 0.5 ]

# Equality constraints
equalities = []

#Simple monomial substitutions
monomial_substitution = {}
monomial_substitution[X[0]**2] = X[0]

#Obtain SDP relaxation
sdpRelaxation = SdpRelaxation(X)
sdpRelaxation.get_relaxation(obj, inequalities, equalities, 
                      monomial_substitution, order)
sdpRelaxation.write_to_sdpa('examplenc.dat-s')``

Further examples are under the examples folder.

Installation
============
Follow the standard procedure for installing Python modules:

``$ sudo python_interpreter setup.py install``

If you install the module to your CPython library, but you want to use Pypy, please ensure that the PYTHONPATH variable is set up correctly, otherwise Pypy will not find the relevant modules.

Acknowledgment
==============
This work is supported by the European Commission Seventh Framework Programme under Grant Agreement Number FP7-601138 `PERICLES <http://pericles-project.eu/>`_, by the `Red Española de Supercomputación <http://www.bsc.es/RES>`_ grants number FI-2013-1-0008 and  FI-2013-3-0004, and by the `Swedish National Infrastructure for Computing <http://www.snic.se/>`_ project number SNIC 2014/2-7.

More Information
================
For more information refer to the following manuscript:

`http://arxiv.org/abs/1308.6029 <http://arxiv.org/abs/1308.6029>`_
