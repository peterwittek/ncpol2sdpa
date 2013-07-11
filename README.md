Ncpol2sdpa
==
Ncpol2sdpa is a set of scripts to convert a noncommutative polynomial optimization problem to a sparse semidefinite programming (SDP) problem that can be processed by the [SDPA](http://sdpa.sourceforge.net/) family of solvers. The optimization problem can be unconstrained or constrained by equalities and inequalities.

The objective is to be able to solve very large scale optimization problems. For example, a convergent series of lower bounds can be obtained for ground state problems with arbitrary Hamiltonians.

The implementation has an intuitive syntax for entering Hamiltonians and it scales for a larger number of noncommutative variables using a sparse representation of the SDP problem. 

Dependencies
==
The code requires [SymPy](http://sympy.org/)>=0.7.2 in the Python search path. The code is known to work with Python 2.6.8 and 2.7.5, and also with Pypy 2.0.2. 

Usage
==
A simple usage example is included in examplencpol.py. A more sophisticated application is given in hamiltonian.py, which implements the Hamiltonian of a fermionic system in a 2D grid.

Using Pypy is recommended, as execution time is several times faster and memory use is reduced.
