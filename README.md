Ncpol2sdpa
==
Ncpol2sdpa is a set of scripts to convert a noncommutative polynomial optimization problem to a sparse semidefinite programming (SDP) problem that can be processed by the [SDPA](http://sdpa.sourceforge.net/) family of solvers. The optimization problem can be unconstrained or constrained by equalities and inequalities.

The objective is to be able to solve very large scale optimization problems. For example, a convergent series of lower bounds can be obtained for ground state problems with arbitrary Hamiltonians.

Two implementations are provided, one in Python and one in Matlab. The Python implementation has a more intuitive syntax for entering Hamiltonians and it scales better for a larger number of noncommutative variables. It does not handle equality constraints efficiently. The Matlab variant is tedious to work with when the variables are Hermitian, and it introduces numerical errors in the final SDPA formulation. On the other hand, it deals with equality constraints more efficiently.

Dependencies
==
The Python code requires [PICOS](http://picos.zib.de/) and [SymPy >=0.7.2](http://sympy.org/) in the Python search path. The code is known to work with Python 2.6.8 and 2.7.5.

The Matlab code requires [SeDuMi](http://sedumi.ie.lehigh.edu/) and [Yalmip](http://users.isy.liu.se/johanl/yalmip/) in the Matlab search path.

Usage
==
A simple usage example is included in examplencpol.{py,m}. A more sophisticated application is given in hamiltonian.{py,m}, which implements the Hamiltonian of a fermionic system in a 2D grid.

Acknowledgement
==
The Matlab code for momentum generation is based on the noncommutative example in Bermuja:

http://math.berkeley.edu/~philipp/Software/Moments
