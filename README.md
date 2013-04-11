Ncpol2sdpa
==
Ncpol2sdpa is a Matlab script to convert a noncommutative polynomial optimization problem to a sparse SDP problem that can be processed by the SDPA family of solvers. The optimization problem can be unconstrained or constrained by equalities and inequalities.

The objective is to be able to solve very large scale optimization problems. For example, a convergent series of lower bounds can be obtained for ground state problems with arbitrary Hamiltonians.

Dependencies
==
The code requires [SeDuMi](http://sedumi.ie.lehigh.edu/) and [Yalmip](http://users.isy.liu.se/johanl/yalmip/) in the Matlab search path.

Usage
==
A simple usage example is included in example.m. A more sophisticated application is given in hamiltonian.m, which implements the Hamiltonian of a fermionic system in a 2D grid.

Acknowledgement
==
The code for momentum generation is based on the noncommutative example in Bermuja:

http://math.berkeley.edu/~philipp/Software/Moments
