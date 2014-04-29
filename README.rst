Ncpol2sdpa
==========
Ncpol2sdpa is a set of scripts to convert a noncommutative polynomial optimization problem to a sparse semidefinite programming (SDP) problem that can be processed by the [SDPA](http://sdpa.sourceforge.net/) family of solvers. The optimization problem can be unconstrained or constrained by equalities and inequalities.

The objective is to be able to solve very large scale optimization problems. For example, a convergent series of lower bounds can be obtained for ground state problems with arbitrary Hamiltonians.

The implementation has an intuitive syntax for entering Hamiltonians and it scales for a larger number of noncommutative variables using a sparse representation of the SDP problem. 

Dependencies
============
The code requires `SymPy<http://sympy.org/>_>=0.7.2 in the Python search path. The code is known to work with Python 2.6.8 and 2.7.5, and also with Pypy 1.8 and 2.0.2. Using Pypy is highly recommended, as execution time is several times faster and memory use is reduced. The code is compatible with Python 3, but using Python 3.3.2 incurs a major decrease in performance; the case is likely to be similar in with other Python 3 versions.

Usage
=====
A simple usage example is included in examplencpol.py. A more sophisticated application is given in hamiltonian.py, which implements the Hamiltonian of a fermionic system in a 2D grid.

Installation
============
Follow the standard procedure for installing Python modules:

``$ sudo python_interpreter setup.py install``

If you install the module to your CPython library, but you want to use Pypy, please ensure that the PYTHONPATH variable is set up correctly, otherwise Pypy will not find the relevant modules.

Acknowledgment
==============
This work is supported by the European Commission Seventh Framework Programme under Grant Agreement Number FP7-601138 [PERICLES](http://pericles-project.eu/), by the [Red Española de Supercomputación](http://www.bsc.es/RES) grants number FI-2013-1-0008 and  FI-2013-3-0004, and by the [Swedish National Infrastructure for Computing](http://www.snic.se/) project number SNIC 2014/2-7.

More Information
================
For more information refer to the following manuscript:

[http://arxiv.org/abs/1308.6029](http://arxiv.org/abs/1308.6029)
