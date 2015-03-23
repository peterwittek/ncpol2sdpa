# -*- coding: utf-8 -*-
"""
This example calculates the maximum quantum violation of the elegant Bell
inequality presented in the following paper:

Gisin, N. Bell Inequalities: Many Questions, a Few Answers. Quantum Reality,
Relativistic Causality, and Closing the Epistemic Circle. Springer Netherlands,
2009, 73, 125-138.

Created on Thu May 29 20:12:39 2014

@author: Peter Wittek
"""
from ncpol2sdpa import maximum_violation

level = 2
A_configuration = [2, 2, 2]
B_configuration = [2, 2, 2, 2]
I = [[0, -1.5,  0.5,  0.5,  0.5],
     [0,    1,    1,   -1,   -1],
     [0,    1,   -1,    1,   -1],
     [0,    1,   -1,   -1,    1]]

print maximum_violation(A_configuration, B_configuration, I, level)
