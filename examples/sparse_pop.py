# -*- coding: utf-8 -*-
"""
This example 18.1 from the following paper:

Kim, S. & Kojima, M. (2012). Exploiting Sparsity in SDP Relaxation of
Polynomial Optimization Problems. In Handbook on Semidefinite, Conic and
Polynomial Optimization. Springer, 2012, 499--531.

Created on Sun Nov 30 19:18:04 2014

@author: Peter Wittek
"""

from ncpol2sdpa import generate_variables, SdpRelaxation

level = 2
X = generate_variables(3, commutative=True)
inequalities = [1-X[0]**2-X[1]**2 >= 0,
                1-X[1]**2-X[2]**2 >= 0]
sdpRelaxation = SdpRelaxation(X, verbose=2)
sdpRelaxation.get_relaxation(level, objective=X[1] - 2*X[0]*X[1] + X[1]*X[2],
                             inequalities=inequalities, chordal_extension=True)
sdpRelaxation.solve()
sdpRelaxation.write_to_file("/home/wittek/sparse2.csv")
print(sdpRelaxation.primal, sdpRelaxation.dual, sdpRelaxation.status)
