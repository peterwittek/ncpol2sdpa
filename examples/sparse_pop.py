# -*- coding: utf-8 -*-
"""
This example 18.1 from the following paper:

Kim, S. & Kojima, M. (2012). Exploiting Sparsity in SDP Relaxation of
Polynomial Optimization Problems. In Handbook on Semidefinite, Conic and
Polynomial Optimization. Springer, 2012, 499--531.

Created on Sun Nov 30 19:18:04 2014

@author: Peter Wittek
"""

from ncpol2sdpa import generate_variables, SdpRelaxation, solve_sdp

# Number of variables
n_vars = 3
# Level of relaxation
level = 2

# Get commutative variables
X = generate_variables(n_vars, commutative=True)

# Define the objective function
obj = X[1] - 2*X[0]*X[1] + X[1]*X[2]

# Inequality constraints
inequalities = [1-X[0]**2-X[1]**2, 1-X[1]**2-X[2]**2]

# Obtain SDP relaxation
sdpRelaxation = SdpRelaxation(X, hierarchy="npa_chordal")
sdpRelaxation.get_relaxation(level, objective=obj, inequalities=inequalities)
print(solve_sdp(sdpRelaxation))

