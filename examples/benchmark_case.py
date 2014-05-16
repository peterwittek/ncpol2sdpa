# -*- coding: utf-8 -*-
"""
An example that exports to sparse SDPA format for benchmarking.

Created on Fri May 10 09:45:11 2013

@author: Peter Wittek
"""
from ncpol2sdpa import generate_variables, SdpRelaxation
import time

# Number of Hermitian variables
n_vars = 2
# Order of relaxation
order = 1

# Get Hermitian variables
X = generate_variables(n_vars, hermitian=True)

# Define the objective function
obj = 0
for i in range(n_vars):
    for j in range(n_vars):
        obj += X[i] * X[j]

# Inequality constraints
inequalities = []

# Equality constraints
equalities = []
for i in range(n_vars):
    equalities.append(X[i] * X[i] - 1.0)

# Simple monomial substitutions
monomial_substitution = {}
for i in range(n_vars):
    for j in range(i + 1, n_vars):
        # [X_i, X_j] = 0
        monomial_substitution[X[i] * X[j]] = X[j] * X[i]

# Obtain SDP relaxation
time0 = time.time()
sdpRelaxation = SdpRelaxation(X)
sdpRelaxation.get_relaxation(obj, inequalities, equalities,
                             monomial_substitution, order)
sdpRelaxation.write_to_sdpa('benchmark.dat-s')
print('%0.2f s' % ((time.time() - time0)))
