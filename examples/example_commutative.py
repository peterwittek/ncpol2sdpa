# -*- coding: utf-8 -*-
"""
An example that exports to sparse SDPA format for scalable computation. The
description of the commutative example is in the following paper:

Pironio, S.; Navascués, M. & Acín, A. Convergent relaxations of
polynomial optimization problems with noncommuting variables SIAM Journal
on Optimization, SIAM, 2010, 20, 2157-2180.

Created on Fri May 10 09:45:11 2013

@author: Peter Wittek
"""

from ncpol2sdpa import generate_variables, SdpRelaxation

# Number of variables
n_vars = 2
# Order of relaxation
order = 2

# Get commutative variables
X = generate_variables(n_vars, commutative=True)

# Define the objective function
obj = X[0] * X[1] + X[1] * X[0]

# Inequality constraints
inequalities = [-X[1] ** 2 + X[1] + 0.5]

# Equality constraints
equalities = []

# Simple monomial substitutions
monomial_substitutions = {}
monomial_substitutions[X[0] ** 2] = X[0]

# Obtain SDP relaxation
sdpRelaxation = SdpRelaxation(X)
sdpRelaxation.get_relaxation(obj, inequalities, equalities,
                             monomial_substitutions, order)
sdpRelaxation.write_to_sdpa('example_commutative.dat-s')
