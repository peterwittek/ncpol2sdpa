# -*- coding: utf-8 -*-
"""
This example reproduces the Tsirelson bound for the CHSH inequality.

Created on Wed May 28 13:35:09 2014

@author: wittek
"""
from ncpol2sdpa import generate_variables, SdpRelaxation, \
                       projective_measurement_constraints


def expectation_values(measurement, outcomes):
    exp_values = []
    for k in range(len(measurement)):
        exp_value = 0
        for j in range(len(measurement[k])):
            exp_value += outcomes[k][j] * measurement[k][j]
        exp_values.append(exp_value)
    return exp_values

# Number of Hermitian variables
n_vars = 8
# Level of relaxation
level = 1

# Get Hermitian variables
E = generate_variables(n_vars, name='E', hermitian=True)

# Define measurements and outcomes
M, outcomes = [], []
for i in range(int(n_vars / 2)):
    M.append([E[2 * i], E[2 * i + 1]])
    outcomes.append([1, -1])

# Define which measurements Alice and Bob have
A = [M[0], M[1]]
B = [M[2], M[3]]

substitutions = projective_measurement_constraints(A, B)

C = expectation_values(M, outcomes)

chsh = -(C[0] * C[2] + C[0] * C[3] + C[1] * C[2] - C[1] * C[3])

sdpRelaxation = SdpRelaxation(E, verbose=2)
sdpRelaxation.get_relaxation(level, objective=chsh,
                             substitutions=substitutions)
sdpRelaxation.solve()
print(sdpRelaxation.primal, sdpRelaxation.status)
