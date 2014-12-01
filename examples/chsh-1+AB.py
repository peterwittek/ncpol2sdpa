# -*- coding: utf-8 -*-
"""
This example calculates the maximum quantum violation of the CHSH inequality in
the probability picture with a mixed-level relaxation of 1+AB.

Created on Mon Dec  1 14:19:08 2014

@author: Peter Wittek
"""
from ncpol2sdpa import generate_measurements, \
                       projective_measurement_constraints, flatten, \
                       SdpRelaxation, define_objective_with_I, solve_sdp

level = 1
A_configuration = [2, 2]
B_configuration = [2, 2]
I = [[ 0,   -1,    0 ],
     [-1,    1,    1 ],
     [ 0,    1,   -1 ]]
A = generate_measurements(A_configuration, 'A')
B = generate_measurements(B_configuration, 'B')
monomial_substitutions = projective_measurement_constraints(
    A, B)
objective = define_objective_with_I(I, A, B)

AB = [Ai*Bj for Ai in flatten(A) for Bj in flatten(B)]

sdpRelaxation = SdpRelaxation(flatten([A, B]))
sdpRelaxation.get_relaxation(objective, [], [],
                             monomial_substitutions, level,
                             extramonomials=AB)
print solve_sdp(sdpRelaxation)
