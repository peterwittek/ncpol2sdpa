# -*- coding: utf-8 -*-
"""
This example calculates the maximum quantum violation of the CHSH inequality in
the probability picture with a mixed-level relaxation of 1+AB.

Created on Mon Dec  1 14:19:08 2014

@author: Peter Wittek
"""
from ncpol2sdpa import Probability, SdpRelaxation, define_objective_with_I

level = 1
I = [[ 0,   -1,    0 ],
     [-1,    1,    1 ],
     [ 0,    1,   -1 ]]
P = Probability([2, 2], [2, 2])
objective = define_objective_with_I(I, P)

sdpRelaxation = SdpRelaxation(P.get_all_operators())
sdpRelaxation.get_relaxation(level, objective=objective,
                             substitutions=P.substitutions,
                             extramonomials=P.get_extra_monomials('AB'))
sdpRelaxation.solve()
print(sdpRelaxation.primal, sdpRelaxation.status)
