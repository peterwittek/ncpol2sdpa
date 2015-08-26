# -*- coding: utf-8 -*-
"""
This script replicates the results of gloptipolydemo.m, which is packaged with
Gloptipoly3.

Created on Thu May 15 11:16:58 2014

@author: wittek
"""
from ncpol2sdpa import SdpRelaxation, write_to_sdpa, generate_variables

# Get commutative variables
x = generate_variables(2, commutative=True)

g0 = 4 * x[0] ** 2 + x[0] * x[1] - 4 * x[1] ** 2 - \
    2.1 * x[0] ** 4 + 4 * x[1] ** 4 + x[0] ** 6 / 3

# Obtain SDP relaxation
sdpRelaxation = SdpRelaxation(x)
sdpRelaxation.get_relaxation(3, objective=g0)
write_to_sdpa(sdpRelaxation, 'gloptipoly_demo.dat-s')
