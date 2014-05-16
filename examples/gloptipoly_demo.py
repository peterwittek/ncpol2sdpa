# -*- coding: utf-8 -*-
"""
This script replicates the results of gloptipolydemo.m, which is packaged with
Gloptipoly3.

Created on Thu May 15 11:16:58 2014

@author: wittek
"""
from sympy.physics.quantum.operator import HermitianOperator
from ncpol2sdpa import SdpRelaxation

# Get commutative variables
x1 = HermitianOperator("x1")
x1.is_commutative = True
x2 = HermitianOperator("x2")
x2.is_commutative = True

g0 = 4 * x1 ** 2 + x1 * x2 - 4 * x2 ** 2 - \
    2.1 * x1 ** 4 + 4 * x2 ** 4 + x1 ** 6 / 3

# Obtain SDP relaxation
sdpRelaxation = SdpRelaxation([x1, x2])
sdpRelaxation.get_relaxation(g0, [], [], {}, 3)
sdpRelaxation.write_to_sdpa('gloptipoly_demo.dat-s')
