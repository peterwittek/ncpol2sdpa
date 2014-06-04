# -*- coding: utf-8 -*-
"""
A polynomial optimization problem of commutative variables. It is mentioned in
Section 5.12 of the following paper:

Henrion, D.; Lasserre, J. & Löfberg, J. GloptiPoly 3: moments, optimization and
semidefinite programming. Optimization Methods & Software, 2009, 24, 761-779

Created on Thu May 15 12:12:40 2014

@author: wittek
"""
import numpy as np
from sympy.simplify import simplify
from sympy.physics.quantum.operator import HermitianOperator
from ncpol2sdpa import SdpRelaxation

W = np.diag(np.ones(8), 1) + np.diag(np.ones(7), 2) + np.diag([1, 1], 7) + \
    np.diag([1], 8)
W = W + W.T
n = len(W)
e = np.ones(n)
Q = (np.diag(np.dot(e.T, W)) - W) / 4

x = []
equalities = []
for i in range(n):
    x.append(HermitianOperator('x%s' % i))
    x[i].is_commutative = True
    equalities.append(x[i] ** 2 - 1)

objective = simplify(np.dot(x, np.dot(Q, np.transpose(x))))

sdpRelaxation = SdpRelaxation(x)
sdpRelaxation.get_relaxation(objective, [], equalities, {}, 1,
                             removeequalities=True)
sdpRelaxation.write_to_sdpa('max_cut.dat-s')
