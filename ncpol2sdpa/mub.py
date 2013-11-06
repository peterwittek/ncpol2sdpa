# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:30:58 2013

@author: wittek
"""
from sympy.physics.quantum.operator import HermitianOperator
from sdprelaxation import SdpRelaxation

D = 6
K = 4

def idx(x, i):
    return x*D+i

def generateTriplets(P):
    triplets = []
    for x in range(K):
        for y in range(K):
            for z in range(K):
                if (x!=y and y!=z):
                    for i in range(D):
                        for j in range(D):
                            for k in range(D):
                                triplets.append(P[idx(x,i)]*P[idx(y,j)]*P[idx(z,k)])
    return triplets

P = [0]*D*K
for x in range(K):
    for i in range(D):
        P[idx(x,i)] = HermitianOperator('P(x_%s, %s)' % (i,x))

#Simple monomial substitutions
monomial_substitution = {}

#Idempotency and orthogonality
for x in range(K):
    for i in range(D):
        for j in range(i,D):
            if (j==i):
                monomial_substitution[P[idx(x,i)]**2] = P[idx(x,i)]
            else:
                monomial_substitution[P[idx(x,i)]*P[idx(x,j)]] = 0

equalities = []
equalities = [0] * (K + K*(K-1)*D**2)
n_eq = 0

for x in range(K):
    sum = 0
    for i in range(D):
        sum += P[idx(x,i)]
    equalities[n_eq] = sum - 1
    n_eq+=1

for x in range(K):
    for y in range(K):
        if (x!=y):
            for i in range(D):
                for j in range(D):
                    equalities[n_eq]=-P[idx(x,i)]/D+P[idx(x,i)]*P[idx(y,j)]*P[idx(x,i)]
                    #monomial_substitution[P[idx(x,i)]+D*P[idx(x,i)]*P[idx(y,j)]*P[idx(x,i)]]=P[idx(x,i)]/D
                    n_eq+=1

#No inequalities
inequalities = []

#No objective function
obj = 0

#Order of relaxation
order = 2

sdpRelaxation = SdpRelaxation(P)
sdpRelaxation.get_relaxation(obj, inequalities, equalities, 
                      monomial_substitution, order)
sdpRelaxation.write_to_sdpa('mub-D%s-K%s.dat-s' % (D,K) )                      
