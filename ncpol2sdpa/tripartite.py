# -*- coding: utf-8 -*-
"""
Script to generate the SDP relaxation of the bipartite separation problem of 
the following paper:

Bancal, J.-D.; Gisin, N.; Liang, Y.-C. & Pironio, S. Device-Independent 
Witnesses of Genuine Multipartite Entanglement. Physics Review Letters, 
2011, 106, 250404.

Created on Thu Oct 31 09:38:12 2013

@author: Peter Wittek and Jean-Daniel Bancal
"""
import csv, sys, warnings
from sympy.physics.quantum.operator import HermitianOperator
from sdprelaxation import SdpRelaxation

'''
Global variables
'''
parties = ['A', 'B', 'C']
N = len(parties) # Number of parties
M = 4 # Number of measuerment settings
K = 2 # Number of outcomes
partitions = ['BC|A', 'AC|B', 'AB|C']
S = len(partitions) # Number of possible partitions

'''
This function turns the parameters of a variable to a linear index
'''
def idx(n, m, k, s):
    return N*M*(K-1)*s + M*(K-1)*n + (K-1)*m + k
#    return N*M*(K-1)*s+N*M*k+N*m+n

'''
This function returns the index in the probability vector corresponding to a 
given choice of inputs and output for each party.

The convention is Prob(idxP([x,y,z],[a,b,c])) = P(a,b,c|x,y,z)
with x,y,z from 0 to M-1
 and a,b,c from 0 to K-2

To get the marginal, use a negative input number for the remaining parties.
    e.g. P(a,c|x,z) is Prob(idxP([x,-1,z],[a,0,c]))
'''
def idxP(ms, ks):
    tmp = [0]*N
    for n in range(N):
        if ms[n]>=0:
            tmp[n] = 1+idx(0,ms[n],ks[n],0)
        else:
            tmp[n] = 0
    tot = 0
    for n in range(N):
        tot = tot + tmp[n]*((1+(K-1)*M)**n)
    return tot

'''
The monomial basis can be greatly simplified by exploiting the relations 
among the projectors. This function defines all such relations. The input
parameter A is the set of noncommuting variables.
'''
def generate_monomial_substitutions(A):
    #Simple monomial substitutions
    monomial_substitution = {}
    
    #Idempotency and orthogonality of projectors
    for s in range(S):
        for n in range(N):
            for m in range(M):
                for k1 in range(K-1):
                    for k2 in range(K-1):
                            if (k1==k2):
                                monomial_substitution[A[idx(n,m,k1,s)]**2] = A[idx(n,m,k1,s)]
                            else:
                                monomial_substitution[A[idx(n,m,k1,s)]*A[idx(n,m,k2,s)]] = 0
    
    #Parties identification
    for s in range(S):
        for k1 in range(K-1):
            for k2 in range(K-1):
                for m1 in range(M):
                    for m2 in range(M):
                        for n1 in range(N):
                            for n2 in range(n1+1,N):
                                monomial_substitution[A[idx(n1,m1,k1,s)]*A[idx(n2,m2,k2,s)]] = A[idx(n2,m2,k2,s)]*A[idx(n1,m1,k1,s)]
    
    # Biseparation
    for k1 in range(K-1):
        for k2 in range(K-1):
            if k1!=k2:
                for m1 in range(M):
                    for m2 in range(M):
                        for n in range(N):
                            monomial_substitution[A[idx(n,m1,k1,n)]*A[idx(n,m2,k2,n)]]=A[idx(n,m2,k2,n)]*A[idx(n,m1,k1,n)]
    
    # Independence of algebras
    for s1 in range(S):
        for s2 in range(S):
            if s1!=s2:
                for k1 in range(K-1):
                    for k2 in range(K-1):
                        for m1 in range(M):
                            for m2 in range(M):
                                for n1 in range(N):
                                    for n2 in range(n1,N):
                                        if n1==n2:
                                            monomial_substitution[A[idx(n1,m1,k1,s1)]*A[idx(n1,m2,k2,s2)]]=0
                                        elif n2>n1:
                                            monomial_substitution[A[idx(n1,m1,k1,s1)]*A[idx(n2,m2,k2,s2)]]=0                            
                                            monomial_substitution[A[idx(n2,m1,k1,s1)]*A[idx(n1,m2,k2,s2)]]=0
    return monomial_substitution

'''
Some constraints cannot be expressed as monomial substitutions, but only as
equalities. Most notably, the correlation constraints are such. This functions 
generates the necessary equalities.
'''
def generate_equality_constraints(A, lamb, Prob):
    #equation constraints
    equalities = [0]*( (1+(K-1)*M)**N - 1 )
    counter = 0
    # Correlations reproduced
    for m1 in range(M):
        for k1 in range(K-1): # 1-partite marginals:
            equalities[idxP([m1,-1,-1], [k1,-1,-1])-1] = sum(A[idx(0,m1,k1,s)] for s in range(S)) - ( (1-lamb)*1/K + lamb*Prob[idxP([m1,-1,-1], [k1,-1,-1])] )
            equalities[idxP([-1,m1,-1], [-1,k1,-1])-1] = sum(A[idx(1,m1,k1,s)] for s in range(S)) - ( (1-lamb)*1/K + lamb*Prob[idxP([-1,m1,-1], [-1,k1,-1])] )
            equalities[idxP([-1,-1,m1], [-1,-1,k1])-1] = sum(A[idx(2,m1,k1,s)] for s in range(S)) - ( (1-lamb)*1/K + lamb*Prob[idxP([-1,-1,m1], [-1,-1,k1])] )
            counter+=3
            for m2 in range(M):
                for k2 in range(K-1): # 2-partite marginals:
                    equalities[idxP([m1,m2,-1], [k1,k2,-1])-1] = sum(A[idx(0,m1,k1,s)]*A[idx(1,m2,k2,s)] for s in range(S)) - ( (1-lamb)*1/(K**2) + lamb*Prob[idxP([m1,m2,-1], [k1,k2,-1])] )
                    equalities[idxP([m1,-1,m2], [k1,-1,k2])-1] = sum(A[idx(0,m1,k1,s)]*A[idx(2,m2,k2,s)] for s in range(S)) - ( (1-lamb)*1/(K**2) + lamb*Prob[idxP([m1,-1,m2], [k1,-1,k2])] )
                    equalities[idxP([-1,m1,m2], [-1,k1,k2])-1] = sum(A[idx(1,m1,k1,s)]*A[idx(2,m2,k2,s)] for s in range(S)) - ( (1-lamb)*1/(K**2) + lamb*Prob[idxP([-1,m1,m2], [-1,k1,k2])] )
                    counter+=3
                    for m3 in range(M):
                        for k3 in range(K-1): # joint probabilities:
                            equalities[idxP([m1,m2,m3], [k1,k2,k3])-1] = sum(A[idx(0,m1,k1,s)]*A[idx(1,m2,k2,s)]*A[idx(2,m3,k3,s)] for s in range(S)) - ( (1-lamb)*1/(K**3) + lamb*Prob[idxP([m1,m2,m3], [k1,k2,k3])] )
                            counter+=1
    return equalities

def main(argv=sys.argv):

    # Noncommuting variables
    A = [0]*(N*M*(K-1)*S+1)
    for n in range(N):
        for m in range(M):
            for k in range(K-1):
                for s in range(S):
                    A[idx(n,m,k,s)] = HermitianOperator('%s(%s,%s)%s' % (parties[n],m,k,partitions[s]))
 
    # Commuting, real-valued variable
    lambda_index = N*M*(K-1)*S
    A[lambda_index] = HermitianOperator('lambda')
    A[lambda_index].is_commutative=True
    
    # Obtain monomial substitutions to simplify the monomial basis     
    monomial_substitution = generate_monomial_substitutions(A)
    print('Total number of substitutions: %s' % len(monomial_substitution))

    with open('correlations.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            Prob = [ float(i) for i in row ]

    # Input check : the probabilities should be normalized, otherwise issue a warning:
    if Prob[0] != 1.0:
        warnings.warn('Warning: probabilities not normalized. They will be assumed to be normalized in the SDP.', UserWarning)

    # The probabilities enter the problem through equality constraints
    equalities = generate_equality_constraints(A, A[lambda_index], Prob)
    print('Total number of equality constraints: %s' % len(equalities))
    
    objective = -A[lambda_index]
    
    print('Objective function: %s' % objective)
    
    # There are no inequalities
    inequalities = []
    
    # Order of relaxation
    order = 1

    #Obtain SDP relaxation
    sdpRelaxation = SdpRelaxation(A)
    sdpRelaxation.get_relaxation(objective, inequalities, equalities, 
                          monomial_substitution, order)
    sdpRelaxation.write_to_sdpa('tripartite.dat-s')

    return 0

if __name__ == "__main__":
    main()
