# -*- coding: utf-8 -*-
"""
Script to calculate correlations for the SDP relaxation of the bipartite
separation problem of the following paper:

Bancal, J.-D.; Gisin, N.; Liang, Y.-C. & Pironio, S. Device-Independent 
Witnesses of Genuine Multipartite Entanglement. Physics Review Letters, 
2011, 106, 250404.

Created on Tue Nov 12 10:29:54 2013

@author: Jean-Daniel Bancal
"""
import csv, sys, warnings
from numpy import array, cos, dot, equal, kron, mod, random, real, reshape, sin, sqrt, zeros

'''
Global variables
'''
parties = ['A', 'B', 'C']
N = len(parties) # Number of parties
M = 4 # Number of measuerment settings
K = 2 # Number of outcomes

'''
This function computes the correlations expected when some quantum state is 
measured according to some settings.
'''
def correl_qubits(psi, sett):
    # Setting up context and checking input
    nbInputs = len(sett)/2./N
    
    if ~equal(mod(nbInputs,1), 0):
        warnings.warn('Warning: Bad input for correl_qubits.', UserWarning)
    else:
        nbInputs = int(nbInputs)
    
    # We initialize basic matrices
    id = array([[1,0],[0,1]])
    sx = array([[0,1],[1,0]])
    sy = array([[0,-1j],[1j,0]])
    sz = array([[1,0],[0,-1]])
    
    
    # Measurement operators definition
    c = cos(sett)
    s = sin(sett)
    
    A = zeros((1+nbInputs,2,2))+0j # Careful!!! Python matrices are either real or complex, need to add a null imaginary part to make sure the variable will accept complex entries...
    B = zeros((1+nbInputs,2,2))+0j
    C = zeros((1+nbInputs,2,2))+0j
    A[0,:,:] = id
    B[0,:,:] = id
    C[0,:,:] = id
    for i in range(nbInputs):
        A[1+i,:,:] = s[2*i]*c[2*i+1]*sx + s[2*i]*s[2*i+1]*sy + c[2*i]*sz
        B[1+i,:,:] = s[2*i+2*nbInputs]*c[2*i+2*nbInputs+1]*sx +         s[2*i+2*nbInputs]*s[2*i+2*nbInputs+1]*sy + c[2*i+2*nbInputs]*sz
        C[1+i,:,:] = s[2*i+4*nbInputs]*c[2*i+4*nbInputs+1]*sx +         s[2*i+4*nbInputs]*s[2*i+4*nbInputs+1]*sy + c[2*i+4*nbInputs]*sz
    
    # Now we compute the multipartite operators.
    operators = kron(kron(A,B),C)
    
    # And compute the corresponding expectation values.
    Prob = real(dot(psi.T, dot(operators, psi))).squeeze()
    
    # Note that the kronecker product before incremented Charles' operator index
    # first, instead of Alice's. So we correct for this order change now.
    Prob = reshape(reshape(Prob, array([1,1,1])*(1+nbInputs)).transpose((2,1,0)), array([(1+(K-1)*M)**N]))
    return Prob

def main(argv=sys.argv):
    # Here we try the function with some states and measurements:
    psi = array([[1,0,0,0,0,0,0,1]]/sqrt(2)).T+0j # The GHZ state
    #psi = array([[0,1,1,0,1,0,0,0]]/sqrt(3)).T # The W state
    sett = random.rand(2*M*N)#array([1,2,3,4,5,6,7,8,9,10,11,12])# the settings
    #sett = array([pi/2, 0, pi/2, pi/4, pi/2, 0, pi/2, pi/4, pi/2, 0, pi/2, pi/4]) # Settings of the GHZ paradox
    Prob = correl_qubits(psi, sett)

    print Prob    
    
    with open('correlations.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(Prob)

if __name__ == "__main__":
    main()
