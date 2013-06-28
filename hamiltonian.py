# -*- coding: utf-8 -*-
"""
Exporting a Hamiltonian ground state problem to SDPA. The Hamiltonian 
is described in the following paper:

Corboz, P.; Evenbly, G.; Verstraete, F. & Vidal, G. (2009), 
Simulation of interacting fermions with entanglement renormalization.
arXiv:0904.4151

Created on Fri May 10 09:45:11 2013

@author: Peter Wittek
"""
import time
from sympy.core import S
from sympy.physics.quantum.operator import HermitianOperator
from sympy.physics.quantum.dagger import Dagger
from sdprelaxation import get_relaxation
from ncutils import get_neighbors

#Lattice dimension
lattice_dimension = 2
n_vars = lattice_dimension * lattice_dimension
#Order of relaxation
order = 2

# Parameters for the Hamiltonian
gam, lam = 1, 2

#Get Hermitian variables
C = [0]*n_vars
for i in xrange(n_vars):
    C[i] = HermitianOperator('C%s' % i)

hamiltonian = 0
for r in xrange(n_vars):
    hamiltonian -= 2*lam*Dagger(C[r])*C[r]
    neighbors = get_neighbors(r, lattice_dimension)
    for s in neighbors:
        hamiltonian += Dagger(C[r])*C[s]+Dagger(C[s])*C[r]
        hamiltonian -= gam*(Dagger(C[r])*Dagger(C[s])+C[s]*C[r])

monomial_substitution = {}
equalities = []
for r in xrange(n_vars):
    for s in xrange(n_vars):
        if not r == s:
            monomial_substitution[C[r] * C[s]] = -C[s] * C[r]
        else:
            equalities.append(C[r]*Dagger(C[s]) + Dagger(C[s])*C[r] - S.One)

inequalities = []

time0 = time.time()
#Obtain SDP relaxation
print "Obtaining SDP relaxation..."
prob = get_relaxation(C, hamiltonian, inequalities, equalities, 
                      monomial_substitution, order)
#Export relaxation to SDPA format
print "Writing to disk..."
prob.write_to_file('hamiltonian.dat-s')

print '%d %0.2f s' % (lattice_dimension, (time.time()-time0))
