# -*- coding: utf-8 -*-
"""
Exporting a Hamiltonian ground state problem to SDPA. The Hamiltonian 
is of a simple harmonic oscillator.

Created on Tue Dec  3 09:05:20 2013

@author: Peter Wittek
"""
import time
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.dagger import Dagger
from sdprelaxation import SdpRelaxation

# Order of relaxation
order = 2

# Parameters for the Hamiltonian
hbar, omega = 1, 1

# Define ladder operator
a = Operator('a')   # Annihilation

hamiltonian = hbar*omega*(Dagger(a)*a+0.5)
print hamiltonian

monomial_substitution = {}

# [a,aT]=1
equalities = []
equalities.append(a*Dagger(a)-Dagger(a)*a-1.0)

inequalities = []

time0 = time.time()
#Obtain SDP relaxation
print("Obtaining SDP relaxation...")
verbose = 1
sdpRelaxation = SdpRelaxation(a)
sdpRelaxation.get_relaxation(hamiltonian, inequalities, equalities, 
                      monomial_substitution, order, verbose)
#Export relaxation to SDPA format
print("Writing to disk...")
sdpRelaxation.write_to_sdpa('harmonic_oscillator.dat-s')                      

print('%0.2f s' % ((time.time()-time0)))
