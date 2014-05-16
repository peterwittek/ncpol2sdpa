# -*- coding: utf-8 -*-
"""
Exporting a Hamiltonian ground state problem to SDPA. The Hamiltonian
is of a simple harmonic oscillator. Bosonic systems reach the optimum
solution at relaxation order 1:

Navascués, M. García-Sáez, A. Acín, A. and Pironio, S. A paradox in bosonic
energy computations via semidefinite programming relaxations. New Journal of
Physics, 2013, 15, 023026.

Created on Fri May 10 09:45:11 2013

@author: Peter Wittek
"""
import time
from sympy.physics.quantum.dagger import Dagger
from ncpol2sdpa import generate_variables, bosonic_constraints, SdpRelaxation

# Order of relaxation
order = 2

# Number of variables
N = 3

# Parameters for the Hamiltonian
hbar, omega = 1, 1

# Define ladder operators
a = generate_variables(N, name='a')

hamiltonian = 0
for i in range(N):
    hamiltonian += hbar * omega * (Dagger(a[i]) * a[i] + 0.5)

monomial_substitutions, equalities = bosonic_constraints(a)
inequalities = []

time0 = time.time()
# Obtain SDP relaxation
print("Obtaining SDP relaxation...")
verbose = 1
sdpRelaxation = SdpRelaxation(a)
sdpRelaxation.get_relaxation(hamiltonian, inequalities, equalities,
                             monomial_substitutions, order, verbose)
# Export relaxation to SDPA format
print("Writing to disk...")
sdpRelaxation.write_to_sdpa('harmonic_oscillator.dat-s')

print('%0.2f s' % ((time.time() - time0)))
