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
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.dagger import Dagger
from ncpol2sdpa.sdprelaxation import SdpRelaxation

# Order of relaxation
order = 1

# Number of variables
N = 4

# Parameters for the Hamiltonian
hbar, omega = 1, 1

# Define ladder operators
a = []
for i in range(N):
    a.append(Operator('a%s' % i))   # Annihilation

hamiltonian = 0
for i in range(N):
    hamiltonian += hbar * omega * (Dagger(a[i]) * a[i] + 0.5)

monomial_substitution = {}

for i in range(N):
    for j in range(i + 1, N):
        # [a_i,a_jT] = 0 for i\neq j
        monomial_substitution[a[i] * Dagger(a[j])] = Dagger(a[j]) * a[i]
        # [a_i, a_j] = 0
        monomial_substitution[a[i] * a[j]] = a[j] * a[i]
        # [a_iT, a_jT] = 0
        monomial_substitution[
            Dagger(a[i]) * Dagger(a[j])] = Dagger(a[j]) * Dagger(a[i])

# [a_i,a_iT]=1
equalities = []
for i in range(N):
    equalities.append(a[i] * Dagger(a[i]) - Dagger(a[i]) * a[i] - 1.0)

inequalities = []

time0 = time.time()
# Obtain SDP relaxation
print("Obtaining SDP relaxation...")
verbose = 1
sdpRelaxation = SdpRelaxation(a)
sdpRelaxation.get_relaxation(hamiltonian, inequalities, equalities,
                             monomial_substitution, order, verbose)
# Export relaxation to SDPA format
print("Writing to disk...")
sdpRelaxation.write_to_sdpa('harmonic_oscillator.dat-s')

print('%0.2f s' % ((time.time() - time0)))
