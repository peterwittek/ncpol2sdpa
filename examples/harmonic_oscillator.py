# -*- coding: utf-8 -*-
"""
Exporting a Hamiltonian ground state problem to SDPA. The Hamiltonian
is of a simple harmonic oscillator. Bosonic systems reach the optimum
solution at relaxation level 1:

Navascués, M. García-Sáez, A. Acín, A. and Pironio, S. A paradox in bosonic
energy computations via semidefinite programming relaxations. New Journal of
Physics, 2013, 15, 023026.

Created on Fri May 10 09:45:11 2013

@author: Peter Wittek
"""
import time
from sympy.physics.quantum.dagger import Dagger
from ncpol2sdpa import generate_variables, bosonic_constraints, SdpRelaxation,\
                       write_to_sdpa

# Level of relaxation
level = 1

# Number of variables
N = 3

# Parameters for the Hamiltonian
hbar, omega = 1, 1

# Define ladder operators
a = generate_variables(N, name='a')

hamiltonian = 0
for i in range(N):
    hamiltonian += hbar * omega * (Dagger(a[i]) * a[i] + 0.5)

substitutions, equalities = bosonic_constraints(a)
inequalities = []

time0 = time.time()
# Obtain SDP relaxation
print("Obtaining SDP relaxation...")
sdpRelaxation = SdpRelaxation(a, verbose=1)
sdpRelaxation.get_relaxation(level, objective=hamiltonian,
                             equalities=equalities,
                             substitutions=substitutions,
                             removeequalities=True)
# Export relaxation to SDPA format
print("Writing to disk...")
write_to_sdpa(sdpRelaxation, 'harmonic_oscillator.dat-s')

print('%0.2f s' % ((time.time() - time0)))
