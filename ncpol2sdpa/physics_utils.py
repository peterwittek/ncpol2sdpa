# -*- coding: utf-8 -*-
"""
The module contains helper functions for physics applications.

Created on Fri May 16 14:27:47 2014

@author: Peter Wittek
"""
from sympy.physics.quantum.dagger import Dagger


def get_neighbors(index, lattice_length, W=0, periodic=False):
    """Get the neighbors of an operator in a lattice.

    Arguments:
    index -- linear index of operator
    lattice_length -- the size of the 2D lattice in either dimension

    Returns a list of neighbors in linear index.
    """
    if W == 0:
        W = lattice_length
    neighbors = []
    coords = divmod(index, W)
    if coords[1] < W - 1:
        neighbors.append(index + 1)
    elif periodic:
        neighbors.append(index - W + 1)
    if coords[0] < lattice_length - 1:
        neighbors.append(index + W)
    elif periodic:
        neighbors.append(index - (lattice_length - 1) * W)
    return neighbors


def bosonic_constraints(a):
    n_vars = len(a)
    monomial_substitutions = {}
    equalities = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            # [a_i,a_jT] = 0 for i\neq j
            monomial_substitutions[Dagger(a[j]) * a[i]] = a[i] * Dagger(a[j])
            # [a_i, a_j] = 0
            monomial_substitutions[a[j] * a[i]] = a[i] * a[j]
            # [a_iT, a_jT] = 0
            monomial_substitutions[Dagger(a[j]) * Dagger(a[i])] = \
                Dagger(a[i]) * Dagger(a[j])

    # [a_i,a_iT]=1
    for i in range(n_vars):
        equalities.append(a[i] * Dagger(a[i]) - Dagger(a[i]) * a[i] - 1.0)

    return monomial_substitutions, equalities


def fermionic_constraints(a):
    n_vars = len(a)
    monomial_substitutions = {}
    equalities = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            # {a_i,a_jT} = 0 for i\neq j
            equalities.append(Dagger(a[j]) * a[i] + a[i] * Dagger(a[j]))
            equalities.append(a[j] * Dagger(a[i]) + Dagger(a[i]) * a[j])
            # {a_i, a_j} = 0
            equalities.append(a[j] * a[i] + a[i] * a[j])
            # {a_iT, a_jT} = 0
            equalities.append(Dagger(a[j]) * Dagger(a[i]) +
                              Dagger(a[i]) * Dagger(a[j]))

    # {a_i,a_iT} = 1
    for i in range(n_vars):
        equalities.append(a[i] ** 2)
        equalities.append(Dagger(a[i]) ** 2)
        equalities.append(a[i] * Dagger(a[i]) + Dagger(a[i]) * a[i] - 1.0)

    return monomial_substitutions, equalities

def projective_measurement_constraints(A, B):
    monomial_substitutions = {}
    equalities = []
    for Mk in [M for M_list in [A, B] for M in M_list]:
        sum = -1
        for Ei in Mk:
            sum += Ei
            for Ej in Mk:
                if Ei != Ej:
                    # They are orthogonal in each M_k
                    monomial_substitutions[Ei*Ej] = 0
                    monomial_substitutions[Ej*Ei] = 0
                else:
                    # Every projector is idempotent
                    monomial_substitutions[Ei*Ei] = Ei
        # Projectors add up to the identity in each M_k
        equalities.append(sum)
    
    # Projectors in A and B commute
    for Ei in [E for Mk in A for E in Mk]:
        for Ej in [F for Ml in B for F in Ml]:
            monomial_substitutions[Ej*Ei] = Ei*Ej
        
    return monomial_substitutions, equalities
    
