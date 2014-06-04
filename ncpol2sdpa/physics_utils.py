# -*- coding: utf-8 -*-
"""
The module contains helper functions for physics applications.

Created on Fri May 16 14:27:47 2014

@author: Peter Wittek
"""
from sympy.physics.quantum.dagger import Dagger
from ncpol2sdpa import generate_variables, SdpRelaxation, solve_sdp


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
            # monomial_substitutions[Dagger(a[j]) * a[i]] = - a[i] *
            # Dagger(a[j])
            equalities.append(Dagger(a[j]) * a[i] + a[i] * Dagger(a[j]))
            # monomial_substitutions[a[j] * Dagger(a[i])] = - Dagger(a[i]) *
            # a[j]
            equalities.append(a[j] * Dagger(a[i]) + Dagger(a[i]) * a[j])
            # {a_i, a_j} = 0
            # monomial_substitutions[a[j] * a[i]] = - a[i] * a[j]
            equalities.append(a[j] * a[i] + a[i] * a[j])
            # {a_iT, a_jT} = 0
            # monomial_substitutions[Dagger(a[j]) * Dagger(a[i])] = -
            # Dagger(a[i]) * Dagger(a[j])
            equalities.append(Dagger(a[j]) * Dagger(a[i]) +
                              Dagger(a[i]) * Dagger(a[j]))

    # {a_i,a_iT} = 1
    for i in range(n_vars):
        # monomial_substitutions[a[i] ** 2] = 0
        equalities.append(a[i] ** 2)
        # monomial_substitutions[Dagger(a[i]) ** 2] = 0
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
                    monomial_substitutions[Ei * Ej] = 0
                    monomial_substitutions[Ej * Ei] = 0
                else:
                    # Every projector is idempotent
                    monomial_substitutions[Ei * Ei] = Ei
        # Projectors add up to the identity in each M_k
        equalities.append(sum)

    # Projectors in A and B commute
    for Ei in [E for Mk in A for E in Mk]:
        for Ej in [F for Ml in B for F in Ml]:
            monomial_substitutions[Ej * Ei] = Ei * Ej

    return monomial_substitutions, equalities


def flatten(lol):
    new_list = []
    for element in lol:
        if isinstance(element[0], list):
            element = flatten(element)
        new_list.extend(element)
    return new_list


def define_objective_with_I(I, A, B):
    objective = 0
    i, j = 0, 1  # Row and column index in I
    for m_Bj in B:  # Define first row
        for Bj in m_Bj[:-1]:
            objective += I[i][j] * Bj
            j += 1
    i += 1
    for m_Ai in A:
        for Ai in m_Ai[:-1]:
            objective += I[i][0] * Ai
            j = 1
            for m_Bj in B:
                for Bj in m_Bj[:-1]:
                        objective += I[i][j] * Ai * Bj
                        j += 1
            i += 1
    return -objective


def generate_measurements(party, label):
    measurements = []
    for i in range(len(party)):
        measurements.append(generate_variables(party[i], hermitian=True,
                                               name=label + '%s' % i))
    return measurements


def correlator(A, B):
    correlators = []
    for i in range(len(A)):
        correlator_row = []
        for j in range(len(B)):
            correlator = 0
            for k in range(len(A[i])):
                for l in range(len(B[j])):
                    if k == l:
                        correlator += A[i][k] * B[j][l]
                    else:
                        correlator -= A[i][k] * B[j][l]
            correlator_row.append(correlator)
        correlators.append(correlator_row)
    return correlators


def maximum_violation(A_configuration, B_configuration, I, level):
    A = generate_measurements(A_configuration, 'A')
    B = generate_measurements(B_configuration, 'B')

    monomial_substitutions, equalities = projective_measurement_constraints(
        A, B)

    objective = define_objective_with_I(I, A, B)

    sdpRelaxation = SdpRelaxation(flatten([A, B]), verbose=2)
    sdpRelaxation.get_relaxation(objective, [], equalities,
                                 monomial_substitutions, level)
    return solve_sdp(sdpRelaxation)
