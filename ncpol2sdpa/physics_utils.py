# -*- coding: utf-8 -*-
"""
The module contains helper functions for physics applications.

Created on Fri May 16 14:27:47 2014

@author: Peter Wittek
"""
from sympy.physics.quantum.dagger import Dagger
from .nc_utils import generate_variables, flatten
from .sdpa_utils import solve_sdp
from .sdp_relaxation import SdpRelaxation

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
    """Return a set of constraints that define bosonic ladder
    operators.
    """
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
    """Return  a set of constraints that define fermionic ladder
    operators.
    """
    n_vars = len(a)
    monomial_substitutions = {}
    equalities = []
    inequalities = []
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

    for i in range(n_vars):
        inequalities.append(Dagger(a[i])*a[i])
        inequalities.append(1-Dagger(a[i])*a[i])
    return monomial_substitutions, equalities, inequalities

def pauli_constraints(X, Y, Z):
    monomial_substitutions = {}
    equalities = []
    n_vars = len(X)
    for r in range(n_vars):
        # They square to the identity
        monomial_substitutions[X[r]*X[r]] = 1
        monomial_substitutions[Y[r]*Y[r]] = 1
        monomial_substitutions[Z[r]*Z[r]] = 1

        # Anticommutation relations
        equalities.append(Y[r]*X[r]+X[r]*Y[r])
        equalities.append(Z[r]*X[r]+X[r]*Z[r])
        equalities.append(Z[r]*Y[r]+Y[r]*Z[r])
        # Commutation relations.
        '''
        equalities.append(X[r]*Y[r] - 1j*Z[r])
        equalities.append(X[r]*Z[r] + 1j*Y[r])
        equalities.append(Y[r]*Z[r] - 1j*X[r])
        '''
        # They commute between the sites
        for s in range(r+1,n_vars):
            monomial_substitutions[X[s]*X[r]] = X[r]*X[s]
            monomial_substitutions[Y[s]*Y[r]] = Y[r]*Y[s]
            monomial_substitutions[Y[s]*X[r]] = X[r]*Y[s]
            monomial_substitutions[Y[r]*X[s]] = X[s]*Y[r]
            monomial_substitutions[Z[s]*Z[r]] = Z[r]*Z[s]
            monomial_substitutions[Z[s]*X[r]] = X[r]*Z[s]
            monomial_substitutions[Z[r]*X[s]] = X[s]*Z[r]
            monomial_substitutions[Z[s]*Y[r]] = Y[r]*Z[s]
            monomial_substitutions[Z[r]*Y[s]] = Y[s]*Z[r]
    return monomial_substitutions, equalities

def generate_measurements(party, label):
    """Generate variables that behave like measurements.
    """
    measurements = []
    for i in range(len(party)):
        measurements.append(generate_variables(party[i]-1, hermitian=True,
                                               name=label + '%s' % i))
    return measurements

def projective_measurement_constraints(A, B):
    """Return a set of constraints that define projective
    measurements.
    """
    monomial_substitutions = {}
    for Mk in [M for M_list in [A, B] for M in M_list]:
        for Ei in Mk:
            for Ej in Mk:
                if Ei != Ej:
                    # They are orthogonal in each M_k
                    monomial_substitutions[Ei * Ej] = 0
                    monomial_substitutions[Ej * Ei] = 0
                else:
                    # Every projector is idempotent
                    monomial_substitutions[Ei * Ei] = Ei

    # Projectors in A and B commute
    for Ei in [E for Mk in A for E in Mk]:
        for Ej in [F for Ml in B for F in Ml]:
            monomial_substitutions[Ej * Ei] = Ei * Ej

    return monomial_substitutions


def define_objective_with_I(I, A, B):
    objective = 0
    i, j = 0, 1  # Row and column index in I
    for m_Bj in B:  # Define first row
        for Bj in m_Bj:
            objective += I[i][j] * Bj
            j += 1
    i += 1
    for m_Ai in A:
        for Ai in m_Ai:
            objective += I[i][0] * Ai
            j = 1
            for m_Bj in B:
                for Bj in m_Bj:
                    objective += I[i][j] * Ai * Bj
                    j += 1
            i += 1
    return -objective

def correlator(A, B):
    """Correlators between the probabilities of two parties.
    """
    correlators = []
    for i in range(len(A)):
        correlator_row = []
        for j in range(len(B)):
            corr = 0
            for k in range(len(A[i])):
                for l in range(len(B[j])):
                    if k == l:
                        corr += A[i][k] * B[j][l]
                    else:
                        corr -= A[i][k] * B[j][l]
            correlator_row.append(corr)
        correlators.append(correlator_row)
    return correlators


def maximum_violation(A_configuration, B_configuration, I, level):
    """Get the maximum violation of a Bell inequality.
    """
    A = generate_measurements(A_configuration, 'A')
    B = generate_measurements(B_configuration, 'B')

    monomial_substitutions = projective_measurement_constraints(
        A, B)

    objective = define_objective_with_I(I, A, B)

    sdpRelaxation = SdpRelaxation(flatten([A, B]), verbose=2)
    sdpRelaxation.get_relaxation(objective, [], [],
                                 monomial_substitutions, level)
    return solve_sdp(sdpRelaxation)
