# -*- coding: utf-8 -*-
"""
The module contains helper functions for physics applications.

Created on Fri May 16 14:27:47 2014

@author: Peter Wittek
"""
from sympy.physics.quantum.dagger import Dagger
from .nc_utils import generate_variables, flatten
from .solver_common import solve_sdp
from .sdp_relaxation import SdpRelaxation


def get_neighbors(index, lattice_length, width=0, periodic=False):
    """Get the forward neighbors of a site in a lattice.

    :param index: Linear index of operator.
    :type index: int.
    :param lattice_length: The size of the 2D lattice in either dimension
    :type lattice_length: int.
    :param width: Optional parameter to define width.
    :type width: int.
    :param periodic: Optional parameter to indicate periodic boundary
                     conditions.
    :type periodic: bool

    :returns: list of int -- the neighbors in linear index.
    """
    if width == 0:
        width = lattice_length
    neighbors = []
    coords = divmod(index, width)
    if coords[1] < width - 1:
        neighbors.append(index + 1)
    elif periodic:
        neighbors.append(index - width + 1)
    if coords[0] < lattice_length - 1:
        neighbors.append(index + width)
    elif periodic:
        neighbors.append(index - (lattice_length - 1) * width)
    return neighbors


def get_next_neighbors(indices, lattice_length, width=0, distance=1, periodic=False):
    """Get the forward neighbors at a given distance of a site or set of sites in a lattice.

    :param index: Linear index of operator.
    :type index: int.
    :param lattice_length: The size of the 2D lattice in either dimension
    :type lattice_length: int.
    :param width: Optional parameter to define width.
    :type width: int.
    :param distance: Optional parameter to define distance.
    :type width: int.
    :param periodic: Optional parameter to indicate periodic boundary
                     conditions.
    :type periodic: bool

    :returns: list of int -- the neighbors at given distance in linear index.
    """
    if not isinstance(indices,list):
        indices = [indices]
    if distance == 1:
        return flatten(get_neighbors(index, lattice_length, width, periodic) for index in indices)
    else:
        return list(set(flatten(get_next_neighbors(get_neighbors(index, lattice_length, width, periodic), lattice_length, width, distance-1, periodic) for index in indices)) - set(get_next_neighbors(indices, lattice_length, width, distance-1, periodic)))
    

def bosonic_constraints(a):
    """Return  a set of constraints that define fermionic ladder operators.

    :param a: The non-Hermitian variables.
    :type a: list of :class:`sympy.physics.quantum.operator.Operator`.
    :returns: a dict of substitutions.
    """
    substitutions = {}
    for i, ai in enumerate(a):
        substitutions[ai * Dagger(ai)] = 1.0 + Dagger(ai) * ai
        for aj in a[i+1:]:
            #substitutions[ai*Dagger(aj)] = -Dagger(ai)*aj
            substitutions[ai*Dagger(aj)] = Dagger(aj)*ai
            substitutions[Dagger(ai)*aj] = aj*Dagger(ai)
            substitutions[ai*aj] = aj*ai
            substitutions[Dagger(ai) * Dagger(aj)] = Dagger(aj) * Dagger(ai)

    return substitutions

def fermionic_constraints(a):
    """Return  a set of constraints that define fermionic ladder operators.

    :param a: The non-Hermitian variables.
    :type a: list of :class:`sympy.physics.quantum.operator.Operator`.
    :returns: a dict of substitutions.
    """
    substitutions = {}
    for i, ai in enumerate(a):
        substitutions[ai ** 2] = 0
        substitutions[Dagger(ai) ** 2] = 0
        substitutions[ai * Dagger(ai)] = 1.0 - Dagger(ai) * ai
        for aj in a[i+1:]:
            #substitutions[ai*Dagger(aj)] = -Dagger(ai)*aj
            substitutions[ai*Dagger(aj)] = -Dagger(aj)*ai
            substitutions[Dagger(ai)*aj] = -aj*Dagger(ai)
            substitutions[ai*aj] = -aj*ai
            substitutions[Dagger(ai) * Dagger(aj)] = - Dagger(aj) * Dagger(ai)

    return substitutions

def pauli_constraints(X, Y, Z):
    """Return  a set of constraints that define Pauli spin operators.

    :param X: List of Pauli X operator on sites.
    :type X: list of :class:`sympy.physics.quantum.operator.HermitianOperator`.
    :param Y: List of Pauli Y operator on sites.
    :type Y: list of :class:`sympy.physics.quantum.operator.HermitianOperator`.
    :param Z: List of Pauli Z operator on sites.
    :type Z: list of :class:`sympy.physics.quantum.operator.HermitianOperator`.

    :returns: tuple of substitutions and equalities.
    """
    substitutions = {}
    n_vars = len(X)
    for i in range(n_vars):
        # They square to the identity
        substitutions[X[i] * X[i]] = 1
        substitutions[Y[i] * Y[i]] = 1
        substitutions[Z[i] * Z[i]] = 1

        # Anticommutation relations
        substitutions[Y[i] * X[i]] = - X[i] * Y[i]
        substitutions[Z[i] * X[i]] = - X[i] * Z[i]
        substitutions[Z[i] * Y[i]] = - Y[i] * Z[i]
        # Commutation relations.
        #equalities.append(X[i]*Y[i] - 1j*Z[i])
        #equalities.append(X[i]*Z[i] + 1j*Y[i])
        #equalities.append(Y[i]*Z[i] - 1j*X[i])
        # They commute between the sites
        for j in range(i + 1, n_vars):
            substitutions[X[j] * X[i]] = X[i] * X[j]
            substitutions[Y[j] * Y[i]] = Y[i] * Y[j]
            substitutions[Y[j] * X[i]] = X[i] * Y[j]
            substitutions[Y[i] * X[j]] = X[j] * Y[i]
            substitutions[Z[j] * Z[i]] = Z[i] * Z[j]
            substitutions[Z[j] * X[i]] = X[i] * Z[j]
            substitutions[Z[i] * X[j]] = X[j] * Z[i]
            substitutions[Z[j] * Y[i]] = Y[i] * Z[j]
            substitutions[Z[i] * Y[j]] = Y[j] * Z[i]
    return substitutions


def generate_measurements(party, label):
    """Generate variables that behave like measurements.

    :param party: The list of number of measurement outputs a party has.
    :type party: list of int.
    :param label: The label to be given to the symbolic variables.
    :type label: str.

    :returns: list of list of
             :class:`sympy.physics.quantum.operator.HermitianOperator`.
    """
    measurements = []
    for i in range(len(party)):
        measurements.append(generate_variables(party[i] - 1, hermitian=True,
                                               name=label + '%s' % i))
    return measurements


def projective_measurement_constraints(*parties):
    """Return a set of constraints that define projective measurements.

    :param parties: Measurements of different parties.
    :type A: list or tuple of list of list of
             :class:`sympy.physics.quantum.operator.HermitianOperator`.

    :returns: substitutions containing idempotency, orthogonality and
              commutation relations.
    """
    substitutions = {}
    #Idempotency and orthogonality of projectors
    for party in parties:
        for measurement in party:
            for projector1 in measurement:
                for projector2 in measurement:
                    if projector1 == projector2:
                        substitutions[projector1**2] = projector1
                    else:
                        substitutions[projector1*projector2] = 0
                        substitutions[projector2*projector1] = 0
    #Projectors commute between parties in a partition
    for n1 in range(len(parties)):
        for n2 in range(n1+1, len(parties)):
            for measurement1 in parties[n1]:
                for measurement2 in parties[n2]:
                    for projector1 in measurement1:
                        for projector2 in measurement2:
                            substitutions[projector2*projector1] = projector1*projector2
    return substitutions

def define_objective_with_I(I, A, B):
    """Define a polynomial using measurements and an I matrix describing a Bell
    inequality.

    :param I: The I matrix of a Bell inequality in the Collins-Gisin notation.
    :type I: list of list of int.
    :param A: Measurements of Alice.
    :type A: list of list of
             :class:`sympy.physics.quantum.operator.HermitianOperator`.
    :param B: Measurements of Bob.
    :type B: list of list of
             :class:`sympy.physics.quantum.operator.HermitianOperator`.

    :returns: :class:`sympy.core.expr.Expr` -- the objective function to be
              solved by SDPA as minimization problem to find the maximum quantum
              violation.
    """
    objective = I[0][0]
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

    :param A: Measurements of Alice.
    :type A: list of list of
             :class:`sympy.physics.quantum.operator.HermitianOperator`.
    :param B: Measurements of Bob.
    :type B: list of list of
             :class:`sympy.physics.quantum.operator.HermitianOperator`.

    :returns: list of correlators.
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

    :param A_configuration: Measurement settings of Alice.
    :type A_configuration: list of int.
    :param B_configuration: Measurement settings of Bob.
    :type B_configuration: list of int.
    :param I: The I matrix of a Bell inequality in the Collins-Gisin notation.
    :type I: list of list of int.
    :param level: Level of relaxation.
    :type level: int.

    :returns: tuple of primal and dual solutions of the SDP relaxation.
    """
    A = generate_measurements(A_configuration, 'A')
    B = generate_measurements(B_configuration, 'B')

    substitutions = projective_measurement_constraints(
        A, B)

    objective = define_objective_with_I(I, A, B)

    sdpRelaxation = SdpRelaxation(flatten([A, B]), verbose=0)
    sdpRelaxation.get_relaxation(level, objective=objective,
                                 substitutions=substitutions)
    primal, dual, _, _ = solve_sdp(sdpRelaxation)
    return primal, dual
