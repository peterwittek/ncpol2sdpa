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


def get_neighbors(index, lattice_length, width=0, periodic=False):
    """Get the neighbors of an operator in a lattice.

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


def bosonic_constraints(a):
    """Return a set of constraints that define bosonic ladder operators.

    :param a: The non-Hermitian variables.
    :type a: list of :class:`sympy.physics.quantum.operator.Operator`.
    :returns: tuple of dict of substitutions and list of equalities.
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
    """Return  a set of constraints that define fermionic ladder operators.

    :param a: The non-Hermitian variables.
    :type a: list of :class:`sympy.physics.quantum.operator.Operator`.
    :returns: tuple of dict of substitutions and list of equalities and
              inequalities.
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
        inequalities.append(Dagger(a[i]) * a[i])
        inequalities.append(1 - Dagger(a[i]) * a[i])
    return monomial_substitutions, equalities, inequalities


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
    monomial_substitutions = {}
    equalities = []
    n_vars = len(X)
    for i in range(n_vars):
        # They square to the identity
        monomial_substitutions[X[i] * X[i]] = 1
        monomial_substitutions[Y[i] * Y[i]] = 1
        monomial_substitutions[Z[i] * Z[i]] = 1

        # Anticommutation relations
        equalities.append(Y[i] * X[i] + X[i] * Y[i])
        equalities.append(Z[i] * X[i] + X[i] * Z[i])
        equalities.append(Z[i] * Y[i] + Y[i] * Z[i])
        # Commutation relations.
        #equalities.append(X[i]*Y[i] - 1j*Z[i])
        #equalities.append(X[i]*Z[i] + 1j*Y[i])
        #equalities.append(Y[i]*Z[i] - 1j*X[i])
        # They commute between the sites
        for j in range(i + 1, n_vars):
            monomial_substitutions[X[j] * X[i]] = X[i] * X[j]
            monomial_substitutions[Y[j] * Y[i]] = Y[i] * Y[j]
            monomial_substitutions[Y[j] * X[i]] = X[i] * Y[j]
            monomial_substitutions[Y[i] * X[j]] = X[j] * Y[i]
            monomial_substitutions[Z[j] * Z[i]] = Z[i] * Z[j]
            monomial_substitutions[Z[j] * X[i]] = X[i] * Z[j]
            monomial_substitutions[Z[i] * X[j]] = X[j] * Z[i]
            monomial_substitutions[Z[j] * Y[i]] = Y[i] * Z[j]
            monomial_substitutions[Z[i] * Y[j]] = Y[j] * Z[i]
    return monomial_substitutions, equalities


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


def projective_measurement_constraints(A, B):
    """Return a set of constraints that define projective measurements.

    :param A: Measurements of Alice.
    :type A: list of list of
             :class:`sympy.physics.quantum.operator.HermitianOperator`.
    :param B: Measurements of Bob.
    :type B: list of list of
             :class:`sympy.physics.quantum.operator.HermitianOperator`.

    :returns: substitutions containing idempotency, orthogonality and
              commutation relations.
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
    return solve_sdp(sdpRelaxation)
