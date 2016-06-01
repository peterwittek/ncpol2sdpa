# -*- coding: utf-8 -*-
"""
The module contains helper functions for physics applications.

Created on Fri May 16 14:27:47 2014

@author: Peter Wittek
"""
from __future__ import division, print_function
from sympy.core import S
from sympy.physics.quantum.dagger import Dagger
from .nc_utils import generate_operators, flatten
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
    elif periodic and width > 1:
        neighbors.append(index - width + 1)
    if coords[0] < lattice_length - 1:
        neighbors.append(index + width)
    elif periodic:
        neighbors.append(index - (lattice_length - 1) * width)
    return neighbors


def get_next_neighbors(indices, lattice_length, width=0, distance=1,
                       periodic=False):
    """Get the forward neighbors at a given distance of a site or set of sites
    in a lattice.

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
    if not isinstance(indices, list):
        indices = [indices]
    if distance == 1:
        return flatten(get_neighbors(index, lattice_length, width, periodic)
                       for index in indices)
    else:
        s1 = set(flatten(get_next_neighbors(get_neighbors(index,
                                                          lattice_length,
                                                          width, periodic),
                                            lattice_length, width, distance-1,
                                            periodic) for index in indices))
        s2 = set(get_next_neighbors(indices, lattice_length, width, distance-1,
                                    periodic))
        return list(s1 - s2)


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
            # substitutions[ai*Dagger(aj)] = -Dagger(ai)*aj
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
            # substitutions[ai*Dagger(aj)] = -Dagger(ai)*aj
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
        # equalities.append(X[i]*Y[i] - 1j*Z[i])
        # equalities.append(X[i]*Z[i] + 1j*Y[i])
        # equalities.append(Y[i]*Z[i] - 1j*X[i])
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
        measurements.append(generate_operators(label + '%s' % i, party[i] - 1,
                                               hermitian=True))
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
    # Idempotency and orthogonality of projectors
    if isinstance(parties[0][0][0], list):
        parties = parties[0]
    for party in parties:
        for measurement in party:
            for projector1 in measurement:
                for projector2 in measurement:
                    if projector1 == projector2:
                        substitutions[projector1**2] = projector1
                    else:
                        substitutions[projector1*projector2] = 0
                        substitutions[projector2*projector1] = 0
    # Projectors commute between parties in a partition
    for n1 in range(len(parties)):
        for n2 in range(n1+1, len(parties)):
            for measurement1 in parties[n1]:
                for measurement2 in parties[n2]:
                    for projector1 in measurement1:
                        for projector2 in measurement2:
                            substitutions[projector2*projector1] = \
                                projector1*projector2
    return substitutions


def define_objective_with_I(I, *args):
    """Define a polynomial using measurements and an I matrix describing a Bell
    inequality.

    :param I: The I matrix of a Bell inequality in the Collins-Gisin notation.
    :type I: list of list of int.
    :param args: Either the measurements of Alice and Bob or a `Probability`
                 class describing their measurement operators.
    :type A: tuple of list of list of
             :class:`sympy.physics.quantum.operator.HermitianOperator` or
             :class:`ncpol2sdpa.Probability`

    :returns: :class:`sympy.core.expr.Expr` -- the objective function to be
              solved as a minimization problem to find the maximum quantum
              violation. Note that the sign is flipped compared to the Bell
              inequality.
    """
    objective = I[0][0]
    if len(args) > 2 or len(args) == 0:
        raise Exception("Wrong number of arguments!")
    elif len(args) == 1:
        A = args[0].parties[0]
        B = args[0].parties[1]
    else:
        A = args[0]
        B = args[1]
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


def maximum_violation(A_configuration, B_configuration, I, level, extra=None):
    """Get the maximum violation of a two-party Bell inequality.

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
    P = Probability(A_configuration, B_configuration)
    objective = define_objective_with_I(I, P)
    if extra is None:
        extramonomials = []
    else:
        extramonomials = P.get_extra_monomials(extra)
    sdpRelaxation = SdpRelaxation(P.get_all_operators(), verbose=0)
    sdpRelaxation.get_relaxation(level, objective=objective,
                                 substitutions=P.substitutions,
                                 extramonomials=extramonomials)
    solve_sdp(sdpRelaxation)
    return sdpRelaxation.primal, sdpRelaxation.dual


class Probability(object):

    def __init__(self, *args, **kwargs):
        """Class for working with quantum probabilities.

        :param *args: Input configurations for each parties
        :type *args: tuple of list of lists
        :param labels: Optional parameter string to define the label of each
                       party.
        :type labels: list of str.
        :Example:

        For a CHSH scenario, instantiate the class as

            P = Probability([2, 2], [2, 2])

        """

        self.n_parties = len(args)
        self.parties = []
        self.labels = [chr(ord('A') + i) for i in range(self.n_parties)]
        for name, value in kwargs.items():
            if name == "labels":
                if len(value) != self.n_parties:
                    raise Exception("Incorrect number of labels!")
                else:
                    self.labels = value
            else:
                raise Exception("Unknown parameter " + name)
        for i, configuration in enumerate(args):
            self.parties.append(generate_measurements(configuration,
                                                      self.labels[i]))
        self.substitutions = projective_measurement_constraints(self.parties)

    def get_all_operators(self):
        """Return all operators across all parties and measurements to supply
        them to the `ncpol2sdpa.SdpRelaxation` class.

        """
        return flatten(self.parties)

    def _monomial_generator(self, monomials, label_indices):
        if label_indices == []:
            return monomials
        elif monomials == []:
            return self._monomial_generator(
                flatten(self.parties[label_indices[0]]), label_indices[1:])
        else:
            result = [m1*m2 for m1 in monomials
                      for m2 in flatten(self.parties[label_indices[0]])]
            return self._monomial_generator(result, label_indices[1:])

    def get_extra_monomials(self, *args):
        if len(args) == 0:
            return []
        if isinstance(args[0], list):
            args = args[0]
        extra_monomials = []
        for s in args:
            label_indices = [self.labels.index(party) for party in s]
            extra_monomials.extend(self._monomial_generator([], label_indices))
        return extra_monomials

    def _convert_marginal_index(self, marginal):
        if isinstance(marginal, str):
            return [self.labels.index(marginal)]
        else:
            return sorted([self.labels.index(m) if isinstance(m, str) else m
                           for m in marginal])

    def __call__(self, output_, input_, marginal=None):
        """Obtain your probabilities in the p(ab...|xy...) notation.

        :param output_: Conditional output as [a, b, ...]
        :type output_: list of ints.
        :param input_: The input to condition on as [x, y, ...]
        :type input_: list of ints.
        :param marginal: Optional parameter. If it is a marginal, then you can
                         define which party or parties it belongs to.
        :type marginal: list of str.
        :returns: polynomial of `sympy.physics.quantum.HermitianOperator`.

        :Example:

        For the CHSH scenario, to get p(10|01), write

            P([1,0], [0,1])

        To get the marginal p_A(0|1), write

            P([0], [1], ['A'])

        """

        if len(output_) != len(input_):
            raise Exception("The number of inputs does not match the number of"
                            "outputs!")
        elif len(input_) > self.n_parties:
            raise Exception("The number of inputs exceeds the number of "
                            "parties!")
        elif marginal is None and len(input_) < self.n_parties:
            raise Exception("Marginal requested, but without defining which!")
        elif marginal is None:
            marginal = self._convert_marginal_index(self.labels)
        else:
            marginal = self._convert_marginal_index(marginal)
            if len(marginal) != len(input_):
                raise Exception("The number of parties in the marginal does "
                                "not match the number of inputs!")
        result = S.One
        for party, (proj, meas) in enumerate(zip(output_, input_)):
            if len(self.parties[marginal[party]]) < meas + 1:
                raise Exception("Invalid measurement index " + str(meas) +
                                " for party " + self.labels[party])
            elif len(self.parties[marginal[party]][meas]) < proj:
                raise Exception("Invalid projection operator index " +
                                str(proj) + " for party " + self.labels[party])
            elif len(self.parties[marginal[party]][meas]) == proj:
                # We are in the Collins-Gisin picture: the last projector
                # is not part of the measurement.
                result *= S.One - \
                    sum(op for op in self.parties[marginal[party]][meas])
            else:
                result *= self.parties[marginal[party]][meas][proj]
        return result
