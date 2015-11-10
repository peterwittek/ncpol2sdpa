# -*- coding: utf-8 -*-
"""
Created on Wed Nov  10 11:24:48 2015

@author: Peter Wittek
"""
from sympy import expand
from sympy.physics.quantum.dagger import Dagger
from .sdp_relaxation import SdpRelaxation
from .nc_utils import apply_substitutions, is_number_type, build_monomial


class RdmHierarchy(SdpRelaxation):

    """Class for obtaining a level in the reduced density matrix method
    :param variables: Commutative or noncommutative, Hermitian or nonhermiatian
                      variables, possibly a list of list of variables if the
                      hierarchy is not NPA.
    :type variables: list of :class:`sympy.physics.quantum.operator.Operator`
                     or
                     :class:`sympy.physics.quantum.operator.HermitianOperator`
                     or a list of list.
    :param nonrelaxed: Optional variables which are not to be relaxed.
    :type nonrelaxed: list of :class:`sympy.physics.quantum.operator.Operator`
                     or
                     :class:`sympy.physics.quantum.operator.HermitianOperator`
                     or a list of list.
    :param verbose: Optional parameter for level of verbosity:

                       * 0: quiet
                       * 1: verbose
                       * 2: debug level
    :type verbose: int.
    :param normalized: Optional parameter for changing the normalization of
                       states over which the optimization happens. Turn it off
                       if further processing is done on the SDP matrix before
                       solving it.
    :type normalized: bool.
    :param ppt: Optional parameter to impose a partial positivity constraint
                on the moment matrix.
    :type ppt: bool.
    """

    def __init__(self, variables, nonrelaxed=None, verbose=0, circulant=False):
        super(RdmHierarchy, self).__init__(variables, nonrelaxed, verbose,
                                           False)
        self.circulant = circulant
        self.correspondence = {}

    def _push_monomials(self, monomials, n_vars, row_offset, coords, N):
        monomial0 = apply_substitutions(monomials[0], self.substitutions,
                                        self.pure_substitution_rules)
        k, coeff = [], []
        if is_number_type(monomial0):
            coeff = [monomial0]
            k = [0]
        elif monomial0.is_Add:
            for element in monomial0.as_ordered_terms():
                n_vars, k1, coeff1 = self._push_monomials([element], n_vars,
                                                          row_offset, coords,
                                                          N)
                k += k1
                coeff += coeff1

        elif monomial0 != 0:
            k1, coeff1 = self._process_monomial(monomial0, n_vars)
            if isinstance(k1, list):
                k += k1
                coeff.append(coeff1)
            else:
                k.append(k1)
                coeff.append(coeff1)
        for rowA, columnA in coords:
            for ki, coeffi in zip(k, coeff):
                self.F_struct[row_offset + rowA * N + columnA, ki] = coeffi

        for monomial in monomials[1:]:
            monomial = apply_substitutions(monomial, self.substitutions,
                                           self.pure_substitution_rules)
            if is_number_type(monomial):
                continue
            elif monomial.is_Add:
                for element in monomial.as_ordered_terms():
                    if is_number_type(element):
                        continue
                    else:
                        element, coeff = build_monomial(element)
                        remainder = monomial - element
                        self.correspondence[element] = expand((monomial0 -
                                                              remainder)/coeff)
            elif monomial != 0:
                monomial, coeff = build_monomial(monomial)
                self.correspondence[monomial] = monomial0/coeff
        if max(k) > n_vars:
            n_vars = max(k)
        return n_vars, k, coeff

    def _generate_moment_matrix(self, n_vars, block_index, processed_entries,
                                monomialsA, monomialsB):
        if self.circulant:
            N = len(monomialsA)
            row_offset = 0
            if block_index > 0:
                for block_size in self.block_struct[0:block_index]:
                    row_offset += block_size ** 2
            coords, mons = [], []
            for column in range(N // 2):
                coords.append([(row, column + row)
                               for row in range(N // 2)
                               if column + row < N // 2])
                mons.append([Dagger(monomialsA[row]) * monomialsA[col]
                             for row, col in coords[-1]])
            for column in range(N // 2, N):
                coords.append([(row, column + row)
                               for row in range(N // 2)
                               if column + row < N])
                mons.append([Dagger(monomialsA[row]) * monomialsA[col]
                             for row, col in coords[-1]])
                coords.append([(row, column + row - N // 2)
                               for row in range(N // 2, N)
                               if column + row - N // 2 < N])
                mons.append([Dagger(monomialsA[row]) * monomialsA[col]
                             for row, col in coords[-1]])
            for mon, coord in zip(mons, coords):
                n_vars, _, _ = self._push_monomials(mon, n_vars, row_offset,
                                                    coord, len(monomialsA))
            return n_vars, block_index + 1, processed_entries
        else:
            return super(RdmHierarchy, self).\
                    _generate_moment_matrix(n_vars, block_index,
                                            processed_entries,
                                            monomialsA, monomialsB)

    def _get_facvar(self, polynomial):
        """Return dense vector representation of a polynomial. This function is
        nearly identical to __push_facvar_sparse, but instead of pushing
        sparse entries to the constraint matrices, it returns a dense
        vector.
        """
        facvar = [0] * (self.n_vars + 1)
        # Preprocess the polynomial for uniform handling later
        if is_number_type(polynomial):
            facvar[0] = polynomial
            return facvar
        polynomial = polynomial.expand()
        if polynomial.is_Mul:
            elements = [polynomial]
        else:
            elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]
        for element in elements:
            print("BEFORE", element)
            element, coeff0 = build_monomial(element)
            element = apply_substitutions(element, self.substitutions,
                                          self.pure_substitution_rules)
            if element != 0 and element.as_coeff_Mul()[0] < 0:
                    element = -element
                    coeff0 = -1.0 * coeff0
            print("AFTER", element)
            if element in self.correspondence:
                element = self.correspondence[element]
            if element.is_Mul or is_number_type(element):
                sub_elements = [element]
            else:
                sub_elements = element.as_coeff_mul()[1][0].as_coeff_add()[1]
            for sub_element in sub_elements:
                results = self._get_index_of_monomial(sub_element)
                for (k, coeff) in results:
                    if not isinstance(k, list):
                        k = [k]
                        for ki in k:
                            facvar[ki] += coeff * coeff0
        return facvar
