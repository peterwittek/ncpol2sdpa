# -*- coding: utf-8 -*-
"""
Created on Wed Nov  10 11:24:48 2015

@author: Peter Wittek
"""
from __future__ import division, print_function
from math import sqrt
from sympy import S
from sympy.physics.quantum.dagger import Dagger
import sys
from .sdp_relaxation import SdpRelaxation
from .nc_utils import apply_substitutions, is_number_type, \
    separate_scalar_factor, ncdegree


class RdmHierarchy(SdpRelaxation):

    """Class for obtaining a level in the reduced density matrix method
    :param variables: Commutative or noncommutative, Hermitian or nonhermiatian
                      variables, possibly a list of list of variables if the
                      hierarchy is not NPA.
    :type variables: list of :class:`sympy.physics.quantum.operator.Operator`
                     or
                     :class:`sympy.physics.quantum.operator.HermitianOperator`
                     or a list of list.
    :param parameters: Optional symbolic variables for which moments are not
                       generated.
    :type parameters: list of :class:`sympy.physics.quantum.operator.Operator`
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

    Attributes:
      - `monomial_sets`: The monomial sets that generate the moment matrix blocks.

      - `monomial_index`: Dictionary that maps monomials to SDP variables.

      - `constraints`: The complete set of constraints after preprocesssing.

      - `primal`: The primal optimal value.

      - `dual`: The dual optimal value.

      - `x_mat`: The primal solution matrix.

      - `y_mat`: The dual solution matrix.

      - `solution_time`: The amount of time taken to solve the relaxation.

      - `status`: The solution status of the relaxation.
    """

    def __init__(self, variables, parameters=None, verbose=0, circulant=False,
                 parallel=False):
        super(RdmHierarchy, self).__init__(variables, parameters, verbose,
                                           False, parallel)
        self.circulant = circulant
        self.correspondence = {}
        self.m_block = 0

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
                self.F[row_offset + rowA * N + columnA, ki] = coeffi
        '''
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
                        element, coeff = separate_scalar_factor(element)
                        remainder = monomial - coeff*element
                        if element not in self.correspondence:
                            self.correspondence[element] = expand((monomial0 -
                                                              remainder)/coeff)
            elif monomial != 0:
                monomial, coeff = separate_scalar_factor(monomial)
                if monomial not in self.correspondence:
                    self.correspondence[monomial] = monomial0/coeff
        '''
        if max(k) > n_vars:
            n_vars = max(k)
        return n_vars, k, coeff

    def __second_moments(self, n_vars, monomialsA, block_index,
                         processed_entries):
        N = len(monomialsA)
        row_offset = 0
        if block_index > 0:
            for block_size in self.block_struct[0:block_index]:
                row_offset += block_size ** 2
        coords, mons =  \
            generate_block_coords(monomialsA[:N // 2], monomialsA[:N // 2],
                                  0, N // 2, 0, N // 2, 0, 0, N // 2)
        coords_, mons_ = \
            generate_block_coords(monomialsA[:N // 2], monomialsA[N // 2:],
                                  N // 2, N, 0, N // 2, 0, 0, N)
        coords += coords_
        mons += mons_
        coords_, mons_ = \
            generate_block_coords(monomialsA[N // 2:], monomialsA[N // 2:],
                                  N // 2, N, N // 2, N, 0, 0, N)
        coords += coords_
        mons += mons_
        for mon, coord in zip(mons, coords):
            n_vars, _, _ = self._push_monomials(mon, n_vars, row_offset,
                                                coord, len(monomialsA))
        self.correspondence = {}
        return n_vars, block_index + 1, processed_entries

    def __fourth_moments(self, n_vars, monomialsA, block_index,
                         processed_entries):
        self.m_block += 1
        if self.m_block == 1 or self.m_block == 3:
            N = int(sqrt(len(monomialsA)))
            row_offset = 0
            if block_index > 0:
                for block_size in self.block_struct[0:block_index]:
                    row_offset += block_size ** 2
            for block_row in range(N):
                monsA = monomialsA[N*block_row:N*(block_row+1)]
                for block_col in range(block_row, N):
                    monsB = monomialsA[N*block_col:N*(block_col+1)]
                    coords, mons = \
                        generate_block_coords(monsA, monsB, 0, N, 0, N,
                                              N*block_row, N*block_col, N)
                    for mon, coord in zip(mons, coords):
                        n_vars, _, _ = self._push_monomials(mon, n_vars,
                                                            row_offset,
                                                            coord, N**2)
            self.correspondence = {}
            return n_vars, block_index + 1, processed_entries
        elif self.m_block == 2:
            N = int(sqrt(len(monomialsA)//2))
            row_offset = 0
            if block_index > 0:
                for block_size in self.block_struct[0:block_index]:
                    row_offset += block_size ** 2
            for block_row in range(N):
                monsA = monomialsA[N*block_row:N*(block_row+1)]
                for block_col in range(block_row, N):
                    monsB = monomialsA[N*block_col:N*(block_col+1)]
                    coords, mons = \
                        generate_block_coords(monsA, monsB, 0, N, 0, N,
                                              N*block_row, N*block_col, N)
                    for mon, coord in zip(mons, coords):
                        n_vars, _, _ = self._push_monomials(mon, n_vars,
                                                            row_offset,
                                                            coord, 2*N**2)

            for block_row in range(N):
                monsA = monomialsA[N*block_row:N*(block_row+1)]
                for block_col in range(N):
                    monsB = monomialsA[N**2+N*block_col:N**2+N*(block_col+1)]
                    coords, mons = \
                        generate_block_coords(monsA, monsB, 0, N, 0, N,
                                              N*block_row,
                                              N**2 + N*block_col, N)
                    for mon, coord in zip(mons, coords):
                        n_vars, _, _ = self._push_monomials(mon, n_vars,
                                                            row_offset,
                                                            coord, 2*N**2)

            for block_row in range(N):
                monsA = monomialsA[N**2+N*block_row:N**2+N*(block_row+1)]
                for block_col in range(block_row, N):
                    monsB = monomialsA[N**2+N*block_col:N**2+N*(block_col+1)]
                    coords, mons = \
                        generate_block_coords(monsA, monsB, 0, N, 0, N,
                                              N**2 + N*block_row,
                                              N**2 + N*block_col, N)
                    for mon, coord in zip(mons, coords):
                        n_vars, _, _ = self._push_monomials(mon, n_vars,
                                                            row_offset,
                                                            coord, 2*N**2)
            self.correspondence = {}
            return n_vars, block_index + 1, processed_entries
        else:
            return super(RdmHierarchy, self).\
                    _generate_moment_matrix(n_vars, block_index,
                                            processed_entries,
                                            monomialsA, [S.One])

    def __generate_semisymmetric_moment_matrix(self, n_vars, block_index,
                                               processed_entries, monomialsA,
                                               monomialsB):
        """Generate the moment matrix of monomials.

        Arguments:
        n_vars -- current number of variables
        block_index -- current block index in the SDP matrix
        monomials -- |W_d| set of words of length up to the relaxation level
        """
        row_offset = 0
        if block_index > 0:
            for block_size in self.block_struct[0:block_index]:
                row_offset += block_size ** 2
        N = len(monomialsA)
        for rowB in range(len(monomialsB)):
            for columnA in range(rowB, len(monomialsA)):
                processed_entries += 1
                monomial = monomialsB[rowB].adjoint() * \
                    monomialsA[columnA]
                n_vars = self._push_monomial(monomial, n_vars, row_offset,
                                             rowB, columnA, N, 0, 0, 1)
            if self.verbose > 0:
                percentage = \
                    "{0:.0f}%".format(float(processed_entries-1)/self.n_vars *
                                      100)
                sys.stdout.write("\r\x1b[KCurrent number of SDP variables: %d"
                                 " (done: %s)" % (n_vars, percentage))
                sys.stdout.flush()
        if self.verbose > 0:
            sys.stdout.write("\r")
        return n_vars, block_index + 1, processed_entries

    def _generate_moment_matrix(self, n_vars, block_index, processed_entries,
                                monomialsA, monomialsB, ppt=False):
        if self.circulant:
            if ncdegree(monomialsA[0]) == 1:
                return self.__second_moments(n_vars, monomialsA, block_index,
                                             processed_entries)
            elif ncdegree(monomialsA[0]) == 2:
                return self.__fourth_moments(n_vars, monomialsA, block_index,
                                             processed_entries)
            else:
                raise Exception("Cannot generate circulant moment matrix with "
                                "degree-" + str(ncdegree(monomialsA[0])) +
                                " terms.")

        else:
            if len(monomialsB) == 1 and monomialsB[0] == S.One:
                return super(RdmHierarchy, self).\
                        _generate_moment_matrix(n_vars, block_index,
                                                processed_entries,
                                                monomialsA, monomialsB)
            else:
                return \
                  self.__generate_semisymmetric_moment_matrix(n_vars,
                                                              block_index,
                                                              processed_entries,
                                                              monomialsA,
                                                              monomialsB)

    def _get_index_of_monomial(self, element, enablesubstitution=True,
                               daggered=False):
        """Returns the index of a monomial.
        """
        processed_element, coeff1 = separate_scalar_factor(element)
        if enablesubstitution:
            processed_element = \
                apply_substitutions(processed_element, self.substitutions,
                                    self.pure_substitution_rules)
        if processed_element.is_Number:
            return [(0, coeff1)]
        elif processed_element.is_Add:
            monomials = \
                processed_element.as_coeff_mul()[1][0].as_coeff_add()[1]
        else:
            monomials = [processed_element]
        result = []
        for monomial in monomials:
            monomial, coeff2 = separate_scalar_factor(monomial)
            coeff = coeff1*coeff2
            if monomial.is_Number:
                result.append((0, coeff))
                continue
            k = -1
            if monomial != 0:
                if monomial.as_coeff_Mul()[0] < 0:
                    monomial = -monomial
                    coeff = -1.0 * coeff
            if monomial in self.correspondence:
                match = self.correspondence[monomial]
                if is_number_type(match):
                    result.append((0, coeff))
                    continue
                elif match.is_Mul:
                    elements = [match]
                else:
                    elements = match.as_coeff_mul()[1][0].as_coeff_add()[1]
                for el in elements:
                    sub_result = self._get_index_of_monomial(el)
                for (ki, coeffi) in sub_result:
                    result.append((ki, coeffi*coeff))
                continue
            try:
                k = self.monomial_index[monomial]
                result.append((k, coeff))
            except KeyError:
                if not daggered:
                    dag_result = self._get_index_of_monomial(Dagger(monomial),
                                                             daggered=True)
                    result += [(k, coeff0*coeff) for k, coeff0 in dag_result]
                else:
                    raise RuntimeError("The requested monomial " +
                                       str(monomial) + " could not be found.")
        return result

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
            results = self._get_index_of_monomial(element)
            for (k, coeff) in results:
                if not isinstance(k, list):
                    k = [k]
                    for ki in k:
                        facvar[ki] += coeff
        return facvar


def generate_block_coords(monomialsA, monomialsB, col0, colN, row0, rowN,
                          row_offset, col_offset, row_length):
    coords, mons = [], []
    for column in range(col0, colN):
        coords.append([(row_offset + row, col_offset + column + row - row0)
                       for row in range(row0, rowN)
                       if column + row - row0 < row_length])
        mons.append([Dagger(monomialsA[row-row_offset-row0]) *
                     monomialsB[col-col_offset-col0]
                     for row, col in coords[-1]])
        if row_offset != col_offset or row0 < col0:
            lower_triangular = [(col - col0 - col_offset + row_offset + row0,
                                 row - row0 - row_offset + col_offset + col0)
                                for row, col in coords[-1]
                                if row - row0 - row_offset !=
                                col - col0 - col_offset]
            if lower_triangular != []:
                lower_mons = [Dagger(monomialsA[row-row_offset-row0]) *
                              monomialsB[col-col_offset-col0]
                              for row, col in lower_triangular]
                coords.append(lower_triangular)
                mons.append(lower_mons)
    return coords, mons
