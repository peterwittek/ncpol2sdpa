# -*- coding: utf-8 -*-
"""
Created on Wed Nov  10 11:24:48 2015

@author: Peter Wittek
"""
from __future__ import division, print_function
from math import sqrt
from sympy import S
from sympy.physics.quantum.dagger import Dagger
from .sdp_relaxation import SdpRelaxation
from .nc_utils import apply_substitutions, is_number_type, ncdegree, \
    separate_scalar_factor


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
        self.m_block = 0

    def _push_monomials(self, monomials, n_vars, row_offset, coords, N):
        monomial0 = apply_substitutions(monomials[0], self.substitutions,
                                        self.pure_substitution_rules)
        for mon1 in monomials[1:]:
            self.moment_substitutions[mon1] = monomial0
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
            entries = self._process_monomial(monomial0, n_vars)
            for entry in entries:
                k1, coeff1 = entry
                k.append(k1)
                coeff.append(coeff1)
        for rowA, columnA in coords:
            for ki, coeffi in zip(k, coeff):
                self.F[row_offset + rowA * N + columnA, ki] = coeffi
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
            return n_vars, block_index + 1, processed_entries
        else:
            return super(RdmHierarchy, self).\
                    _generate_moment_matrix(n_vars, block_index,
                                            processed_entries,
                                            monomialsA, [S.One])

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
            return super(RdmHierarchy, self).\
                    _generate_moment_matrix(n_vars, block_index,
                                            processed_entries,
                                            monomialsA, monomialsB)


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
