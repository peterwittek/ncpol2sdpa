# -*- coding: utf-8 -*-
"""
The module converts a noncommutative optimization problem provided in SymPy
format to an SDPA semidefinite programming problem.

Created on Sun May 26 15:06:17 2013

@author: Peter Wittek
"""
from __future__ import division, print_function
import os
from sympy import S, zeros
from sympy.physics.quantum.dagger import Dagger
import tempfile
from .nc_utils import apply_substitutions, is_number_type
from .sdp_relaxation import SdpRelaxation
from .sdpa_utils import write_to_sdpa


class SteeringHierarchy(SdpRelaxation):

    """Class for obtaining a step in the steering hierarchy.

    :param variables: Commutative or noncommutative, Hermitian or nonhermiatian
                      variables.
    :type variables: list of :class:`sympy.physics.quantum.operator.Operator`
                     or
                     :class:`sympy.physics.quantum.operator.HermitianOperator`
                     or a list of list.
    :param verbose: Optional parameter for level of verbosity:

                       * 0: quiet
                       * 1: verbose
                       * 2: debug level

    :type verbose: int.
    :param matrix_var_dim: Optional parameter to specify the size of matrix
                           variable blocks
    :type matrix_var_dim: int.
    :param mark_conjugate: Use this optional parameter to generate a symbolic
                           representation of the steering hierarchy for export.
    :type mark_conjugate: bool.

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
    hierarchy_types = ["npa", "npa_chordal", "moroder"]

    def __init__(self, variables, verbose=0, matrix_var_dim=None,
                 mark_conjugate=False, parallel=False):
        """Constructor for the class.
        """
        if matrix_var_dim is None and not mark_conjugate:
            raise Exception("Invalid steering hierarchy requested!")
        super(SteeringHierarchy, self).__init__(variables, verbose=verbose,
                                                parallel=parallel)
        if matrix_var_dim is not None:
            self.complex_matrix = True
            self.normalized = False
        self.matrix_var_dim = matrix_var_dim
        self.mark_conjugate = mark_conjugate

    def _process_monomial(self, monomial, n_vars):
        """Process a single monomial when building the moment matrix.
        """
        coeff, monomial = monomial.as_coeff_Mul()
        k = 0
        # Have we seen this monomial before?
        conjugate = False
        try:
            # If yes, then we improve sparsity by reusing the
            # previous variable to denote this entry in the matrix
            k = self.monomial_index[monomial]
        except KeyError:
            # An extra round of substitutions is granted on the conjugate of
            # the monomial if all the variables are Hermitian
            daggered_monomial = \
                apply_substitutions(Dagger(monomial), self.substitutions,
                                    self.pure_substitution_rules)
            try:
                k = self.monomial_index[daggered_monomial]
                conjugate = True
            except KeyError:
                # Otherwise we define a new entry in the associated
                # array recording the monomials, and add an entry in
                # the moment matrix
                k = n_vars + 1
                self.monomial_index[monomial] = k
        if conjugate:
            k = -k
        return k, coeff

    def _push_monomial(self, monomial, n_vars, row_offset, rowA, columnA, N,
                       rowB, columnB, lenB, prevent_substitutions=False):
        monomial = apply_substitutions(monomial, self.substitutions,
                                       self.pure_substitution_rules)
        if is_number_type(monomial):
            if rowA == 0 and columnA == 0 and rowB == 0 and columnB == 0 and \
                    monomial == 1.0 and not self.normalized:
                if self.matrix_var_dim is None:
                    n_vars += 1
                    self.F[row_offset + rowA * N*lenB + rowB * N +
                           columnA * lenB + columnB, n_vars] = 1
                else:
                    n_vars = self.__add_matrix_variable(row_offset, rowA,
                                                        columnA, N, rowB,
                                                        columnB, lenB,
                                                        n_vars + 1, False, 1)
            else:
                self.F[row_offset + rowA * N*lenB + rowB * N +
                       columnA * lenB + columnB, 0] = monomial
        elif monomial.is_Add:
            for element in monomial.as_ordered_terms():
                n_vars = self._push_monomial(element, n_vars, row_offset,
                                             rowA, columnA, N,
                                             rowB, columnB, lenB)
        elif monomial != 0:
            k, coeff = self._process_monomial(monomial, n_vars)
            # We push the entry to the moment matrix
            if self.matrix_var_dim is None:
                if self.mark_conjugate:
                    if k < 0:
                        coeff = -coeff
                        k = -k
                self.F[row_offset + rowA * N*lenB + rowB * N +
                       columnA * lenB + columnB, k] = coeff
            else:
                conjugate = False
                if k < 0:
                    conjugate = True
                    k = -k
                k = self.__add_matrix_variable(row_offset, rowA, columnA, N,
                                               rowB, columnB, lenB, k,
                                               conjugate, coeff)
            if k > n_vars:
                n_vars = k
        return n_vars

    def __add_matrix_variable(self, row_offset, rowA, columnA, N, rowB,
                              columnB, lenB, k, conjugate, coeff):
        if conjugate:
            imag = -1j
        else:
            imag = 1j
        for sub_row in range(self.matrix_var_dim):
            for sub_column in range(self.matrix_var_dim):
                if conjugate:
                    r, c = sub_column, sub_row
                else:
                    r, c = sub_row, sub_column
                block_row_index = rowA*self.matrix_var_dim + r
                block_column_index = columnA*self.matrix_var_dim + c
                if block_row_index > block_column_index:
                    k += 1
                    continue
                if block_row_index == block_column_index:
                    imag_zero = 0
                else:
                    imag_zero = 1
                value = coeff*(1+imag*imag_zero)
                self.F[row_offset +
                       block_row_index*self.matrix_var_dim*N*lenB +
                       rowB*self.matrix_var_dim*N + block_column_index*lenB +
                       self.matrix_var_dim*columnB, k] = value
                k += 1
        k -= 1
        return k

    def __get_trace_facvar(self, polynomial):
        """Return dense vector representation of a polynomial. This function is
        nearly identical to __push_facvar_sparse, but instead of pushing
        sparse entries to the constraint matrices, it returns a dense
        vector.
        """
        facvar = [0] * (self.n_vars + 1)
        F = {}
        for i in range(self.matrix_var_dim):
            for j in range(self.matrix_var_dim):
                for key, value in \
                        polynomial[i, j].as_coefficients_dict().items():
                    skey = apply_substitutions(key, self.substitutions,
                                               self.pure_substitution_rules)
                    try:
                        Fk = F[skey]
                    except KeyError:
                        Fk = zeros(self.matrix_var_dim, self.matrix_var_dim)
                    Fk[i, j] += value
                    F[skey] = Fk
        # This is the tracing part
        for key, Fk in F.items():
            if key == S.One:
                k = 1
            else:
                k = self.monomial_index[key]
            for i in range(self.matrix_var_dim):
                for j in range(self.matrix_var_dim):
                    sym_matrix = zeros(self.matrix_var_dim,
                                       self.matrix_var_dim)
                    sym_matrix[i, j] = 1
                    facvar[k+i*self.matrix_var_dim+j] = (sym_matrix*Fk).trace()
        facvar = [float(f) for f in facvar]
        return facvar

    def set_objective(self, objective, extraobjexpr=None):
        """Set or change the objective function of the polynomial optimization
        problem.

        :param objective: Describes the objective function.
        :type objective: :class:`sympy.core.expr.Expr`
        :param extraobjexpr: Optional parameter of a string expression of a
                             linear combination of moment matrix elements to be
                             included in the objective function
        :type extraobjexpr: str.
        """
        if objective is not None and self.matrix_var_dim is not None:
            facvar = self.__get_trace_facvar(objective)
            self.obj_facvar = facvar[1:]
            self.constant_term = facvar[0]
            if self.verbose > 0 and facvar[0] != 0:
                print("Warning: The objective function has a non-zero %s "
                      "constant term. It is not included in the SDP objective."
                      % facvar[0])
        else:
            super(SteeringHierarchy, self).\
              set_objective(objective, extraobjexpr=extraobjexpr)

    def _calculate_block_structure(self, inequalities, equalities,
                                   momentinequalities, momentequalities,
                                   extramomentmatrix, removeequalities,
                                   block_struct=None):
        """Calculates the block_struct array for the output file.
        """
        super(SteeringHierarchy, self).\
          _calculate_block_structure(inequalities, equalities,
                                     momentinequalities, momentequalities,
                                     extramomentmatrix, removeequalities)
        if self.matrix_var_dim is not None:
            self.block_struct = [self.matrix_var_dim*bs
                                 for bs in self.block_struct]

    def _estimate_n_vars(self):
        if self.matrix_var_dim is None:
            super(SteeringHierarchy, self)._estimate_n_vars()
        else:
            self.n_vars = 0
            for monomials in self.monomial_sets:
                n_monomials = len(monomials)
                self.n_vars += int(n_monomials * (n_monomials + 1) / 2) *\
                               self.matrix_var_dim**2

    def write_to_file(self, filename, filetype=None):
        """Write the relaxation to a file.

        :param filename: The name of the file to write to. The type can be
                         autodetected from the extension: .dat-s for SDPA,
                         .task for mosek, .csv for human readable format, or
                         .txt for a symbolic export
        :type filename: str.
        :param filetype: Optional parameter to define the filetype. It can be
                         "sdpa" for SDPA , "mosek" for Mosek, "csv" for
                         human readable format, or "txt" for a symbolic export.
        :type filetype: str.
        """
        if filetype == "txt" and not filename.endswith(".txt"):
            raise Exception("TXT files must have .txt extension!")
        elif filetype is None and filename.endswith(".txt"):
            filetype = "txt"
        else:
            return super(SteeringHierarchy, self).write_to_file(filename,
                                                                filetype=filetype)
        tempfile_ = tempfile.NamedTemporaryFile()
        tmp_filename = tempfile_.name
        tempfile_.close()
        tmp_dats_filename = tmp_filename + ".dat-s"
        write_to_sdpa(self, tmp_dats_filename)
        f = open(tmp_dats_filename, 'r')
        f.readline();f.readline();f.readline()
        blocks = ((f.readline().strip().split(" = ")[0])[1:-1]).split(", ")
        block_offset, matrix_size = [0], 0
        for block in blocks:
            matrix_size += abs(int(block))
            block_offset.append(matrix_size)
        f.readline()
        matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
        for line in f:
            entry = line.strip().split("\t")
            var, block = int(entry[0]), int(entry[1])-1
            row, column = int(entry[2]) - 1, int(entry[3]) - 1
            value = float(entry[4])
            offset = block_offset[block]
            matrix[offset+row][offset+column] = int(value*var)
            matrix[offset+column][offset+row] = int(value*var)
        f.close()
        f = open(filename, 'w')
        for matrix_line in matrix:
            f.write(str(matrix_line).replace('[', '').replace(']', '') + '\n')
        f.close()
        os.remove(tmp_dats_filename)
