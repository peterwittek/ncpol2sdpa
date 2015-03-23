# -*- coding: utf-8 -*-
"""
The module converts a noncommutative optimization problem provided in SymPy
format to an SDPA semidefinite programming problem.

Created on Sun May 26 15:06:17 2013

@author: Peter Wittek
"""
from __future__ import division, print_function
from math import floor, copysign
import numpy as np
from sympy import S, Number
from sympy.matrices import Matrix
from sympy.physics.quantum.dagger import Dagger
import sys
try:
    from scipy.linalg import qr
    from scipy.sparse import lil_matrix, hstack
except ImportError:
    from .sparse_utils import lil_matrix
from .nc_utils import apply_substitutions, build_monomial, \
    pick_monomials_up_to_degree, ncdegree, \
    separate_scalar_factor, flatten, build_permutation_matrix, \
    simplify_polynomial, get_monomials, unique
from .chordal_extension import generate_clique, find_clique_index
from .faacets_utils import get_faacets_moment_matrix, collinsgisin_to_faacets

class SdpRelaxation(object):

    """Class for obtaining sparse SDP relaxation.

    :param variables: Commutative or noncommutative, Hermitian or nonhermiatian
                      variables, possibly a list of list of variables if the
                      hierarchy is not NPA.
    :type variables: list of :class:`sympy.physics.quantum.operator.Operator` or
                     :class:`sympy.physics.quantum.operator.HermitianOperator`
                     or a list of list.
    :param nonrelaxed: Optional variables which are not to be relaxed.
    :type nonrelaxed: list of :class:`sympy.physics.quantum.operator.Operator` or
                     :class:`sympy.physics.quantum.operator.HermitianOperator`
                     or a list of list.
    :param verbose: Optional parameter for level of verbosity:

                       * 0: quiet
                       * 1: verbose
                       * 2: debug level
    :type verbose: int.
    :param hierarchy:  Optional parameter for defining the type of hierarchy
                       (default: "npa"):

                       * "npa": the standard NPA hierarchy (`doi:10.1137/090760155 <http://dx.doi.org/10.1137/090760155>`_). When the variables are commutative, this formulation is identical to the Lasserre hierarchy.
                       * "npa_chordal": chordal graph extension to improve sparsity (`doi:10.1137/050623802 <http://dx.doi.org/doi:10.1137/050623802>`_)
                       * "moroder": `doi:10.1103/PhysRevLett.111.030501 <http://dx.doi.org/10.1103/PhysRevLett.111.030501>`_
    :type hierarchy: str.
    :param normalized: Optional parameter for changing the normalization of
                       states over which the optimization happens. Turn it off
                       if further processing is done on the SDP matrix before
                       solving it.
    :type normalized: bool.
    :param ppt: Optional parameter for imposing the partial positivity
                constraint in the Moroder hierarchy.
    :type normalized: bool.
    """
    hierarchy_types = ["npa", "npa_chordal", "moroder"]

    def __init__(self, variables, nonrelaxed=None, verbose=0, hierarchy="npa",
                 normalized=True, ppt=False):
        """Constructor for the class.
        """

        self.substitutions = {}
        self.monomial_index = {}
        self.n_vars = 0
        self.var_offsets = [0]
        self.F_struct = None
        self.block_struct = []
        self.obj_facvar = 0
        self.variables = []
        self.verbose = verbose
        self.localization_order = []
        self.normalized = normalized
        self.constraint_starting_block = 0
        self.level = 0
        self.clique_set = []
        self.monomial_sets = []
        if hierarchy in self.hierarchy_types:
            self.hierarchy = hierarchy
        else:
            raise Exception('Not allowed hierarchy type:', hierarchy)
        self.ppt = ppt
        if hierarchy != "moroder" and ppt:
            raise Exception('PPT condition only makes sense with the Moroder \
                             hierarchy')
        if isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = [variables]
        self.is_hermitian_variables = True
        # Check hermicity of all variables
        for var in flatten(self.variables):
            if not var.is_hermitian:
                self.is_hermitian_variables = False
                break
        self.nonrelaxed = nonrelaxed

    ########################################################################
    # ROUTINES RELATED TO GENERATING THE MOMENT MATRICES                   #
    ########################################################################

    def __process_monomial(self, monomial, n_vars):
        """Process a single monomial when building the moment matrix.
        """
        coeff, monomial = monomial.as_coeff_Mul()
        k = 0
        # Have we seen this monomial before?
        try:
            # If yes, then we improve sparsity by reusing the
            # previous variable to denote this entry in the matrix
            k = self.monomial_index[monomial]
        except KeyError:
            # An extra round of substitutions is granted on the conjugate of
            # the monomial if all the variables are Hermitian
            need_new_variable = True
            if self.is_hermitian_variables and ncdegree(monomial) > 2:
                daggered_monomial = apply_substitutions(Dagger(monomial),
                                                        self.substitutions)
                try:
                    k = self.monomial_index[daggered_monomial]
                    need_new_variable = False
                except KeyError:
                    need_new_variable = True
            if need_new_variable:
                # Otherwise we define a new entry in the associated
                # array recording the monomials, and add an entry in
                # the moment matrix
                k = n_vars + 1
                self.monomial_index[monomial] = k
        return k, coeff

    def __push_monomial(self, monomial, n_vars, row_offset, rowA, columnA, N,
                        rowB, columnB, lenB):
        monomial = apply_substitutions(monomial, self.substitutions)
        if isinstance(monomial, Number) or isinstance(monomial, int) or isinstance(monomial, float):
            if rowA == 0 and columnA == 0 and rowB == 0 and columnB == 0 and \
              monomial == 1.0 and not self.normalized:
                n_vars += 1
                self.F_struct[row_offset + rowA * N*lenB +
                              rowB * N + columnA * lenB + columnB, n_vars] = 1
            else:
                self.F_struct[row_offset + rowA * N*lenB +
                              rowB * N + columnA * lenB + columnB, 0] = monomial
        elif monomial.is_Add:
            for element in monomial.as_ordered_terms():
                n_vars = self.__push_monomial(element, n_vars, row_offset,
                                              rowA, columnA, N,
                                              rowB, columnB, lenB)
        elif monomial != 0:
            k, coeff = self.__process_monomial(monomial, n_vars)
            if k > n_vars:
                n_vars = k
            # We push the entry to the moment matrix
            self.F_struct[row_offset + rowA * N*lenB +
                          rowB * N +
                          columnA * lenB + columnB, k] = coeff
        return n_vars

    def __generate_moment_matrix(self, n_vars, block_index, processed_entries,
                                 monomialsA, monomialsB):
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
        N = len(monomialsA)*len(monomialsB)
        # We process the M_d(u,w) entries in the moment matrix
        for rowA in range(len(monomialsA)):
            for columnA in range(rowA, len(monomialsA)):
                for rowB in range(len(monomialsB)):
                    start_columnB = 0
                    if rowA == columnA:
                        start_columnB = rowB
                    for columnB in range(start_columnB, len(monomialsB)):
                        processed_entries += 1
                        if (not self.ppt) or (columnB >= rowB):
                            monomial = Dagger(monomialsA[rowA]) * \
                                       monomialsA[columnA] * \
                                       Dagger(monomialsB[rowB]) * \
                                       monomialsB[columnB]
                        else:
                            monomial = Dagger(monomialsA[rowA]) * \
                                       monomialsA[columnA] * \
                                       Dagger(monomialsB[columnB]) * \
                                       monomialsB[rowB]
                        # Apply the substitutions if any
                        n_vars = self.__push_monomial(monomial, n_vars,
                                                      row_offset, rowA,
                                                      columnA, N, rowB,
                                                      columnB, len(monomialsB))
            if self.verbose > 0:
                percentage = "{0:.0f}%".format(
                  float(processed_entries-1)/self.n_vars * 100)
                sys.stdout.write("\r\x1b[KCurrent number of SDP variables: %d"\
                                 " (done: %s)" % (n_vars, percentage) )
                sys.stdout.flush()
        if self.verbose > 0:
            sys.stdout.write("\r")
        return n_vars, block_index + 1, processed_entries

    ########################################################################
    # ROUTINES RELATED TO GENERATING THE LOCALIZING MATRICES AND PROCESSING#
    # CONSTRAINTS                                                          #
    ########################################################################

    def __get_index_of_monomial(self, element, enablesubstitution=True):
        """Returns the index of a monomial.
        """
        monomial, coeff = build_monomial(element)
        if enablesubstitution:
            monomial = apply_substitutions(monomial,
                                           self.substitutions)
        # Given the monomial, we need its mapping L_y(w) to push it into
        # a corresponding constraint matrix
        if monomial != 0:
            if monomial.as_coeff_Mul()[0] < 0:
                monomial = -monomial
                coeff = -1.0 * coeff
        k = -1
        if monomial.is_Number:
            k = 0
        else:
            try:
                k = self.monomial_index[monomial]
            except KeyError:
                try:
                    [monomial, coeff] = build_monomial(element)
                    monomial, scalar_factor = separate_scalar_factor(
                        apply_substitutions(Dagger(monomial),
                                            self.substitutions))
                    coeff *= scalar_factor
                    k = self.monomial_index[monomial]
                except KeyError:
                    # An extra round of substitutions is granted on the
                    # conjugate of the monomial if all the variables are
                    #Hermitian
                    exists = False
                    if self.is_hermitian_variables:
                        daggered_monomial = \
                          apply_substitutions(Dagger(monomial),
                                              self.substitutions)
                        try:
                            k = self.monomial_index[daggered_monomial]
                            exists = True
                        except KeyError:
                            exists = False
                    if not exists and self.verbose > 0:
                        [monomial, coeff] = build_monomial(element)
                        sub = apply_substitutions(Dagger(monomial),
                                                  self.substitutions)
                        print(("DEBUG: %s, %s, %s" % (element,
                                                      Dagger(monomial), sub)))
        return k, coeff

    def __push_facvar_sparse(self, polynomial, block_index, row_offset, i, j):
        """Calculate the sparse vector representation of a polynomial
        and pushes it to the F structure.
        """
        width = self.block_struct[block_index - 1]
        # Preprocess the polynomial for uniform handling later
        # DO NOT EXPAND THE POLYNOMIAL HERE!!!!!!!!!!!!!!!!!!!
        # The simplify_polynomial bypasses the problem.
        # Simplifying here will trigger a bug in SymPy related to
        # the powers of daggered variables.
        # polynomial = polynomial.expand()
        if isinstance(polynomial, float) or polynomial == 0 or\
           polynomial.is_Mul:
            elements = [polynomial]
        else:
            elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]
        # Identify its constituent monomials
        for element in elements:
            k, coeff = self.__get_index_of_monomial(element)
            # k identifies the mapped value of a word (monomial) w
            if k > -1 and coeff != 0:
                self.F_struct[row_offset + i * width + j, k] += coeff

    def __get_facvar(self, polynomial):
        """Return dense vector representation of a polynomial. This function is
        nearly identical to __push_facvar_sparse, but instead of pushing
        sparse entries to the constraint matrices, it returns a dense
        vector.
        """
        facvar = [0] * (self.n_vars + 1)
        # Preprocess the polynomial for uniform handling later
        if isinstance(polynomial, Number) or isinstance(polynomial, float) or\
           isinstance(polynomial, int):
            facvar[0] = polynomial
            return facvar
        polynomial = polynomial.expand()
        if polynomial.is_Mul:
            elements = [polynomial]
        else:
            elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]
        for element in elements:
            k, coeff = self.__get_index_of_monomial(
                element)
            facvar[k] += coeff
        return facvar

    def __process_inequalities(
            self, inequalities, block_index):
        """Generate localizing matrices

        Arguments:
        inequalities -- list of inequality constraints
        monomials    -- localizing monomials
        block_index -- the current block index in constraint matrices of the
                       SDP relaxation
        """

        all_monomials = flatten(self.monomial_sets)
        initial_block_index = block_index
        row_offsets = [0]
        for block, block_size in enumerate(self.block_struct):
            row_offsets.append(row_offsets[block] + block_size ** 2)
        for k, ineq in enumerate(inequalities):
            block_index += 1
            if isinstance(ineq, str):
                self.__parse_expression(ineq, row_offsets[block_index-1])
                continue
            localization_order = self.localization_order[
                block_index - initial_block_index - 1]
            if self.hierarchy == "npa_chordal":
                index = find_clique_index(self.variables, ineq, self.clique_set)
                monomials = \
                 pick_monomials_up_to_degree(self.monomial_sets[index],
                                             localization_order)

            else:
                monomials = \
                  pick_monomials_up_to_degree(all_monomials,
                                              localization_order)
            monomials = unique(monomials)
            # Process M_y(gy)(u,w) entries
            for row in range(len(monomials)):
                for column in range(row, len(monomials)):
                    # Calculate the moments of polynomial entries
                    polynomial = \
                        simplify_polynomial(
                            Dagger(monomials[row]) * ineq * monomials[column],
                            self.substitutions)
                    self.__push_facvar_sparse(polynomial, block_index,
                                              row_offsets[block_index-1],
                                              row, column)
            if self.verbose > 0:
                sys.stdout.write("\r\x1b[KProcessing %d/%d constraints..." %
                                 (k+1, len(inequalities)))
                sys.stdout.flush()
        if self.verbose > 0:
            sys.stdout.write("\n")
        return block_index

    def __process_psd(self, psd, block_index):
        row_offsets = [0]
        for block, block_size in enumerate(self.block_struct):
            row_offsets.append(row_offsets[block] + block_size ** 2)
        for matrix in psd:
            for row in range(self.block_struct[block_index]):
                for column in range(row, self.block_struct[block_index]):
                    if isinstance(matrix, list):
                        polynomial = \
                          simplify_polynomial(matrix[row][column],
                                              self.substitutions)
                    elif isinstance(matrix, Matrix):
                        polynomial = \
                          simplify_polynomial(matrix[row, column],
                                              self.substitutions)
                    self.__push_facvar_sparse(polynomial, block_index+1,
                                              row_offsets[block_index],
                                              row, column)
            block_index += 1
        return block_index

    def __process_equalities(
            self, equalities, all_monomials):
        """Generate localizing matrices

        Arguments:
        equalities -- list of equality constraints
        monomials  -- localizing monomials
        level -- the level of the relaxation
        """
        max_localization_order = 0
        for equality in equalities:
            # Find the order of the localizing matrix
            eq_order = ncdegree(equality)
            if eq_order > 2 * self.level:
                print("An equality constraint has degree %d. Choose a "\
                      "higher level of relaxation." % eq_order)
                raise Exception
            localization_order = int(floor((2 * self.level - eq_order) / 2))
            if localization_order > max_localization_order:
                max_localization_order = localization_order
        monomials = \
            pick_monomials_up_to_degree(all_monomials, max_localization_order)
        A = np.zeros(
            (len(equalities) * len(monomials) * (len(monomials) + 1) / 2,
             self.n_vars + 1))
        n_rows = 0
        for equality in equalities:
            # Find the order of the localizing matrix
            # Process M_y(gy)(u,w) entries
            for row in range(len(monomials)):
                for column in range(row, len(monomials)):
                    # Calculate the moments of polynomial entries
                    polynomial = \
                        simplify_polynomial(Dagger(monomials[row]) *
                                            equality * monomials[column],
                                            self.substitutions)
                    A[n_rows] = self.__get_facvar(polynomial)
                    # This is something really weird: we want the constant
                    # terms in equalities to be positive. Otherwise funny
                    # things happen in the QR decomposition and the basis
                    # transformation.
                    if A[n_rows, 0] < 0:
                        A[n_rows] = -A[n_rows]
                    n_rows += 1
        return A

    def __remove_equalities(self, equalities, A):
        """Attempt to remove equalities by solving the linear equations.
        """
        if len(equalities) == 0:
            return
        c = np.array(self.obj_facvar)
        Q, R, P = qr(np.transpose(A[:, 1:]), pivoting=True)
        E = build_permutation_matrix(P)
        n = np.max(np.nonzero(np.sum(np.abs(R), axis=1) > 0)) + 1
        x = np.dot(Q[:, 0:n], np.linalg.solve(np.transpose(R[0:n, :]),
                                              E.T.dot(-A[:, 0])))
        x = np.append(1, x)
        H = lil_matrix(Q[:, n:])  # New basis
        # Transforming the objective function
        self.obj_facvar = H.T.dot(c)
        # Transforming the moment matrix and localizing matrices
        new_constant_column = lil_matrix(self.F_struct.dot(x))
        self.F_struct = hstack([new_constant_column.T,
                                self.F_struct[:, 1:].dot(H)])
        self.F_struct = self.F_struct.tolil()
        self.n_vars = self.F_struct.shape[1] - 1

    def __duplicate_momentmatrix(self, original_n_vars, n_vars, block_index):
        self.var_offsets.append(n_vars)
        row_offset = 0
        for block_size in self.block_struct[0:block_index]:
            row_offset += block_size ** 2
        width = self.block_struct[0]
        for row in range(width**2):
            self.F_struct[row_offset + row,
                          n_vars+1:n_vars + original_n_vars+2] =\
              self.F_struct[row, :original_n_vars+1]
        return n_vars + original_n_vars + 1, block_index + 1

    def __add_new_momentmatrix(self, n_vars, block_index):
        self.var_offsets.append(n_vars)
        row_offset = 0
        for block_size in self.block_struct[0:block_index]:
            row_offset += block_size ** 2
        width = self.block_struct[0]
        for i in range(width):
            for j in range(i, width):
                n_vars += 1
                self.F_struct[row_offset + i * width + j, n_vars] = 1
        return n_vars, block_index + 1

    def __impose_ppt(self, block_index):
        row_offset = 0
        for block_size in self.block_struct[0:block_index-1]:
            row_offset += block_size ** 2
        lenA = len(self.monomial_sets[0])
        lenB = len(self.monomial_sets[1])
        N = lenA*lenB
        for rowA in range(lenA):
            for columnA in range(rowA, lenA):
                for rowB in range(lenB):
                    start_columnB = 0
                    if rowA == columnA:
                        start_columnB = rowB
                    for columnB in range(start_columnB, rowB):
                        original_row = self.F_struct[row_offset + rowA*N*lenB +
                                                     rowB * N + columnA * lenB
                                                     + columnB]
                        self.F_struct[row_offset + rowA*N*lenB +
                                      rowB * N +
                                      columnA * lenB + columnB] = \
                                        self.F_struct[row_offset + rowA*N*lenB
                                                      + columnB * N
                                                      + columnA * lenB + rowB]
                        self.F_struct[row_offset + rowA*N*lenB + columnB * N +
                                      columnA * lenB + rowB] = original_row

    def __add_extra_momentmatrices(self, extramomentmatrices, n_vars,
                                   block_index):
        original_n_vars = n_vars
        if extramomentmatrices is not None:
            for parameters in extramomentmatrices:
                copy = False
                ppt = False
                for parameter in parameters:
                    if parameter == "copy":
                        copy = True
                    if parameter == "ppt":
                        ppt = True
                if copy:
                    n_vars, block_index = \
                      self.__duplicate_momentmatrix(original_n_vars, n_vars,
                                                    block_index)
                else:
                    n_vars, block_index = \
                      self.__add_new_momentmatrix(n_vars, block_index)
                if ppt:
                    self.__impose_ppt(block_index)
        return n_vars, block_index

    def __parse_expression(self, expr, row_offset):
        if expr.find("]") > -1:
            sub_exprs = expr.split(']')
            for sub_expr in sub_exprs:
                ind = sub_expr.find('[')
                if ind > -1:
                    idx = sub_expr[ind+1:].split(",")
                    i, j = int(idx[0]), int(idx[1])
                    mm_ind = int(sub_expr[ind-1:ind])
                    if sub_expr.find('*') > -1:
                        value = float(sub_expr[:sub_expr.find('*')])
                    elif sub_expr.startswith('-'):
                        value = -1.0
                    else:
                        value = 1.0
                    base_row_offset = sum([bs**2 for bs in
                                           self.block_struct[:mm_ind]])
                    width = self.block_struct[mm_ind]
                    self.F_struct[row_offset] += \
                      value*self.F_struct[base_row_offset + i*width + j]
                else:
                    value = float(sub_expr)
                    self.F_struct[row_offset, 0] += value

    ########################################################################
    # ROUTINES RELATED TO INITIALIZING DATA STRUCTURES                     #
    ########################################################################

    def __calculate_block_structure(self, inequalities, equalities, bounds,
                                    psd, extramomentmatrix, removeequalities):
        """Calculates the block_struct array for the output file.
        """
        self.block_struct = []
        if self.verbose > 0:
            print("Calculating block structure...")
        if self.nonrelaxed is not None:
            self.block_struct.append(-len(self.nonrelaxed))
        if  self.hierarchy == "moroder":
            self.block_struct.append(len(self.monomial_sets[0])*
                                     len(self.monomial_sets[1]))
        else:
            for monomials in self.monomial_sets:
                self.block_struct.append(len(monomials))
        if extramomentmatrix is not None:
            for _ in extramomentmatrix:
                if  self.hierarchy == "moroder":
                    self.block_struct.append(len(self.monomial_sets[0])*
                                             len(self.monomial_sets[1]))
                else:
                    for monomials in self.monomial_sets:
                        self.block_struct.append(len(monomials))
        if psd is not None:
            for matrix in psd:
                if isinstance(matrix, list):
                    self.block_struct.append(len(matrix))
                elif isinstance(matrix, Matrix):
                    self.block_struct.append(matrix.shape[0])
                else:
                    raise Exception("Unknown format for PSD constraint")
        degree_warning = False
        if inequalities is not None:
            n_inequalities = len(inequalities)
        else:
            n_inequalities = 0
        if removeequalities:
            constraints = enumerate(flatten([inequalities]))
        else:
            constraints = enumerate(flatten([inequalities, equalities]))
        for k, constraint in constraints:
            # Find the order of the localizing matrix
            if isinstance(constraint, str):
                ineq_order = 2 * self.level
            else:
                ineq_order = ncdegree(constraint)
            if ineq_order > 2 * self.level:
                degree_warning = True
            localization_order = int(floor((2 * self.level - ineq_order) / 2))
            self.localization_order.append(localization_order)
            if self.hierarchy == "npa_chordal":
                index = find_clique_index(self.variables, constraint,
                                          self.clique_set)
                localizing_monomials = \
                    pick_monomials_up_to_degree(self.monomial_sets[index],
                                                localization_order)
            else:
                localizing_monomials = \
                    pick_monomials_up_to_degree(flatten(self.monomial_sets),
                                                localization_order)
            if len(localizing_monomials) == 0:
                localizing_monomials = [1]
            self.block_struct.append(len(localizing_monomials))
            if k >= n_inequalities:
                self.localization_order.append(localization_order)
                self.block_struct.append(len(localizing_monomials))

        if degree_warning and self.verbose > 0:
            print("A constraint has degree %d. Either choose a higher level "\
                  "relaxation or ensure that a mixed-order relaxation has the"\
                  " necessary monomials" % (ineq_order))

        if bounds is not None:
            for _ in bounds:
                self.localization_order.append(0)
                self.block_struct.append(1)
        #self.block_struct.append(-(self.block_struct[0]**2))
        #self.block_struct.append(-2)

    def __generate_monomial_sets(self, objective, inequalities, equalities,
                                 extramonomials):
        if self.level == -1:
            if extramonomials == None:
                raise Exception("Cannot build relaxation at level -1 without \
                                monomials specified.")
            if isinstance(extramonomials[0], list):
                self.monomial_sets = extramonomials
            else:
                self.monomial_sets.append(extramonomials)
            return
        if isinstance(self.variables[0], list):
            k = 0
            for variables in self.variables:
                extramonomials_ = None
                if extramonomials is not None:
                    extramonomials_ = extramonomials[k]
                self.monomial_sets.append(
                  get_monomials(variables, extramonomials_, self.substitutions,
                                self.level,
                                removesubstitutions=self.is_hermitian_variables))
                k += 1
        elif self.hierarchy == "npa_chordal":
            self.clique_set = generate_clique(self.variables, objective,
                                              inequalities, equalities)
            if self.verbose > 1:
                print(self.clique_set)
            for clique in self.clique_set:
                variables = [self.variables[i] for i in np.nonzero(clique)[0]]
                self.monomial_sets.append(
                  get_monomials(variables, extramonomials, self.substitutions,
                                self.level,
                                removesubstitutions=self.is_hermitian_variables))
        else:
            self.monomial_sets.append(
              get_monomials(self.variables, extramonomials, self.substitutions,
                            self.level,
                            removesubstitutions=self.is_hermitian_variables))

    def __estimate_n_vars(self):
        self.n_vars = 0
        if self.nonrelaxed is not None:
            self.n_vars = len(self.nonrelaxed)
        if not self.hierarchy == "moroder":
            for monomials in self.monomial_sets:
                n_monomials = len(monomials)

                # The minus one compensates for the constant term in the
                # top left corner of the moment matrix
                self.n_vars += int(n_monomials * (n_monomials + 1) / 2)
                if self.hierarchy == "npa":
                    self.n_vars -= 1
        else:
            n_monomials = len(self.monomial_sets[0])*len(self.monomial_sets[1])
            self.n_vars += int(n_monomials * (n_monomials + 1) / 2)
            self.n_vars -= 1

    def __add_non_relaxed(self):
        new_n_vars, block_index = 0, 0
        if self.nonrelaxed is not None:
            block_index = 1
            for var in self.nonrelaxed:
                new_n_vars += 1
                self.monomial_index[var] = new_n_vars
                self.F_struct[new_n_vars - 1, new_n_vars] = 1
        return new_n_vars, block_index

    def __wipe_F_struct_from_constraints(self):
        row_offset = 0
        for block in range(self.constraint_starting_block):
            row_offset += self.block_struct[block]**2
        for row in range(row_offset, len(self.F_struct.rows)):
            self.F_struct.rows[row] = []
            self.F_struct.data[row] = []


    ########################################################################
    # PUBLIC ROUTINES EXPOSED TO THE USER                                  #
    ########################################################################


    def process_constraints(self, inequalities=None, equalities=None,
                            bounds=None, psd=None, block_index=0,
                            removeequalities=False):
        """Process the constraints and generate localizing matrices. Useful only
        if the moment matrix already exists. Call it if you want to replace your
        constraints. The number of the respective types of constraints and the
        maximum degree of each constraint must remain the same.

        :param inequalities: Optional parameter to list inequality constraints.
        :type inequalities: list of :class:`sympy.core.exp.Expr`.
        :param equalities: Optional parameter to list equality constraints.
        :type equalities: list of :class:`sympy.core.exp.Expr`.
        :type substitutions: dict of :class:`sympy.core.exp.Expr`.
        :param bounds: Optional parameter of bounds on variables which will not
                       be relaxed by localizing matrices.
        :type bounds: list of :class:`sympy.core.exp.Expr`.
        :param psd: Optional parameter of list of matrices that should be
                    positive semidefinite.
        :type psd: list of lists or of :class:`sympy.matrices.Matrix`.
        :param removeequalities: Optional parameter to attempt removing the
                                 equalities by solving the linear equations.
        :type removeequalities: bool.
        """
        if block_index == 0:
            block_index = self.constraint_starting_block
            self.__wipe_F_struct_from_constraints()
        if psd is not None:
            block_index = self.__process_psd(psd, block_index)
        constraints = flatten([inequalities])
        if not (removeequalities or equalities is None):
            # Equalities are converted to pairs of inequalities
            for equality in equalities:
                constraints.append(equality)
                constraints.append(-equality)
        if bounds is not None:
            for bound in bounds:
                constraints.append(bound)
        self.__process_inequalities(constraints, block_index)
        if removeequalities and equalities is not None:
            A = self.__process_equalities(equalities, flatten(self.monomial_sets))
            self.__remove_equalities(equalities, A)

    def get_faacets_relaxation(self, A_configuration, B_configuration, I):
        coefficients = collinsgisin_to_faacets(I)
        M, ncIndices = get_faacets_moment_matrix(A_configuration,
                                                 B_configuration, coefficients)
        self.n_vars = M.max() - 1
        bs = len(M) # The block size
        self.block_struct = [bs]
        self.F_struct = lil_matrix((bs**2, self.n_vars + 1))
        # Constructing the internal representation of the constraint matrices
        # See Section 2.1 in the SDPA manual and also Yalmip's internal
        # representation
        for i in range(bs):
            for j in range(i, bs):
                if M[i, j] != 0:
                    self.F_struct[i*bs+j, abs(M[i, j])-1] = copysign(1, M[i, j])
        self.obj_facvar = [0 for _ in range(self.n_vars)]
        for i in range(1, len(ncIndices)):
            self.obj_facvar[abs(ncIndices[i])-2] += copysign(1, ncIndices[i])*coefficients[i]

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
        if objective is not None:
            facvar = self.__get_facvar(simplify_polynomial(objective,
                                                           self.substitutions))
            self.obj_facvar = facvar[1:]

            if self.verbose > 0 and facvar[0] != 0:
                print("Warning: The objective function has a non-zero "\
                      "constant term. It is not included in the SDP objective.")
        else:
            self.obj_facvar = self.__get_facvar(0)
        if extraobjexpr is not None:
            for sub_expr in extraobjexpr.split(']'):
                ind = sub_expr.find('[')
                if ind > -1:
                    idx = sub_expr[ind+1:].split(",")
                    i, j = int(idx[0]), int(idx[1])
                    mm_ind = int(sub_expr[ind-1:ind])
                    if sub_expr.find('*') > -1:
                        value = float(sub_expr[:sub_expr.find('*')])
                    elif sub_expr.startswith('-'):
                        value = -1.0
                    else:
                        value = 1.0
                    base_row_offset = sum([bs**2 for bs in
                                           self.block_struct[:mm_ind]])
                    width = self.block_struct[mm_ind]
                    for column in self.F_struct[base_row_offset + i*width + j].rows[0]:
                        self.obj_facvar[column-1] = \
                          value*self.F_struct[base_row_offset + i*width + j, column]


    def get_relaxation(self, level, objective=None, inequalities=None,
                       equalities=None, substitutions=None, bounds=None,
                       psd=None, removeequalities=False, extramonomials=None,
                       extramomentmatrices=None, extraobjexpr=None):
        """Get the SDP relaxation of a noncommutative polynomial optimization
        problem.

        :param level: The level of the relaxation. The value -1 will skip
                      automatic monomial generation and use only the monomials
                      supplied by the option `extramonomials`.
        :type level: int.
        :param obj: Optional parameter to describe the objective function.
        :type obj: :class:`sympy.core.exp.Expr`.
        :param inequalities: Optional parameter to list inequality constraints.
        :type inequalities: list of :class:`sympy.core.exp.Expr`.
        :param equalities: Optional parameter to list equality constraints.
        :type equalities: list of :class:`sympy.core.exp.Expr`.
        :param substitutions: Optional parameter containing monomials that can
                              be replaced (e.g., idempotent variables).
        :type substitutions: dict of :class:`sympy.core.exp.Expr`.
        :param bounds: Optional parameter of bounds on variables which will not
                       be relaxed by localizing matrices.
        :type bounds: list of :class:`sympy.core.exp.Expr`.
        :param psd: Optional parameter of list of matrices that should be
                    positive semidefinite.
        :type psd: list of lists or of :class:`sympy.matrices.Matrix`.
        :param removeequalities: Optional parameter to attempt removing the
                                 equalities by solving the linear equations.
        :type removeequalities: bool.
        :param extramonomials: Optional paramter of monomials to be included, on
                               top of the requested level of relaxation.
        :type extramonomials: list of :class:`sympy.core.exp.Expr`.
        :param extramomentmatrices: Optional paramter of duplicating or adding
                               moment matrices.  A new moment matrix can be
                               unconstrained (""), a copy  of the first one
                               ("copy"), and satisfying a partial positivity
                               constraint ("ppt"). Each new moment matrix is
                               requested as a list of string of these options.
                               For instance, adding a single new moment matrix
                               as a copy of the first would be
                               ``extramomentmatrices=[["copy"]]``.
        :type extramomentmatrices: list of list of str.
        :param extraobjexpr: Optional parameter of a string expression of a
                             linear combination of moment matrix elements to be
                             included in the objective function.
        :type extraobjexpr: str.
        """
        if self.level < -1:
            raise Exception("Invalid level of relaxation")
        self.level = level
        if substitutions is None:
            self.substitutions = {}
        else:
            self.substitutions = substitutions
        # Generate monomials and remove substituted ones
        self.__generate_monomial_sets(objective, inequalities, equalities,
                                      extramonomials)
        # Figure out basic structure of the SDP
        self.__calculate_block_structure(inequalities, equalities, bounds, psd,
                                         extramomentmatrices, removeequalities)
        self.__estimate_n_vars()
        if extramomentmatrices is not None:
            for parameters in extramomentmatrices:
                copy = False
                for parameter in parameters:
                    if parameter == "copy":
                        copy = True
                if copy:
                    self.n_vars += self.n_vars + 1
                else:
                    self.n_vars += (self.block_struct[0]**2)/2
        self.F_struct = lil_matrix((sum([bs ** 2 for bs in self.block_struct]),
                                    self.n_vars + 1))

        if self.verbose > 0:
            print(('Estimated number of SDP variables: %d' % self.n_vars))
            print('Generating moment matrix...')

        # Generate moment matrices
        new_n_vars, block_index = self.__add_non_relaxed()
        processed_entries = 0
        if self.hierarchy == "moroder":
            new_n_vars, block_index, _ = \
                self.__generate_moment_matrix(new_n_vars, block_index,
                                              processed_entries,
                                              self.monomial_sets[0],
                                              self.monomial_sets[1])
        else:
            for monomials in self.monomial_sets:
                new_n_vars, block_index, processed_entries = \
                    self.__generate_moment_matrix(
                        new_n_vars,
                        block_index,
                        processed_entries,
                        monomials, [S.One])
                self.var_offsets.append(new_n_vars)
        if extramomentmatrices is not None:
            new_n_vars, block_index = \
            self.__add_extra_momentmatrices(extramomentmatrices, new_n_vars,
                                            block_index)
        # The initial estimate for the size of F_struct was overly generous.
        self.n_vars = new_n_vars
        # We don't correct the size of F_struct, because that would trigger
        # memory copies, and extra columns in lil_matrix are free anyway.
        # self.F_struct = self.F_struct[:, 0:self.n_vars + 1]

        if self.verbose > 0:
            print(('Reduced number of SDP variables: %d' % self.n_vars))
        # Objective function
        self.set_objective(objective, extraobjexpr)

        # Process constraints
        self.constraint_starting_block = block_index
        self.process_constraints(inequalities, equalities, bounds, psd,
                                 block_index, removeequalities)
