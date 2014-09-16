# -*- coding: utf-8 -*-
"""
The module converts a noncommutative optimization problem provided in SymPy
format to an SDPA semidefinite programming problem.

Created on Sun May 26 15:06:17 2013

@author: Peter Wittek
"""
from math import floor
import numpy as np
from bisect import bisect_left
from scipy.linalg import qr
from scipy.sparse import lil_matrix, hstack
from sympy import S, Number
from sympy.physics.quantum.dagger import Dagger
from .nc_utils import apply_substitutions, build_monomial, get_ncmonomials, \
    pick_monomials_up_to_degree, ncdegree, unique, remove_scalar_factor, \
    separate_scalar_factor


class SdpRelaxation(object):

    """Class for obtaining sparse SDP relaxation.
    """

    def __init__(self, variables, verbose=0):
        self.monomial_substitutions = {}
        self.monomial_dictionary = {}
        self.n_vars = 0
        self.F_struct = None
        self.block_struct = []
        self.obj_facvar = 0
        self.variables = []
        self.n_monomials = 0
        self.verbose = verbose
        self.localization_order = []
        if isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = [variables]

    def __get_index_of_monomial(self, element, enableSubstitution=False):
        monomial, coeff = build_monomial(element)
        if enableSubstitution:
            monomial = apply_substitutions(
                monomial,
                self.monomial_substitutions)
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
                k = self.monomial_dictionary[monomial]
            except KeyError:
                try:
                    [monomial, coeff] = build_monomial(element)
                    monomial, scalar_factor = separate_scalar_factor(
                        apply_substitutions(Dagger(monomial),
                                            self.monomial_substitutions))
                    coeff *= scalar_factor
                    k = self.monomial_dictionary[monomial]
                except KeyError:
                    if self.verbose > 1:
                        [monomial, coeff] = build_monomial(element)
                        print monomial, coeff
                        print("DEBUG: %s, %s, %s" % (element,
                                                     Dagger(monomial),
                                                     apply_substitutions(
                                                         Dagger(monomial),
                                                     self.monomial_substitutions)))
        return k, coeff

    def __push_facvar_sparse(self, polynomial, block_index, i, j):
        """Calculate the sparse vector representation of a polynomial
        and pushes it to the F structure.
        """

        row_offset = 0
        for block_size in self.block_struct[0:block_index - 1]:
            row_offset += block_size ** 2
        width = self.block_struct[block_index - 1]
        # Preprocess the polynomial for uniform handling later
        # DO NOT EXPAND THE POLYNOMIAL HERE!!!!!!!!!!!!!!!!!!!
        # The __simplify_polynomial bypasses the problem.
        # Simplifying here will trigger a bug in SymPy related to
        # the powers of daggered variables.
        # polynomial = polynomial.expand()
        if polynomial == 0 or polynomial.is_Mul:
            elements = [polynomial]
        else:
            elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]
        # Identify its constituent monomials
        for element in elements:
            k, coeff = self.__get_index_of_monomial(element)
            # k identifies the mapped value of a word (monomial) w
            if k > -1 and coeff != 0:
                self.F_struct[row_offset + i * width + j, k] = coeff

    def __get_facvar(self, polynomial):
        """Return dense vector representation of a polynomial. This function is
        nearly identical to __push_facvar_sparse, but instead of pushing
        sparse entries to the constraint matrices, it returns a dense
        vector.
        """
        facvar = [0] * (self.n_vars + 1)
        # Preprocess the polynomial for uniform handling later
        if isinstance(polynomial, Number) or isinstance(polynomial, float) or isinstance(polynomial, int):
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

    def __process_monomial(self, monomial, n_vars):
        """Process a single monomial when building the moment matrix.
        """
        if monomial.as_coeff_Mul()[0] < 0:
            monomial = -monomial
        k = 0
        # Have we seen this monomial before?
        try:
            # If yes, then we improve sparsity by reusing the
            # previous variable to denote this entry in the matrix
            k = self.monomial_dictionary[monomial]
        except KeyError:
            # Otherwise we define a new entry in the associated
            # array recording the monomials, and add an entry in
            # the moment matrix
            k = n_vars + 1
            self.monomial_dictionary[monomial] = k
        return k

    def __generate_moment_matrix(self, monomials):
        """Generate the moment matrix of monomials.

        Arguments:
        monomials -- |W_d| set of words of length up to the relaxation order
        """
        n_vars = 0
        # We process the M_d(u,w) entries in the moment matrix
        for row in range(self.n_monomials):
            for column in range(row, self.n_monomials):
                # Calculate the monomial u*v
                monomial = Dagger(monomials[row]) * monomials[column]
                # Apply the substitutions if any
                monomial = apply_substitutions(monomial,
                                               self.monomial_substitutions)
                if monomial == 1:
                    self.F_struct[row * self.n_monomials + column, 0] = 1
                elif monomial != 0:
                    k = self.__process_monomial(monomial, n_vars)
                    if k > n_vars:
                        n_vars = k
                    # We push the entry to the moment matrix
                    self.F_struct[row * self.n_monomials + column, k] = 1
        # The initial estimate for the size of F_struct was overly
        # generous. We correct the size here.
        self.n_vars = n_vars
        self.F_struct = self.F_struct[:, 0:n_vars + 1]

    def __simplify_polynomial(self, polynomial):
        # Preprocess the polynomial for uniform handling later
        if isinstance(polynomial, int) or isinstance(polynomial, float):
            return polynomial
        polynomial = (1.0 * polynomial).expand()
        if isinstance(polynomial, Number):
            return polynomial
        if polynomial.is_Mul:
            elements = [polynomial]
        else:
            elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]
        new_polynomial = 0
        # Identify its constituent monomials
        for element in elements:
            monomial, coeff = build_monomial(element)
            monomial = apply_substitutions(
                monomial,
                self.monomial_substitutions)
            new_polynomial += coeff * monomial
        return new_polynomial

    def __process_inequalities(
            self, inequalities, all_monomials, block_index, order):
        """Generate localizing matrices

        Arguments:
        inequalities -- list of inequality constraints
        monomials    -- localizing monomials
        block_index -- the current block index in constraint matrices of the
                       SDP relaxation
        order -- the order of the relaxation
        """
        initial_block_index = block_index
        for ineq in inequalities:
            block_index += 1
            localization_order = self.localization_order[
                block_index - initial_block_index - 1]
            monomials = \
                pick_monomials_up_to_degree(all_monomials, localization_order)

            # Process M_y(gy)(u,w) entries
            for row in range(len(monomials)):
                for column in range(row, len(monomials)):
                    # Calculate the moments of polynomial entries
                    polynomial = \
                        self.__simplify_polynomial(
                            Dagger(monomials[row]) * ineq * monomials[column])
                    self.__push_facvar_sparse(polynomial,
                                              block_index, row, column)
        return block_index

    def __process_equalities(
            self, equalities, all_monomials, order):
        """Generate localizing matrices

        Arguments:
        equalities -- list of equality constraints
        monomials  -- localizing monomials
        order -- the order of the relaxation
        """
        max_localization_order = 0
        for eq in equalities:
            # Find the order of the localizing matrix
            eq_order = ncdegree(eq)
            if eq_order > 2 * order:
                print(
                    "An equality constraint has degree %d. Choose a higher level of relaxation." %
                    eq_order)
                raise Exception
            localization_order = int(floor((2 * order - eq_order) / 2))
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
                        self.__simplify_polynomial(Dagger(monomials[row]) *
                                                   equality * monomials[column])
                    A[n_rows] = self.__get_facvar(polynomial)
                    # This is something really weird: we want the constant
                    # terms in equalities to be positive. Otherwise funny
                    # things happen in the QR decomposition and the basis
                    # transformation.
                    if A[n_rows, 0] < 0:
                        A[n_rows] = -A[n_rows]
                    n_rows += 1
        return A

    def __calculate_block_structure(self, monomials, inequalities, order):
        self.block_struct.append(len(monomials))
        for ineq in inequalities:
            # Find the order of the localizing matrix
            ineq_order = ncdegree(ineq)
            if ineq_order > 2 * order:
                print(
                    "A constraint has degree %d. Choose a higher level of relaxation." %
                    ineq_order)
                raise Exception
            localization_order = int(floor((2 * order - ineq_order) / 2))
            self.localization_order.append(localization_order)
            localizing_monomials = \
                pick_monomials_up_to_degree(monomials, localization_order)
            self.block_struct.append(len(localizing_monomials))

    def get_relaxation(self, obj, inequalities, equalities,
                       monomial_substitutions, level,
                       removeequalities=False, monomials=None,
                       extramonomials=None):
        """Get the SDP relaxation of a noncommutative polynomial optimization
        problem.

        Arguments:
        obj -- the objective function
        inequalities -- list of inequality constraints
        equalities -- list of equality constraints
        monomial_substitutions -- monomials that can be replaced
                                  (e.g., idempotent variables)
        level -- the level of the relaxation
        removeequalities -- whether to attempt removing the equalities by 
                            solving the linear equations
        monomials -- instead of the level, monomials can be supplied directly
        extramonomials -- monomials to be included, on top of the requested
                          level of relaxation
        """
        self.monomial_substitutions = monomial_substitutions
        # Generate monomials and remove substituted ones
        if monomials == None:
            monomials = get_ncmonomials(self.variables, level)
        if extramonomials is not None:
            monomials.extend(extramonomials)
        monomials = [monomial for monomial in monomials if monomial not
                     in self.monomial_substitutions]
        monomials = [remove_scalar_factor(apply_substitutions(monomial,
                                          self.monomial_substitutions))
                     for monomial in monomials]
        monomials = unique(monomials)
        if not removeequalities:
            # Equalities are converted to pairs of inequalities
            for equality in equalities:
                inequalities.append(equality)
                inequalities.append(-equality)

        self.__calculate_block_structure(monomials, inequalities, level)
        # Initialize some helper variables, including the offsets of monomial
        # blocks if there is more than one.
        self.n_monomials = len(monomials)

        # The minus one compensates for the constant term in the
        # top left corner of the moment matrix
        self.n_vars = int(self.n_monomials * (self.n_monomials + 1) / 2) - 1
        rows_in_F_struct = 0
        for block_size in self.block_struct:
            rows_in_F_struct += block_size ** 2
        self.F_struct = lil_matrix((rows_in_F_struct,
                                    self.n_vars + 1))

        if self.verbose > 0:
            print('Number of SDP variables: %d' % self.n_vars)
            print('Generating moment matrix...')

        # Define top left Entry of the moment matrix, y_1 = 1

       # Generate moment matrices
        self.__generate_moment_matrix(monomials)
        if self.verbose > 0:
            print('Reduced number of SDP variables: %d' % self.n_vars)
        if self.verbose > 1:
            self.__save_monomial_dictionary("monomials.txt")

        # Objective function
        self.obj_facvar = (
            self.__get_facvar(self.__simplify_polynomial(obj)))[1:]
        # Process inequalities
        if self.verbose > 0:
            print('Processing %d inequalities...' % len(inequalities))

        self.__process_inequalities(inequalities, monomials, 1, level)
        if removeequalities:
            A = self.__process_equalities(equalities, monomials, level)
            self.__remove_equalities(equalities, A)

    def __build_permutation_matrix(self, P):
        n = len(P)
        E = lil_matrix((n, n))
        column = 0
        for row in P:
            E[row, column] = 1
            column += 1
        return E

    def __remove_equalities(self, equalities, A):
        if len(equalities) == 0:
            return
        c = np.array(self.obj_facvar)
        Q, R, P = qr(np.transpose(A[:, 1:]), pivoting=True)
        E = self.__build_permutation_matrix(P)
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
        self.n_vars = self.F_struct.shape[1] - 1

    def swap_objective(self, new_objective):
        """Swaps the objective function while keeping the moment matrix and
        the localizing matrices untouched"""
        self.obj_facvar = (
            self.__get_facvar(self.__simplify_polynomial(new_objective)))[1:]

    def __save_monomial_dictionary(self, filename):
        """Save the current monomial dictionary for debugging purposes.
        """
        monomial_translation = [''] * (self.n_vars + 1)
        for key, k in self.monomial_dictionary.iteritems():
            monomial = ('%s' % key)
            monomial = monomial.replace('Dagger(', '')
            monomial = monomial.replace(')', 'T')
            monomial = monomial.replace('**', '^')
            monomial_translation[k] = monomial
        f = open(filename, 'w')
        for k in range(len(monomial_translation)):
            f.write('%s %s\n' % (k, monomial_translation[k]))
        f.close()

    def __convert_row_to_SDPA_index(self, row_offsets, row):
        block_index = bisect_left(row_offsets[1:], row + 1)
        width = self.block_struct[block_index]
        row = row - row_offsets[block_index]
        i, j = divmod(row, width)
        return block_index, i, j

    def write_to_sdpa(self, filename):
        """Write the SDP relaxation to SDPA format.

        Arguments:
        filename -- the name of the file. It must have the suffix ".dat-s"
        """
        f = open(filename, 'w')
        f.write('"file ' + filename + ' generated by ncpol2sdpa"\n')
        f.write(str(self.n_vars) + ' = number of vars\n')
        f.write(str(len(self.block_struct)) + ' = number of blocs\n')
        # bloc structure
        f.write(str(self.block_struct).replace('[', '(').replace(']', ')'))
        f.write(' = BlocStructure\n')
        # c vector (objective)
        f.write(str(list(self.obj_facvar)).replace('[', '{').replace(']', '}'))
        f.write('\n')
        # Coefficient matrices
        cx = self.F_struct.tocoo()
        zipped = sorted(zip(cx.col, cx.row, cx.data))
        row_offsets = [0]
        cumulative_sum = 0
        for block_size in self.block_struct:
            cumulative_sum += block_size ** 2
            row_offsets.append(cumulative_sum)
        for k, row, v in zipped:
            block_index, i, j = self.__convert_row_to_SDPA_index(
                row_offsets, row)
            if k == 0:
                v *= -1
            f.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
                k, block_index + 1, i + 1, j + 1, v))
        f.close()
