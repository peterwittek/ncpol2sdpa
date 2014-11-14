# -*- coding: utf-8 -*-
"""
The module converts a noncommutative optimization problem provided in SymPy
format to an SDPA semidefinite programming problem.

Created on Sun May 26 15:06:17 2013

@author: Peter Wittek
"""
from math import floor
import numpy as np
from sympy import Number
from sympy.physics.quantum.dagger import Dagger
import sys
if sys.version.find("PyPy") == -1:
    from scipy.linalg import qr
    from scipy.sparse import lil_matrix, hstack
else:
    from .sparse_utils import lil_matrix
from .nc_utils import apply_substitutions, build_monomial, \
    pick_monomials_up_to_degree, ncdegree, \
    separate_scalar_factor, flatten, build_permutation_matrix, \
    simplify_polynomial, save_monomial_dictionary, get_monomials
from .sdpa_utils import convert_row_to_sdpa_index

class SdpRelaxation(object):

    """Class for obtaining sparse SDP relaxation.
    """

    def __init__(self, variables, verbose=0, independent_algebras=False):
        self.monomial_substitutions = {}
        self.monomial_dictionary = {}
        self.n_vars = 0
        self.F_struct = None
        self.block_struct = []
        self.obj_facvar = 0
        self.variables = []
        self.verbose = verbose
        self.localization_order = []
        self.independent_algebras = independent_algebras
        if isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = [variables]

    def __get_index_of_monomial(self, element, enablesubstitution=True):
        """Returns the index of a monomial.
        """
        monomial, coeff = build_monomial(element)
        if enablesubstitution:
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
                        sub = apply_substitutions(Dagger(monomial),
                                                  self.monomial_substitutions)
                        print(("DEBUG: %s, %s, %s" % (element,
                                                      Dagger(monomial), sub)))
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
        # The simplify_polynomial bypasses the problem.
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

    def __generate_moment_matrix(self, n_vars, block_index, monomials):
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
        # We process the M_d(u,w) entries in the moment matrix
        for row in range(len(monomials)):
            for column in range(row, len(monomials)):
                # Calculate the monomial u*v
                monomial = Dagger(monomials[row]) * monomials[column]
                # Apply the substitutions if any
                monomial = apply_substitutions(monomial,
                                               self.monomial_substitutions)
                if monomial == 1:
                    if not self.independent_algebras:
                        self.F_struct[row_offset + row * len(monomials) +
                                      column, 0] = 1
                    else:
                        k = n_vars + 1
                        n_vars = k
                        self.F_struct[row_offset + row * len(monomials) +
                                      column, k] = 1
                elif monomial != 0:
                    k = self.__process_monomial(monomial, n_vars)
                    if k > n_vars:
                        n_vars = k
                    # We push the entry to the moment matrix
                    self.F_struct[row_offset + row * len(monomials) +
                                  column, k] = 1
        return n_vars, block_index + 1

    def __process_inequalities(
            self, inequalities, all_monomials, block_index):
        """Generate localizing matrices

        Arguments:
        inequalities -- list of inequality constraints
        monomials    -- localizing monomials
        block_index -- the current block index in constraint matrices of the
                       SDP relaxation
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
                        simplify_polynomial(
                            Dagger(monomials[row]) * ineq * monomials[column],
                            self.monomial_substitutions)
                    self.__push_facvar_sparse(polynomial,
                                              block_index, row, column)
        return block_index

    def __process_equalities(
            self, equalities, all_monomials, level):
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
            if eq_order > 2 * level:
                print(("An equality constraint has degree %d. Choose a higher \
                      level of relaxation." % eq_order))
                raise Exception
            localization_order = int(floor((2 * level - eq_order) / 2))
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
                                            self.monomial_substitutions)
                    A[n_rows] = self.__get_facvar(polynomial)
                    # This is something really weird: we want the constant
                    # terms in equalities to be positive. Otherwise funny
                    # things happen in the QR decomposition and the basis
                    # transformation.
                    if A[n_rows, 0] < 0:
                        A[n_rows] = -A[n_rows]
                    n_rows += 1
        return A

    def __calculate_block_structure(self, monomial_sets, inequalities, level):
        """Calculates the block_struct array for the output file.
        """
        for monomials in monomial_sets:
            self.block_struct.append(len(monomials))
        for ineq in inequalities:
            # Find the order of the localizing matrix
            ineq_order = ncdegree(ineq)
            if ineq_order > 2 * level:
                print(("A constraint has degree %d. Choose a higher level of\
                      relaxation." % ineq_order))
                raise Exception
            localization_order = int(floor((2 * level - ineq_order) / 2))
            self.localization_order.append(localization_order)
            localizing_monomials = \
                pick_monomials_up_to_degree(flatten(monomial_sets),
                                            localization_order)
            self.block_struct.append(len(localizing_monomials))

    def get_relaxation(self, obj, inequalities, equalities,
                       monomial_substitutions, level,
                       removeequalities=False,
                       extramonomials=None, target='sdpa'):
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
        extramonomials -- monomials to be included, on top of the requested
                          level of relaxation
        """
        self.monomial_substitutions = monomial_substitutions
        # Generate monomials and remove substituted ones
        monomial_sets = []
        if self.independent_algebras:
            k = 0
            for variables in self.variables:
                extramonomials_ = None
                if extramonomials is not None:
                    extramonomials_ = extramonomials[k]
                monomial_sets.append(get_monomials(variables,
                                                   extramonomials_,
                                                   self.monomial_substitutions,
                                                   level))
                k += 1
        else:
            monomial_sets.append(get_monomials(self.variables,
                                               extramonomials,
                                               self.monomial_substitutions,
                                               level))

        if not (removeequalities or target == 'picos'):
            # Equalities are converted to pairs of inequalities
            for equality in equalities:
                inequalities.append(equality)
                inequalities.append(-equality)

        self.__calculate_block_structure(monomial_sets, inequalities, level)
        if self.independent_algebras:
            self.block_struct.append(2)

        self.n_vars = 0
        for monomials in monomial_sets:
            n_monomials = len(monomials)

            # The minus one compensates for the constant term in the
            # top left corner of the moment matrix
            self.n_vars += int(n_monomials * (n_monomials + 1) / 2)
            if not self.independent_algebras:
                self.n_vars -= 1

        n_rows = 0
        for block_size in self.block_struct:
            n_rows += block_size ** 2
        self.F_struct = lil_matrix((n_rows, self.n_vars + 1))

        if self.verbose > 0:
            print(('Number of SDP variables: %d' % self.n_vars))
            print('Generating moment matrix...')
       # Generate moment matrices
        new_n_vars, block_index = 0, 0
        var_offsets = [new_n_vars]
        for monomials in monomial_sets:
            new_n_vars, block_index = \
                self.__generate_moment_matrix(
                    new_n_vars,
                    block_index,
                    monomials)
            var_offsets.append(new_n_vars)
        self.n_vars = new_n_vars
        # The initial estimate for the size of F_struct was overly
        # generous. We correct the size here.
        self.F_struct = self.F_struct[:, 0:self.n_vars + 1]

        if self.verbose > 0:
            print(('Reduced number of SDP variables: %d' % self.n_vars))
        if self.verbose > 1:
            save_monomial_dictionary(
                "monomials.txt",
                self.monomial_dictionary,
                self.n_vars)

        if target == 'picos':
            return self.__convert_to_picos(inequalities, equalities, obj,
                                           monomial_sets)

        # Objective function
        self.obj_facvar = (self.__get_facvar(simplify_polynomial(obj, self.monomial_substitutions)))[1:]
        # Process inequalities
        if self.verbose > 0:
            print(('Processing %d inequalities...' % len(inequalities)))

        self.__process_inequalities(inequalities, flatten(monomial_sets),
                                    block_index)
        if removeequalities:
            A = self.__process_equalities(equalities, flatten(monomial_sets),
                                          level)
            self.__remove_equalities(equalities, A)
        if self.independent_algebras:
            self.F_struct[-1, 0] = -1
            self.F_struct[-4, 0] = 1
            for var in var_offsets[:-1]:
                self.F_struct[-1, var + 1] = 1
                self.F_struct[-4, var + 1] = -1

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
    

    def swap_objective(self, new_objective):
        """Swaps the objective function while keeping the moment matrix and
        the localizing matrices untouched"""
        self.obj_facvar = (
            self.__get_facvar(simplify_polynomial(new_objective, self.monomial_substitutions)))[1:]

    def __to_affine_expression(self, polynomial, X, row_offsets):
        """Helper function to create PICOS affine expressions from SymPy
        polynomials.
        """
        facvar = self.__get_facvar(polynomial)
        affine_expression = 0
        for k in range(len(facvar)):
            if facvar[k] != 0:
                row0 = self.F_struct[:, k].nonzero()[0][0]
                block_index, i, j = \
                    convert_row_to_sdpa_index(
                        self.block_struct,
                        row_offsets,
                        row0)
                affine_expression += facvar[k] * X[i, j]
        return affine_expression

    def __convert_to_picos(self, inequalities, equalities, obj, all_monomials):
        """PICOS compatibility layer.
        """
        if len(all_monomials) > 1:
            print('Independent algebras are not support by the PICOS \
                  compatibility layer')
            exit()
        all_monomials = all_monomials[0]
        import picos as pic
        P = pic.Problem()
        # Defining the momement matrix
        X = P.add_variable('X', (self.n_vars, self.n_vars), 'symmetric')
        P.add_constraint(X >> 0)
        P.add_constraint(X[0, 0] == 1)
        row_offsets = [0]
        cumulative_sum = 0
        for block_size in self.block_struct:
            cumulative_sum += block_size ** 2
            row_offsets.append(cumulative_sum)
        # Defining the symmetries of the moment matrix
        for k in range(self.n_vars):
            if self.F_struct[:self.block_struct[0] ** 2, k].getnnz() > 1:
                row0 = self.F_struct[:self.block_struct[0] ** 2,
                                     k].nonzero()[0][0]
                for row in self.F_struct[:self.block_struct[0] ** 2,
                                         k].nonzero()[0][1:]:
                    block_index, i1, j1 = \
                        convert_row_to_sdpa_index(
                            self.block_struct,
                            row_offsets,
                            row0)
                    block_index, i2, j2 = \
                        convert_row_to_sdpa_index(
                            self.block_struct,
                            row_offsets,
                            row)
                    if not (i1 == i2 and j1 == j2):
                        P.add_constraint(X[i2, j2] == X[i1, j1])
        # Iterate over the inequalities
        block_index = 0
        for ineq in inequalities:
            block_index += 1
            localization_order = self.localization_order[
                block_index - 1]
            monomials = \
                pick_monomials_up_to_degree(all_monomials, localization_order)
            Y = P.add_variable(('Y%s' % block_index),
                               (len(monomials), len(monomials)), 'symmetric')
            P.add_constraint(Y >> 0)
            # Process M_y(gy)(u,w) entries
            for row in range(len(monomials)):
                for column in range(row, len(monomials)):
                    # Calculate the moments of polynomial entries
                    polynomial = \
                        simplify_polynomial(
                            Dagger(monomials[row]) * ineq * monomials[column],
                            self.monomial_substitutions)
                    affine_expression = \
                        self.__to_affine_expression(polynomial, X, row_offsets)
                    P.add_constraint(Y[row, column] == affine_expression)
        for equality in equalities:
            affine_expression = self.__to_affine_expression(equality, X,
                                                            row_offsets)
            P.add_constraint(affine_expression == 0)

        # Set the objective
        affine_expression = self.__to_affine_expression(obj, X, row_offsets)
        P.set_objective('min', affine_expression)
        return P
