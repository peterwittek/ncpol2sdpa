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
try:
    from scipy.linalg import qr
    from scipy.sparse import lil_matrix, hstack
except ImportError:
    from .sparse_utils import lil_matrix
from .nc_utils import apply_substitutions, build_monomial, \
    pick_monomials_up_to_degree, ncdegree, \
    separate_scalar_factor, flatten, build_permutation_matrix, \
    simplify_polynomial, save_monomial_index, get_monomials, unique
from .chordal_extension import generate_clique, find_clique_index

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
                       * "nieto-silleras": `doi:10.1088/1367-2630/16/1/013035 <http://dx.doi.org/10.1088/1367-2630/16/1/013035>`_
                       * "moroder": `doi:10.1103/PhysRevLett.111.030501 <http://dx.doi.org/10.1103/PhysRevLett.111.030501>`_
    :type hierarchy: str.
    :param normalized: Optional parameter for changing the normalization of
                       states over which the optimization happens. Turn it off
                       if further processing is done on the SDP matrix before
                       solving it.
    :type normalized: bool.
    """
    hierarchy_types = ["npa", "npa_chordal", "nieto-silleras", "moroder"]

    def __init__(self, variables, nonrelaxed=None, verbose=0, hierarchy="npa",
                 normalized=True):
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
        return k

    def __push_monomial(self, monomial, n_vars, row_offset, rowA, columnA, N,
                        rowB, columnB, lenB):
        monomial = apply_substitutions(monomial,
                                       self.substitutions)
        if monomial == 1 and self.normalized:
            if self.hierarchy == "nieto-silleras":
                k = n_vars + 1
                n_vars = k
                self.F_struct[row_offset + rowA * N*lenB +
                              rowB * N + columnA * lenB + columnB, k] = 1
            else:
                self.F_struct[row_offset + rowA * N*lenB +
                              rowB * N + columnA * lenB + columnB, 0] = 1

        elif monomial != 0:
            k = self.__process_monomial(monomial, n_vars)
            if k > n_vars:
                n_vars = k
            # We push the entry to the moment matrix
            self.F_struct[row_offset + rowA * N*lenB +
                          rowB * N +
                          columnA * lenB + columnB, k] = 1
        return n_vars

    def __generate_moment_matrix(self, n_vars, block_index,
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
                        monomial = Dagger(monomialsA[rowA]) * \
                                   monomialsA[columnA] * \
                                   Dagger(monomialsB[rowB]) * \
                                   monomialsB[columnB]
                        # Apply the substitutions if any
                        n_vars = self.__push_monomial(monomial, n_vars,
                                                      row_offset, rowA,
                                                      columnA, N, rowB,
                                                      columnB, len(monomialsB))
                        if self.verbose > 0:
                            sys.stdout.write("\r\x1b[KCurrent number of SDP variables: %d" % n_vars)
                            sys.stdout.flush()
        if self.verbose > 0:
            sys.stdout.write("\r")
        return n_vars, block_index + 1


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
                print(("An equality constraint has degree %d. Choose a higher \
                      level of relaxation." % eq_order))
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

    def __calculate_block_structure(self, inequalities, equalities, bounds,
                                    removeequalities):
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
        if self.hierarchy == "nieto-silleras":
            self.block_struct.append(2)
        degree_warning = False
        if inequalities is not None:
            n_inequalities = len(inequalities)
        else:
            n_inequalities = 0
        if removeequalities:
            constraints = flatten([inequalities])
        else:
            constraints = enumerate(flatten([inequalities, equalities]))
        for k, constraint in constraints:
            # Find the order of the localizing matrix
            ineq_order = ncdegree(constraint)
            if ineq_order > 2 * self.level:
                degree_warning = True
            localization_order = int(floor((2 * self.level - ineq_order) / 2))
            if self.hierarchy == "nieto-silleras":
                localization_order = 0
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
            if self.hierarchy == "nieto-silleras" or \
              len(localizing_monomials) == 0:
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

    def __generate_monomial_sets(self, objective, inequalities, equalities,
                                 extramonomials):
        if self.hierarchy == "nieto-silleras" or self.hierarchy == "moroder":
            k = 0
            for variables in self.variables:
                extramonomials_ = None
                if extramonomials is not None:
                    extramonomials_ = extramonomials[k]
                self.monomial_sets.append(get_monomials(variables,
                                                        extramonomials_,
                                                        self.substitutions,
                                                        self.level))
                k += 1
        elif self.hierarchy == "npa_chordal":
            self.clique_set = generate_clique(self.variables, objective,
                                              inequalities, equalities)
            if self.verbose > 1:
                print(self.clique_set)
            for clique in self.clique_set:
                variables = [self.variables[i] for i in np.nonzero(clique)[0]]
                self.monomial_sets.append(get_monomials(variables,
                                                        extramonomials,
                                                        self.substitutions,
                                                        self.level))
        else:
            self.monomial_sets.append(get_monomials(self.variables,
                                                    extramonomials,
                                                    self.substitutions,
                                                    self.level))

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

    def set_objective(self, objective, nsextraobjvars=None):
        """Set or change the objective function of the polynomial optimization
        problem.

        :param objective: Describes the objective function.
        :type objective: :class:`sympy.core.expr.Expr`
        :param nsextraobjvars: Optional parameter of the coefficients of
                               unnormalized top left elements of the moment
                               matrices of the Nieto-Silleras hierarchy that
                               should be included in the objective function.
        :type nsextraobjvars: list of float.
        """
        if objective is not None:
            self.obj_facvar = (
                self.__get_facvar(
                    simplify_polynomial(
                        objective,
                        self.substitutions)))[1:]
        else:
            self.obj_facvar = self.__get_facvar(0)
        if nsextraobjvars is not None:
            if self.hierarchy == "nieto-silleras":
                if len(nsextraobjvars) == len(self.var_offsets)-1:
                    for i, coeff in enumerate(nsextraobjvars):
                        self.obj_facvar[self.var_offsets[i]] = coeff
                else:
                    raise Exception("The length of nsextraobjvars does not " +
                                    "match the number of blocks in the Nieto-"+
                                    "Silleras relaxation")
            else:
                raise Exception("nsextraobjvars is only meaningful with the " +
                                "Nieto-Silleras relaxation")

    def add_non_relaxed(self):
        new_n_vars, block_index = 0, 0
        if self.nonrelaxed is not None:
            block_index = 1
            for var in self.nonrelaxed:
                new_n_vars += 1
                self.monomial_index[var] = new_n_vars
                self.F_struct[new_n_vars - 1, new_n_vars] = 1
        return new_n_vars, block_index

    def wipe_F_struct_from_constraints(self):
        row_offset = 0
        for block in range(self.constraint_starting_block):
            row_offset += self.block_struct[block]**2
        for row in range(row_offset, len(self.F_struct.rows)):
            self.F_struct.rows[row] = []
            self.F_struct.data[row] = []

    def process_constraints(self, inequalities=None, equalities=None,
                            bounds=None, block_index=0, removeequalities=False):
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
        :param removeequalities: Optional parameter to attempt removing the
                                 equalities by solving the linear equations.
        :type removeequalities: bool.
        """
        if block_index == 0:
            block_index = self.constraint_starting_block
            self.wipe_F_struct_from_constraints()
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

    def normalize_nieto_silleras(self, block_index):
        width = self.block_struct[block_index]
        row_offset = 0
        for block in range(block_index):
            row_offset += self.block_struct[block]**2
        if self.normalized:
            self.F_struct[row_offset, 0] = -1
            self.F_struct[row_offset + width + 1, 0] = 1
            for var in self.var_offsets[:-1]:
                self.F_struct[row_offset, var + 1] = 1
                self.F_struct[row_offset + width + 1, var + 1] = -1
        return block_index + 1


    def get_relaxation(self, level, objective=None, inequalities=None,
                       equalities=None, substitutions=None, bounds=None,
                       removeequalities=False, extramonomials=None,
                       nsextraobjvars=None):
        """Get the SDP relaxation of a noncommutative polynomial optimization
        problem.

        :param level: The level of the relaxation
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
        :param removeequalities: Optional parameter to attempt removing the
                                 equalities by solving the linear equations.
        :type removeequalities: bool.
        :param extramonomials: Optional paramter of monomials to be included, on
                               top of the requested level of relaxation.
        :type extramonomials: list of :class:`sympy.core.exp.Expr`.
        :param nsextraobjvars: Optional parameter of the coefficients of
                               unnormalized top left elements of the moment
                               matrices of the Nieto-Silleras hierarchy that
                               should be included in the objective function.
        :type nsextraobjvars: list of float.
        """
        self.level = level
        if substitutions is None:
            self.substitutions = {}
        else:
            self.substitutions = substitutions
        # Generate monomials and remove substituted ones
        self.__generate_monomial_sets(objective, inequalities, equalities,
                                      extramonomials)
        # Figure out basic structure of the SDP
        self.__calculate_block_structure(inequalities, equalities, bounds,
                                         removeequalities)
        self.__estimate_n_vars()
        self.F_struct = lil_matrix((sum([bs ** 2 for bs in self.block_struct]),
                                    self.n_vars + 1))

        if self.verbose > 0:
            print(('Estimated number of SDP variables: %d' % self.n_vars))
            print('Generating moment matrix...')
       # Generate moment matrices
        new_n_vars, block_index = self.add_non_relaxed()
        if self.hierarchy == "moroder":
            new_n_vars, block_index = \
                self.__generate_moment_matrix(new_n_vars, block_index,
                                              self.monomial_sets[0],
                                              self.monomial_sets[1])
        else:
            for monomials in self.monomial_sets:
                new_n_vars, block_index = \
                    self.__generate_moment_matrix(
                        new_n_vars,
                        block_index,
                        monomials, [monomials[0]])
                self.var_offsets.append(new_n_vars)

        # The initial estimate for the size of F_struct was overly generous.
        self.n_vars = new_n_vars
        # We don't correct the size of F_struct, because that would trigger
        # memory copies, and extra columns in lil_matrix are free anyway.
        # self.F_struct = self.F_struct[:, 0:self.n_vars + 1]

        # Normalizing the Nieto-Silleras hierarchy before processing the
        # constraints
        if self.hierarchy == "nieto-silleras":
            block_index = self.normalize_nieto_silleras(block_index)
        if self.verbose > 0:
            print(('Reduced number of SDP variables: %d' % self.n_vars))
        if self.verbose > 1:
            save_monomial_index("monomials.txt", self.monomial_index,
                                self.n_vars)
        # Objective function
        self.set_objective(objective, nsextraobjvars)

        # Process constraints
        self.constraint_starting_block = block_index
        self.process_constraints(inequalities, equalities, bounds, block_index,
                                 removeequalities)
