# -*- coding: utf-8 -*-
"""
The module converts a noncommutative optimization problem provided in SymPy
format to an SDPA semidefinite programming problem.

Created on Sun May 26 15:06:17 2013

@author: Peter Wittek
"""
from math import floor
from sympy import S
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import HermitianOperator
from .ncutils import apply_substitutions, build_monomial, get_ncmonomials, \
                    pick_monomials_up_to_degree, get_variables_of_polynomial,\
                    ncdegree


class Entry(object):

    """Class for storing entries in the constraint matrices of the SDP
    relaxation.
    """

    def __init__(self, block_index, row, column, value):
        self.block_index = block_index
        self.row = row
        self.column = column
        self.value = value


class SdpRelaxation(object):

    """Class for obtaining sparse SDP relaxation.
    """

    monomial_substitutions = {}
    monomial_dictionary = {}
    n_vars = 0
    F = []
    block_struct = []
    obj_facvar = 0
    variable_blocks = []
    extra_variables = []
    n_monomials_in_blocks = []
    offsets = []

    def __init__(self, variable_blocks, extra_variables=None):
        if isinstance(variable_blocks, list):
            if isinstance(variable_blocks[0], list):
                self.variable_blocks = variable_blocks
            else:
                self.variable_blocks = [variable_blocks]
        else:
            self.variable_blocks = [[variable_blocks]]
        if extra_variables is not None:
            self.extra_variables = extra_variables

    def __index2linear(self, i, j, monomial_block_index):
        n_monomials = self.n_monomials_in_blocks[monomial_block_index]
        if i==0:
            return self.offsets[monomial_block_index] + j 
        else:
            skew = int(i * (i + 1) / 2)
            return self.offsets[monomial_block_index] + \
              i * n_monomials-skew + j 

    def __get_index_of_monomial(self, element):
        monomial, coeff = build_monomial(element)
        monomial = apply_substitutions(monomial, self.monomial_substitutions)
        # Given the monomial, we need its mapping L_y(w) to push it into
        # a corresponding constraint matrix
        if monomial != 0:
            if monomial.as_coeff_Mul()[0] < 0:
                monomial = -monomial
                coeff = -1.0 * coeff
        k = -1
        if monomial in self.extra_variables:
            k = self.offsets[-1] +  \
                self.extra_variables.index(monomial) + 1
        elif monomial.is_Number:
            k = 0
            coeff *= -1
        else:
            try:
                indices = self.monomial_dictionary[monomial]
                k = self.__index2linear(indices[0], indices[1],
                                      indices[2])
            except KeyError:
                [monomial, coeff] = build_monomial(element)
                monomial = apply_substitutions(Dagger(monomial),
                                               self.monomial_substitutions)
                indices = self.monomial_dictionary[monomial]
                indices[0], indices[1] = indices[1], indices[0]
            k = self.__index2linear(indices[0], indices[1],
                                      indices[2])                  
        return k, coeff

    def __push_facvar_sparse(self, polynomial, block_index, i, j):
        """Calculate the sparse vector representation of a polynomial
        and pushes it to the F structure.
        """

        # Preprocess the polynomial for uniform handling later
        polynomial = polynomial.expand()
        if polynomial.is_Mul:
            elements = [polynomial]
        else:
            elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]
        # Identify its constituent monomials
        for element in elements:
            k, coeff = self.__get_index_of_monomial(element)
            # k identifies the mapped value of a word (monomial) w
            if k > -1:
                e = Entry(block_index, i + 1, j + 1, coeff)
                self.F[k].append(e)

    def __get_facvar(self, polynomial):
        """Return dense vector representation of a polynomial. This function is
        nearly identical to __push_facvar_sparse, but instead of pushing
        sparse entries to the constraint matrices, it returns a dense
        vector.
        """
        facvar = [0] * self.n_vars
        if isinstance(polynomial, int):
            return facvar

        # Preprocess the polynomial for uniform handling later
        polynomial = polynomial.expand()
        if polynomial.is_Mul:
            elements = [polynomial]
        else:
            elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]
        for element in elements:
            k, coeff = self.__get_index_of_monomial(element)
            if k > 0:
                facvar[k-1] += coeff
        return facvar

    def __process_monomial(self, monomial, row, column, monomial_block_index):
        """Process a single monomial when building the moment matrix.
        """
        if monomial.as_coeff_Mul()[0] < 0:
            monomial = -monomial
        k = 0
        # Have we seen this monomial before?
        try:
            # If yes, then we improve sparsity by reusing the
            # previous variable to denote this Entry in the matrix
            indices = self.monomial_dictionary[monomial]
            k = self.__index2linear(indices[0], indices[1],
                                        indices[2])
        except KeyError:
            # Otherwise we define a new Entry in the associated
            # array recording the monomials, and add an Entry in
            # the moment matrix
            self.monomial_dictionary[monomial] = \
                [row, column, monomial_block_index]
            k = self.__index2linear(row, column,
                                    monomial_block_index)
        return k

    def __generate_moment_matrix(self, monomials, block_index,
                                 monomial_block_index):
        """Generate the moment matrix of monomials.

        Arguments:
        monomials -- |W_d| set of words of length up to the relaxation order d
        block_index -- current block index in the constraints matrices of the
                       SDP relaxation
        monomial_block_index -- if there are multiple independent algebras
                                defining several monomial blocks, this
                                parameter which block we are in
        """
        block_index += 1
        # We process the M_d(u,w) entries in the moment matrix
        for row in range(len(monomials)):
            for column in range(row, len(monomials)):
                # Calculate the monomial u*v
                monomial = Dagger(monomials[row]) * monomials[column]
                # Apply the substitutions if any
                monomial = apply_substitutions(monomial,
                                               self.monomial_substitutions)
                if monomial == 1:
                    e = Entry(block_index, 1, 1, -1)
                    self.F[0].append(e)
                elif monomial != 0:
                    k = self.__process_monomial(monomial, row, column,
                                                monomial_block_index)
                    # We push the Entry to the moment matrix
                    e = Entry(block_index, row + 1, column + 1, 1)
                    self.F[k].append(e)

        self.block_struct.append(len(monomials))
        return block_index

    def __get_monomial_block_index(self, polynomial):
        """Get the index of the monomial block given a polynomial.
        """
        if len(self.variable_blocks) == 1:
            return [0]
        polynomial_variables = \
            get_variables_of_polynomial(polynomial.expand())
        block_index_list = []
        block_index = 0
        for variables in self.variable_blocks:
            if any([i in polynomial_variables for i in variables]):
                block_index_list.append(block_index)
            block_index += 1
        return block_index_list

    def __process_inequalities(
            self, inequalities, monomial_blocks, block_index, order):
        """Generate localizing matrices

        Arguments:
        inequalities -- list of inequality constraints
        monomial_blocks -- monomials arranged in blocks according to
                           independent algebras
        block_index -- the current block index in constraint matrices of the
                       SDP relaxation
        order -- the order of the relaxation
        """
        for ineq in inequalities:
            # Find the order of the localizing matrix
            max_order = ncdegree(ineq)
            localization_matrix_order = int(floor((2 * order - max_order) / 2))
            if localization_matrix_order >= 0:
                # Identify the correct set of monomials
                monomial_block_index_list = self.__get_monomial_block_index(
                    ineq)
                monomials = \
                    pick_monomials_up_to_degree(
                        monomial_blocks, monomial_block_index_list,
                        localization_matrix_order
                    )
                # Mark length of block in the constraint matrices
                self.block_struct.append(len(monomials))
                block_index += 1
                # Process M_y(gy)(u,w) entries
                for row in range(len(monomials)):
                    for column in range(row, len(monomials)):
                        # Calculate the moments of polynomial entries
                        polynomial = \
                            Dagger(monomials[row]) * ineq * monomials[column]
                        self.__push_facvar_sparse(polynomial, block_index,
                                row, column)
        return block_index


    def get_relaxation(self, obj, inequalities, equalities,
                       monomial_substitutions, order, verbose=0):
        """Get the SDP relaxation of a noncommutative polynomial optimization
        problem.

        Arguments:
        obj -- the objective function
        inequalities -- list of inequality constraints
        equalities -- list of equality constraints
        monomial_substitutions -- monomials that can be replaced
                                  (e.g., idempotent variables)
        order -- the order of the relaxation
        """
        self.monomial_substitutions = monomial_substitutions
        # Generate monomials and remove substituted ones
        monomial_blocks = []
        for variables in self.variable_blocks:
            monomials = get_ncmonomials(variables, order)
            monomials = [monomial for monomial in monomials if monomial not 
              in self.monomial_substitutions]
            monomial_blocks.append(monomials)

        # Adjust monomial blocks. This is only necessary if independent
        # algebras generate the moment matrix.
        monomial_block_index = 0
        for monomials in monomial_blocks:
            if len(monomial_blocks) > 1:
                identity_operator = HermitianOperator(
                    "1_%s" %
                    (monomial_block_index))
                identity_operator.is_commutative = True
                monomials[0] = identity_operator
                self.monomial_substitutions[
                    identity_operator *
                    identity_operator] = identity_operator

            monomial_block_index += 1

        # Initialize some helper variables, including the offsets of monomial
        # blocks if there is more than one.
        self.n_vars = 0
        self.offsets = [0]
        for monomials in monomial_blocks:
            n_monomials = len(monomials)
            self.n_monomials_in_blocks.append(n_monomials)
            # The minus one compensates for the constant term in the
            # top left corner of the moment matrix
            self.n_vars += int(n_monomials*(n_monomials+1)/2) - 1
            self.offsets.append(self.n_vars)
        self.n_vars += len(self.extra_variables)
        
        if verbose > 0:
            print('Number of SDP variables: %d' % self.n_vars)
            print('Generating moment matrix...')

        # Initialize sparse constant matrices in the target SDP
        self.F = [0] * (self.n_vars + 1)
        for i in range(self.n_vars + 1):
            self.F[i] = []

        # Define top left Entry of the moment matrix, y_1 = 1
        block_index = 0

       # Generate moment matrices for each blocks of variables
        for monomial_block_index in range(len(monomial_blocks)):
            block_index = self.__generate_moment_matrix(
                monomial_blocks[monomial_block_index],
                block_index,
                monomial_block_index)

        # Objective function
        self.obj_facvar = self.__get_facvar(obj)

        # Equalities are converted to pairs of inequalities
        for equality in equalities:
            inequalities.append(equality)
            inequalities.append(-equality)

        # Process inequalities
        if verbose > 0:
            print('Processing %d inequalities...' % len(inequalities))

        block_index = self.__process_inequalities(
            inequalities,
            monomial_blocks,
            block_index,
            order)
        self.__compact_sdp_variables()

    def __compact_sdp_variables(self):
        """Discard unused relaxation variables.
        """
        new_n_vars = 0
        new_F = []
        new_obj_facvar = []
        for i in range(self.n_vars + 1):
            if len(self.F[i]) > 0:
                new_n_vars += 1
                new_F.append(self.F[i])
                if i > 0:
                    new_obj_facvar.append(self.obj_facvar[i-1])
        self.n_vars = new_n_vars - 1
        self.F = new_F
        self.obj_facvar = new_obj_facvar

    def __save_monomial_dictionary(self, filename):
        """Save the current monomial dictionary for debugging purposes.
        """
        monomial_translation = [''] * (self.n_vars + 1)
        for key, indices in self.monomial_dictionary.iteritems():
            monomial = ('%s' % key)
            monomial = monomial.replace('Dagger(', '')
            monomial = monomial.replace(')', 'T')
            monomial = monomial.replace('**', '^')
            k = self.__index2linear(indices[0], indices[1], indices[2])
            monomial_translation[k] = monomial
        f = open(filename, 'w')
        for k in range(len(monomial_translation)):
            f.write('%s %s\n' % (k, monomial_translation[k]))
        f.close()

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
        for k in range(self.n_vars + 1):
            for e in self.F[k]:
                f.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
                        k, e.block_index, e.row, e.column, e.value))

        f.close()
