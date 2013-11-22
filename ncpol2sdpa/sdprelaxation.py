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
from ncutils import get_ncmonomials, count_ncmonomials, fastSubstitute, ncdegree

class entry:
    def __init__(self, block_index, row, column, value):
        self.block_index = block_index
        self.row = row
        self.column = column
        self.value = value


class SdpRelaxation:
    'Class for obtaining sparse SDP relaxation'

    monomial_substitutions = {}
    monomial_dictionary = {}
    n_vars = 0
    blacklist = []    
    F = []
    block_struct = []
    obj_facvar = 0
    variable_blocks = []
    extra_variables = []
    n_monomials_in_blocks = []
    offsets = []
    
    def __init__(self, variable_blocks, extra_variables=[]):
        if isinstance(variable_blocks, list):
            if isinstance(variable_blocks[0], list):
                self.variable_blocks = variable_blocks
            else:
                self.variable_blocks = [ variable_blocks ]
        else:
            self.variable_blocks = [ [ variable_blocks ] ]
        self.extra_variables = extra_variables
    
    def __apply_substitutions(self, monomial):
        """Helper function to remove monomials from the basis."""
        originalMonomial = monomial
        changed = True
        while changed:
            for lhs, rhs in self.monomial_substitutions.items():
                #monomial = monomial.subs(lhs, rhs)
                monomial = fastSubstitute(monomial, lhs, rhs)
            if (originalMonomial == monomial):
                changed = False
            originalMonomial = monomial
        return monomial
    
    
    def __index2linear(self, i, j, monomial_block_index):
        n_monomials = self.n_monomials_in_blocks[monomial_block_index]
        if i==0:
            return self.offsets[monomial_block_index] + j + 1
        else:
            skew = int(i * (i + 1) / 2)
            return self.offsets[monomial_block_index] + i * n_monomials-skew + j + 1
    
    def __get_facvar(self, polynomial):
        """Returns a dense vector representation of a polynomial"""
        facvar = [ 0 ] * self.n_vars
        if isinstance( polynomial, int ):
            return facvar

        # Is this a monomial?
        polynomial = polynomial.expand()
        if polynomial.is_Mul:
            elements = [ polynomial ]
        else:
            elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]            
        for element in elements:
            coeff = 1.0
            monomial = S.One
            for var in element.as_coeff_mul()[1]:
                if not var.is_Number:
                    monomial = monomial * var
                else:
                    coeff = float(var)
            coeff = float(element.as_coeff_mul()[0]) * coeff
            monomial = self.__apply_substitutions(monomial)
            if monomial.as_coeff_Mul()[0] < 0:
                monomial = -monomial
                coeff = -1.0 * coeff
            if monomial in self.extra_variables:
                k = self.offsets[-1] + self.extra_variables.index(monomial) 
            else:
                if monomial in self.monomial_dictionary:
                    indices = self.monomial_dictionary[monomial]
                else:
                    indices = self.monomial_dictionary[Dagger(monomial)]
                    indices[0], indices[1] = indices[1], indices[0]
                k = self.__index2linear(indices[0],indices[1],indices[2])-1 
            facvar[k] += coeff    
    
        return facvar

    def __push_facvar_sparse(self, polynomial, block_index, i, j):
        """Calculates the sparse vector representation of a polynomial
        and pushes it to the F structure.
        """
        polynomial = polynomial.expand()
        if polynomial.is_Mul:
            elements = [ polynomial ]
        else:
            elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]            
        for element in elements:
            coeff = 1.0
            monomial = S.One
            for var in element.as_coeff_mul()[1]:
                if not var.is_Number:
                    monomial = monomial * var
                else:
                    coeff = float(var)
            coeff = float(element.as_coeff_mul()[0]) * coeff
            monomial = self.__apply_substitutions(monomial)
            if monomial != 0:
                if monomial.as_coeff_Mul()[0] < 0:
                    monomial = -monomial
                    coeff = -1.0 * coeff
                k = -1
                if monomial in self.extra_variables:
                    k = self.offsets[-1] + self.extra_variables.index(monomial) + 1
                else:                
                    if monomial in self.monomial_dictionary:
                        indices = self.monomial_dictionary[monomial]
                        k = self.__index2linear(indices[0],indices[1],indices[2])
                    elif Dagger(monomial) in self.monomial_dictionary:
                        indices = self.monomial_dictionary[Dagger(monomial)]
                        indices[0], indices[1] = indices[1], indices[0]
                        k = self.__index2linear(indices[0],indices[1],indices[2])
                if k > -1:
                    e=entry(block_index, i+1, j+1, coeff)
                    self.F[k].append(e)
    
    def __generate_moment_matrix(self, n_eq, monomials, block_index, monomial_block_index):
        """Generates the moment matrix of monomials, but does not add SDP
        constraints.
        """
        # Adding symmetry constraints of momentum matrix and generating 
        # monomial dictionary
        for i in range(len(monomials)):
            for j in range(i, len(monomials)):
                monomial = Dagger(monomials[i]) * monomials[j]
                monomial = self.__apply_substitutions(monomial)
                if monomial.as_coeff_Mul()[0] < 0:
                    monomial = -monomial
                if monomial != 0:
                    if monomial in self.monomial_dictionary and monomial != 1:
                        indices = self.monomial_dictionary[monomial]
                        e = entry(block_index, n_eq, n_eq, 1)
                        self.F[self.__index2linear(indices[0],indices[1], indices[2])].append(e)
                        e = entry(block_index, n_eq, n_eq, -1)
                        self.F[self.__index2linear(i,j,monomial_block_index)].append(e)
                        n_eq+=1
                        e = entry(block_index, n_eq, n_eq, -1)
                        self.F[self.__index2linear(indices[0],indices[1], indices[2])].append(e)
                        e = entry(block_index, n_eq, n_eq, 1)
                        self.F[self.__index2linear(i,j,monomial_block_index)].append(e)
                        n_eq+=1
                    else:
                        self.monomial_dictionary[monomial] = [i, j, monomial_block_index]
                else:
                    self.blacklist.append(self.__index2linear(i,j,monomial_block_index))
        n_eq-=1
        self.block_struct=[-n_eq]
        return n_eq+1
            
    def __process_inequalities(self, inequalities, monomials, block_index, order):
        global block_struct
        global F
        for ineq in inequalities:
                max_order = ncdegree(ineq)
                localization_matrix_order = floor((2*order-max_order)/2)
                if localization_matrix_order >= 0:
                    ineq_n_monomials = count_ncmonomials(monomials, localization_matrix_order)
                    self.block_struct.append(ineq_n_monomials)
                    block_index+=1
                    for i in range(ineq_n_monomials):
                        for j in range(i, ineq_n_monomials):
                            polynomial = Dagger(monomials[i]) * ineq * monomials[j]
                            self.__push_facvar_sparse(polynomial, block_index, i, j)
        return block_index
    
    def get_relaxation(self, obj, inequalities, equalities, 
                       monomial_substitutions, order, verbose=0):
        """Gets the SDP relaxation of a noncommutative polynomial optimization
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
            monomial_blocks.append(get_ncmonomials(variables, order))

        for monomials in monomial_blocks:
            for monomial in list(self.monomial_substitutions.keys()):
                if monomials.__contains__(monomial):
                    monomials.remove(monomial)

    
        self.n_vars = 0
        #self.n_monomials_in_blocks = [0] * len(monomial_blocks)
        self.offsets = [ 0 ]
        for monomials in monomial_blocks:
            n_monomials = len(monomials)
            self.n_monomials_in_blocks.append(n_monomials)
            self.n_vars += int(n_monomials*(n_monomials+1)/2)
            self.offsets.append(self.n_vars)
        self.n_vars += len(self.extra_variables)
    
        if verbose>0:
            print('Number of SDP variables: %d' % self.n_vars)
            print('Generating moment matrix...');
        # The diagonal part of the SDP problem contains equalities on symmetry 
        # constraints encoded as pairs of inequalities
        self.F = [0] * (self.n_vars+1)
        for i in range(self.n_vars+1):
            self.F[i] = []

        # Defining top left entry
        block_index=1
        n_eq=1
        e = entry(block_index, n_eq, n_eq, 1)
        self.F[0].append(e)
        for monomial_block_index in range(len(monomial_blocks)):
            self.F[self.__index2linear(0,0,monomial_block_index)].append(e)
        n_eq += 1
        e = entry(block_index, n_eq, n_eq, -1)
        self.F[0].append(e)
        for monomial_block_index in range(len(monomial_blocks)):
            self.F[self.__index2linear(0,0,monomial_block_index)].append(e)
        n_eq += 1

       # Generate moment matrices for each sets of variables
        for monomial_block_index in range(len(monomial_blocks)):
            n_eq = self.__generate_moment_matrix(n_eq, monomial_blocks[monomial_block_index], block_index, monomial_block_index)
        
        # Objective function
        self.obj_facvar=self.__get_facvar(obj)
        
        for eq in equalities:
            inequalities.append(eq)
            inequalities.append(-eq)
    
        if verbose>0:
            print('Processing %d inequalities...' % len(inequalities))

        block_index = self.__process_inequalities(inequalities, monomials, block_index, order)

        for monomial_block_index in range(len(monomial_blocks)):
            block_index+=1
            for i in range(self.n_monomials_in_blocks[monomial_block_index]):
                for j in range(i,self.n_monomials_in_blocks[monomial_block_index]):
                    k=self.__index2linear(i,j, monomial_block_index)
                    if k not in self.blacklist:
                        e = entry(block_index, i+1, j+1, 1)
                        self.F[k].append(e)
                                
            self.block_struct.append(len(monomials))

        
    def write_to_sdpa(self, filename):
        """Writes the SDP relaxation to SDPA format.
        
        Arguments:
        filename -- the name of the file. It must have the suffix ".dat-s"
        """

        f = open(filename,'w')
        f.write('"file '+filename+' generated by ncpol2sdpa"\n')
        f.write(str(self.n_vars)+' = number of vars\n')
        f.write(str(len(self.block_struct))+' = number of blocs\n')
        #bloc structure
        f.write(str(self.block_struct).replace('[','(').replace(']',')'))
        f.write(' = BlocStructure\n')
        #c vector (objective)
        f.write(str(list(self.obj_facvar)).replace('[','{').replace(']','}'))
        f.write('\n')
        #coefs
        for k in range(self.n_vars+1):
            for e in self.F[k]:
                f.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
                k,e.block_index, e.row, e.column, e.value))
        
        f.close()
