# -*- coding: utf-8 -*-
"""
The module converts a noncommutative optimization problem provided in SymPy 
format to an SDPA semidefinite programming problem.

Created on Sun May 26 15:06:17 2013

@author: Peter Wittek
"""
from sympy import S
from sympy.physics.quantum.dagger import Dagger
from ncutils import get_ncmonomials, count_ncmonomials, fastSubstitute

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
    n_monomials = 0
    n_vars = 0
    F = []
    block_struct = []
    obj_facvar = 0
    variables = []
    
    def __init__(self, variables):
        self.variables=variables
    
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
    
    
    def __index2linear(self, i, j):
        if i==0:
            return j + 1
        else:
            skew = int(i * (i + 1) / 2)
            return i * self.n_monomials-skew + j + 1
    
    def __get_facvar(self, polynomial):
        """Returns a dense vector representation of a polynomial"""
        facvar = [ 0 ] * self.n_vars
        for element in polynomial.expand().as_coeff_mul()[1][0].as_coeff_add()[1]:
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
            if monomial in self.monomial_dictionary:
                indices = self.monomial_dictionary[monomial]
            else:
                indices = self.monomial_dictionary[Dagger(monomial)]
                indices[0], indices[1] = indices[1], indices[0]
            facvar[self.__index2linear(indices[0],indices[1])-1] += coeff    
    
        return facvar
        
    
    def __generate_moment_matrix(self, monomials, block_index):
        """Generates the moment matrix of monomials, but does not add SDP
        constraints.
        """
        # Defining top left entry
        n_eq=1
        e = entry(block_index, n_eq, n_eq, 1)
        self.F[0].append(e)
        self.F[self.__index2linear(0,0)].append(e)
        n_eq += 1
        e = entry(block_index, n_eq, n_eq, -1)
        self.F[0].append(e)
        self.F[self.__index2linear(0,0)].append(e)
        n_eq += 1
        # Adding symmetry constraints of momentum matrix and generating 
        # monomial dictionary
        for i in range(self.n_monomials):        
            for j in range(i, self.n_monomials):
                monomial = Dagger(monomials[i]) * monomials[j]
                monomial = self.__apply_substitutions(monomial)
                if monomial.as_coeff_Mul()[0] < 0:
                    monomial = -monomial
                if monomial in self.monomial_dictionary:
                    indices = self.monomial_dictionary[monomial]
                    e = entry(block_index, n_eq, n_eq, 1)
                    self.F[self.__index2linear(indices[0],indices[1])].append(e)
                    e = entry(block_index, n_eq, n_eq, -1)
                    self.F[self.__index2linear(i,j)].append(e)
                    n_eq+=1
                    e = entry(block_index, n_eq, n_eq, -1)
                    self.F[self.__index2linear(indices[0],indices[1])].append(e)
                    e = entry(block_index, n_eq, n_eq, 1)
                    self.F[self.__index2linear(i,j)].append(e)
                    n_eq+=1
                else:
                    self.monomial_dictionary[monomial] = [i, j]
    
        n_eq-=1
        self.block_struct=[-n_eq]
    
    def __push_facvar_sparse(self, polynomial, block_index, i, j):
        """Calculates the sparse vector representation of a polynomial
        and pushes it to the F structure.
        """
        for element in polynomial.expand().as_coeff_mul()[1][0].as_coeff_add()[1]:
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
            if monomial in self.monomial_dictionary:
                indices = self.monomial_dictionary[monomial]
            else:
                indices = self.monomial_dictionary[Dagger(monomial)]
                indices[0], indices[1] = indices[1], indices[0]
            e=entry(block_index, i+1, j+1, coeff)
            k=self.__index2linear(indices[0],indices[1])
            self.F[k].append(e)
    
    
    def __process_inequalities(self, inequalities, monomials, block_index, order):
        global block_struct
        global F
        ineq_n_monomials = count_ncmonomials(self.variables, monomials, order - 1)
        for ineq in inequalities:
             self.block_struct.append(ineq_n_monomials)
             block_index+=1
             for i in range(ineq_n_monomials):
                for j in range(i, ineq_n_monomials):
                    polynomial = Dagger(monomials[i]) * ineq * monomials[j]
                    self.__push_facvar_sparse(polynomial, block_index, i, j)
        return block_index
        
    
    def get_relaxation(self, obj, inequalities, equalities, 
                       monomial_substitutions, order):
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
        monomials = get_ncmonomials(self.variables, order)
        for monomial in list(self.monomial_substitutions.keys()):
            if monomials.__contains__(monomial):
                monomials.remove(monomial)
    
        self.n_monomials = len(monomials)
        self.n_vars = int(self.n_monomials*(self.n_monomials+1)/2)
    
        # The diagonal part of the SDP problem contains equalities on symmetry 
        # constraints encoded as pairs of inequalities
        self.F = [0] * (self.n_vars+1)
        for i in range(self.n_vars+1):
            self.F[i] = []
    
        block_index=1
        self.__generate_moment_matrix(monomials, block_index)
        
        # Objective function
        self.obj_facvar=self.__get_facvar(obj)
    
        print('Transforming %d equalities to %d inequalities...' % (len(equalities), 2*len(equalities)))
        for eq in equalities:
            inequalities.append(eq)
            inequalities.append(-eq)
    
        print('Processing %d inequalities...' % len(inequalities))
        block_index = self.__process_inequalities(inequalities, monomials, block_index, order)
    
        block_index+=1        
        for i in range(self.n_monomials):
            for j in range(i,self.n_monomials):
                k=self.__index2linear(i,j)
                e = entry(block_index, i+1, j+1, 1)
                self.F[k].append(e)
        
        self.block_struct.append(self.n_monomials)
        
        
    def write_to_sdpa(self, filename):
        """Writes the SDP relaxation to SDPA format.
        
        Arguments:
        filename -- the name of the file. It must have the suffix ".dat-s"
        """

        f = open(filename,'w')
        f.write('"file '+filename+' generated by ncpol2sdpa"\n')
        print('writing problem in '+filename+'...')
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
        
        print('done.')
        f.close()
