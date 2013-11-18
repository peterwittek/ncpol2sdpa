# -*- coding: utf-8 -*-
"""
The module contains helper functions to work with noncommutative polynomials 
and Hamiltonians.

Created on Thu May  2 16:03:05 2013

@author: Peter Wittek
"""
from math import floor
from sympy.core import S, Symbol, Pow
from sympy.physics.quantum.operator import HermitianOperator, Operator
from sympy.physics.quantum.dagger import Dagger

def count_ncmonomials(variables, monomials, degree):
    """Given a list of monomials, it counts those that have a certain degree,
    or less. The function is useful when certain monomials were eliminated
    from the basis.
    
    Arguments:
    variables -- the noncommutative variables making up the monomials
    monomials -- the list of monomials (the monomial basis)
    degree -- maximum degree to count

    Returns the count of appropriate monomials.
    """    
    ncmoncount = 0
    for monomial in monomials:
        if ncdegree(variables, monomial) <= degree:
            ncmoncount += 1
        else:
            break
    return ncmoncount
                    
def fastSubstitute(monomial, oldSub, newSub):
    oldFactors = oldSub.as_coeff_mul()[1]
    factors = monomial.as_coeff_mul()[1]
    newVarList = []
    newMonomial = 1
    match = False
    leftRemainder = 1
    rightRemainder = 1
    for i in range(len(factors)-len(oldFactors)+1):
        for j in range(len(oldFactors)):
            if isinstance(factors[i+j],Operator) and isinstance(oldFactors[j],Operator) and factors[i+j] != oldFactors[j]:
                break
            if isinstance(factors[i+j],Pow):
                oldDeg = 1
                oldBase = 1
                if isinstance(oldFactors[j],Pow):
                    oldBase = oldFactors[j].base
                    oldDeg = oldFactors[j].exp
                else:
                    oldBase = oldFactors[j]
                if oldBase != factors[i+j].base:
                    break
                if oldDeg > factors[i+j].exp:
                    break
                if oldDeg < factors[i+j].exp:
                    if j!=len(oldFactors)-1:
                        if j!=0:
                            break
                        else:
                            leftRemainder = oldBase**(factors[i+j].exp - oldDeg)
                    else:
                        rightRemainder = oldBase**(factors[i+j].exp - oldDeg)
            if isinstance(factors[i+j],Operator) and isinstance(oldFactors[j],Pow):
                break
        else:
            match = True
        if not match:
            newVarList.append(factors[i])
        else:
            newMonomial = monomial.as_coeff_mul()[0]
            for var in newVarList:
                newMonomial *= var
            newMonomial *= leftRemainder * newSub * rightRemainder
            for j in range(i+len(oldFactors), len(factors)):
                newMonomial *= factors[j]
            break
    else:
        return monomial
    return newMonomial



def generate_ncvariables(n_vars):
    """Generates a number of noncommutative variables
    
    Arguments:
    n_vars -- the number of variables
    
    Returns a list of noncommutative variables
    """

    variables = [0]*n_vars
    for i in range(n_vars):
        variables[i] = HermitianOperator('X%s' % i)
    return variables

def get_ncmonomials(variables, degree):
    """Generates all noncommutative monomials up to a degree

    Arguments:
    variables -- the noncommutative variables to generate monomials from
    degree -- the maximum degree

    Returns a list of monomials
    """
    if not variables:
        return [S.One]
    else:
        _variables = variables[:]
        _variables.insert(0, 1)
        ncmonomials = [ S.One ]
        for _ in range(degree):
            temp = []
            for var in _variables:
                for new_var in ncmonomials:
                    temp.append(var * new_var)
                    if var != 1 and not var.is_hermitian:
                        temp.append(Dagger(var) * new_var)
            ncmonomials = unique(temp[:])
        return ncmonomials

def get_neighbors(index, lattice_dimension):
    """Get the neighbors of an operator in a lattice.
    
    Arguments:
    index -- linear index of operator
    lattice_dimension -- the size of the 2D lattice in either dimension
    
    Returns a list of neighbors in linear index.
    """
        
    neighbors = []
    coords = _linear2lattice(index, lattice_dimension)
    if coords[0] > 1:
        neighbors.append(index - 1)
    if coords[0] < lattice_dimension - 1:
        neighbors.append(index + 1)
    if coords[1] > 1:
        neighbors.append(index - lattice_dimension)
    if coords[1] < lattice_dimension - 1:
        neighbors.append(index + lattice_dimension)
    return neighbors

def _linear2lattice(index, dimension):
    """Helper function to map linear coordinates to a lattice."""
    coords = [0, 0]
    coords[0] = index % dimension
    coords[1] = int(floor(index/dimension))
    return coords  

def ncdegree(variables, polynomial):
    """Returns the degree of a noncommutative polynomial."""
    degree = 0
    if isinstance(polynomial, (int, float, complex)):
        return degree
    
    for element in polynomial.as_coefficients_dict():
        subdegree = 0
        for monomial in element.as_coeff_mul()[1]:
            for var in variables:
                subdegree += monomial.as_coeff_exponent(var)[1]
        if subdegree > degree:
            degree = subdegree
    return degree
  

def unique(seq): 
    """Helper function to include only unique monomials in a basis."""
    seen = {}
    result = []
    for item in seq:
        marker = item
        if marker in seen: 
            continue
        seen[marker] = 1
        result.append(item)
    return result
