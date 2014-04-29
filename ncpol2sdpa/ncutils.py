# -*- coding: utf-8 -*-
"""
The module contains helper functions to work with noncommutative polynomials
and Hamiltonians.

Created on Thu May  2 16:03:05 2013

@author: Peter Wittek
"""
from math import floor
from sympy.core import S, Symbol, Pow, Number
from sympy.physics.quantum.operator import HermitianOperator, Operator
from sympy.physics.quantum.dagger import Dagger


def count_ncmonomials(monomials, degree):
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
        if ncdegree(monomial) <= degree:
            ncmoncount += 1
        else:
            break
    return ncmoncount


def fast_substitute(monomial, old_sub, new_sub):
    """Experimental fast substitution routine that considers only restricted
    cases of noncommutative algebras. In rare cases, it fails to find a
    substitution. Use it with proper testing.

    Arguments:
    monomial -- the monomial with parts need to be substituted
    old_sub -- the part to be replaced
    new_sub -- the replacement
    """
    old_factors = old_sub.as_coeff_mul()[1]
    factors = monomial.as_coeff_mul()[1]
    new_var_list = []
    new_monomial = 1
    match = False
    if isinstance(monomial, Number):
        return monomial
    left_remainder = 1
    right_remainder = 1
    for i in range(len(factors) - len(old_factors) + 1):
        for j in range(len(old_factors)):
            if isinstance(factors[i + j], Symbol) and \
              (not isinstance(old_factors[j], Operator) or \
              (isinstance(old_factors[j], Symbol) and \
              factors[i + j] != old_factors[j])):
                break
            if isinstance(factors[i + j], Operator) and \
              isinstance(old_factors[j], Operator) and \
              factors[i + j] != old_factors[j]:
                break
            if isinstance(factors[i + j], Pow):
                old_degree = 1
                old_base = 1
                if isinstance(old_factors[j], Pow):
                    old_base = old_factors[j].base
                    old_degree = old_factors[j].exp
                else:
                    old_base = old_factors[j]
                if old_base != factors[i + j].base:
                    break
                if old_degree > factors[i + j].exp:
                    break
                if old_degree < factors[i + j].exp:
                    if j != len(old_factors) - 1:
                        if j != 0:
                            break
                        else:
                            left_remainder = old_base ** (
                                factors[i + j].exp - old_degree)
                    else:
                        right_remainder = old_base ** (
                            factors[i + j].exp - old_degree)
            if isinstance(factors[i + j], Operator) and \
              isinstance(old_factors[j], Pow):
                break
        else:
            match = True
        if not match:
            new_var_list.append(factors[i])
        else:
            new_monomial = monomial.as_coeff_mul()[0]
            for var in new_var_list:
                new_monomial *= var
            new_monomial *= left_remainder * new_sub * right_remainder
            for j in range(i + len(old_factors), len(factors)):
                new_monomial *= factors[j]
            break
    else:
        return monomial
    return new_monomial


def generate_ncvariables(n_vars):
    """Generates a number of noncommutative variables

    Arguments:
    n_vars -- the number of variables

    Returns a list of noncommutative variables
    """

    variables = [0] * n_vars
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
        ncmonomials = [S.One]
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
    coords[1] = int(floor(index / dimension))
    return coords


def get_variables_of_polynomial(polynomial):
    """Returns the degree of a noncommutative polynomial."""
    if isinstance(polynomial, (int, float, complex)):
        return []
    result = []
    for monomial in polynomial.as_coefficients_dict():
        for variable in monomial.as_coeff_mul()[1]:
            if isinstance(variable, Pow):
                result.append(variable.base)
            else:
                result.append(variable)
    return result


def ncdegree(polynomial):
    """Returns the degree of a noncommutative polynomial."""
    degree = 0
    if isinstance(polynomial, (int, float, complex)):
        return degree
    for monomial in polynomial.as_coefficients_dict():
        subdegree = 0
        for variable in monomial.as_coeff_mul()[1]:
            if isinstance(variable, Pow):
                subdegree += variable.exp
            else:
                subdegree += 1
        if subdegree > degree:
            degree = subdegree
    return degree

def pick_monomials_up_to_degree(
        monomial_blocks, monomial_block_index_list, degree):
    """Collect monomials up to a given degree.
    """
    ordered_monomials = []
    for monomial_block_index in monomial_block_index_list:
        for deg in range(degree + 1):
            ordered_monomials.extend(
                pick_monomials_of_degree(
                    monomial_blocks[monomial_block_index], deg
                )
            )
    return ordered_monomials


def pick_monomials_of_degree(monomials, degree):
    """Collect all monomials up of a given degree.
    """
    selected_monomials = []
    for monomial in monomials:
        # Expectation value of the identity operator for the block
        if degree == 0 and monomial.is_commutative:
            selected_monomials.append(monomial)
        elif ncdegree(monomial) == degree:
            selected_monomials.append(monomial)
    return selected_monomials

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
