# -*- coding: utf-8 -*-
"""
The module converts a noncommutative optimization problem provided in SymPy 
format to a PICOS semidefinite programming problem.

Created on Sun May 26 15:06:17 2013

@author: Peter Wittek
"""
from math import ceil
import picos
from sympy import S
from sympy.physics.quantum.dagger import Dagger
from ncutils import get_ncmonomials, count_ncmonomials, ncdegree

def _apply_substitions(monomial, monomial_substitution):
    """Helper function to remove monomials from the basis."""
    for lhs, rhs in monomial_substitution.iteritems():
        monomial = monomial.subs(lhs, rhs)
    return monomial

def get_relaxation(variables, obj, inequalities, equalities, 
                   monomial_substitution, order):
    """Gets the SDP relaxation of a noncommutative polynomial optimization
    problem.
    
    Arguments:
    variables -- the noncommutative variables used in the problem
    obj -- the objective function
    inequalities -- list of inequality constraints
    equalities -- list of equality constraints
    monomial_substitutions -- monomials that can be replaced 
                              (e.g., idempotent variables)
    order -- the order of the relaxation
    
    Returns an SDP problem in PICOS format.
    """
    
    monomials = get_ncmonomials(variables, order)
    for monomial in monomial_substitution.keys():
        monomials.remove(monomial)
    moncount = count_ncmonomials(variables, monomials, order)
    prob = picos.Problem()
    M = prob.add_variable('M', (moncount, moncount), vtype = 'symmetric')
    
    prob.add_constraint(M >> 0)
    prob.add_constraint(M[0, 0] == 1)
    monomial_dictionary = {}
    for i in xrange(moncount):
        for j in xrange(i, moncount):
            monomial = monomials[i] * Dagger(monomials[j])
            monomial = _apply_substitions(monomial, monomial_substitution)
            if monomial.as_coeff_Mul()[0] < 0:
                monomial = -monomial
            if monomial in monomial_dictionary:
                indices = monomial_dictionary[monomial]
                prob.add_constraint(-M[indices[0], indices[1]]+M[i, j] == 0)
            else:
                monomial_dictionary[monomial] = [i, j]
   
    prob.set_objective('min', _sympy2picos(M, monomial_substitution,
                                          monomial_dictionary, obj))
    
    ineqmoncount = count_ncmonomials(variables, monomials, order - 1)
    Mineq = [0] * len(inequalities)
    k = 0
    print 'Processing ', len(inequalities), 'inequalitites...'
    for ineq in inequalities:
        Mineq[k] = prob.add_variable('Mineq%s' % k, 
                                     (ineqmoncount, ineqmoncount), 
                                     vtype = 'symmetric')
        prob.add_constraint(Mineq[k] >> 0)
        for i in xrange(ineqmoncount):
            for j in xrange(i, ineqmoncount):
                polynomial = monomials[i] * ineq * Dagger(monomials[j])
                ineqrelax = _sympy2picos(M, monomial_substitution, 
                                        monomial_dictionary, polynomial)
                prob.add_constraint(Mineq[k][i, j] == ineqrelax)
        k += 1
    
    print 'Processing ', len(equalities), 'equalitites...'
    Meq = [0] * len(equalities)
    for k in xrange(len(equalities)):
        adjusted_order = order - int(ceil(0.5 * ncdegree(variables, 
                                                         equalities[k])))
        eqmoncount = count_ncmonomials(variables, monomials, adjusted_order)
        Meq[k] = prob.add_variable('Meq%s' % k,
                                   (eqmoncount, eqmoncount), 
                                   vtype = 'symmetric')
        prob.add_constraint(Meq[k] == 0)
        for i in xrange(eqmoncount):
            for j in xrange(i, eqmoncount):
                polynomial = monomials[i] * equalities[k] * Dagger(monomials[j])
                eqrelax = _sympy2picos(M, monomial_substitution, 
                                      monomial_dictionary, polynomial)
                prob.add_constraint(Meq[k][i, j] == eqrelax)
    return prob

def _sympy2picos(M, monomial_substitution, monomial_dictionary, polynomial):
    """Helper function that converts SymPy polynomials to PICOS polynomials."""
    picos_pol = 0
    for element in polynomial.expand().as_coeff_factors()[1]:
        coeff = 1.0
        monomial = S.One
        for var in element.as_coeff_mul()[1]:
            if not var.is_Number:
                monomial = monomial * var
            else:
                coeff = float(var)
        coeff = float(element.as_coeff_mul()[0]) * coeff
        monomial = _apply_substitions(monomial, monomial_substitution)
        if monomial.as_coeff_Mul()[0] < 0:
            monomial = -monomial
            coeff = -1.0 * coeff
        if monomial in monomial_dictionary:
            indices = monomial_dictionary[monomial]
        else:
            monomial = Dagger(monomial)
            monomial = _apply_substitions(monomial, monomial_substitution)
            if monomial.as_coeff_Mul()[0] < 0:
                monomial = -monomial
                coeff = -1.0*coeff
            indices = monomial_dictionary[monomial]
            indices[0], indices[1] = indices[1], indices[0]
        picos_pol += coeff * M[indices[0], indices[1]]
    return picos_pol
