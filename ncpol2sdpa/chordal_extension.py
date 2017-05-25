# -*- coding: utf-8 -*-
"""
The module contains helper functions to calculate the chordal extension of the
correlative sparsity pattern matrix. It is largely based on the MATLAB version
in SparsePOP.

Created on Sun Nov 30 15:01:13 2014

@author: Peter Wittek
"""
from __future__ import division, print_function
import numpy as np
import random
from .nc_utils import get_support, flatten


def sliding_cliques(k, n):
    one_sides = []
    cliques = []
    for i in range(n-int(k/2)+1):
        side = [0] * n
        for j in range(int(k/2)):
            side[i+j] = 1
        one_sides.append(side)
    for side1 in one_sides:
        for side2 in one_sides:
            clique = list(side1)
            clique.extend(side2)
            cliques.append(clique)
    return cliques


def _generate_clique(variables, obj, inequalities, equalities,
                     momentinequalities, momentequalities):
    n_dim = len(variables)
    rmat = np.eye(n_dim)
    # Objective: if x_i & x_j in monomial, rmat_ij = rand
    if obj is not None:
        for support in get_support(variables, obj):
            nonzeros = np.nonzero(support)[0]
            value = random.random()
            for i in nonzeros:
                for j in nonzeros:
                    rmat[i, j] = value
    # Constraints: if x_i & x_j in support, rmat_ij = rand
    for polynomial in flatten([inequalities, equalities, momentinequalities,
                               momentequalities]):
        support = np.any(get_support(variables, polynomial), axis=0)
        nonzeros = np.nonzero(support)[0]
        value = random.random()
        for i in nonzeros:
            for j in nonzeros:
                rmat[i, j] = value
    rmat = rmat + 5*n_dim*np.eye(n_dim)
    # TODO: approximate minimum degree ordering should go before the Cholesky
    # decomposition
    R = np.linalg.cholesky(rmat).T
    R[np.nonzero(R)] = 1
    remaining_indices = [0]
    for i in range(1, n_dim):
        check_set = R[i, i:n_dim]
        one = np.nonzero(check_set)[0]
        n_ones = len(one)
        clique_result = np.dot(R[:i, i:n_dim], check_set.T)
        test = False
        for t in clique_result:
            if t == n_ones:
                test = True
                break
        if not test:
            remaining_indices.append(i)
    clique_set = R[remaining_indices, :]
    return clique_set


def _generate_clique_alt(variables, obj, inequalities, equalities,
                         momentinequalities, momentequalities):
    n_dim = len(variables)
    rmat = spmatrix(1.0, range(n_dim), range(n_dim))
    if obj is not None:
        for support in get_support(variables, obj):
            nonzeros = np.nonzero(support)[0]
            value = random.random()
            for i in nonzeros:
                for j in nonzeros:
                    rmat[int(i), int(j)] = value
    for polynomial in flatten([inequalities, equalities, momentinequalities,
                               momentequalities]):
        support = np.any(get_support(variables, polynomial), axis=0)
        nonzeros = np.nonzero(support)[0]
        value = random.random()
        for i in nonzeros:
            for j in nonzeros:
                rmat[int(i), int(j)] = value
    rmat = rmat + 5*n_dim*spmatrix(1.0, range(n_dim), range(n_dim))
    # compute symbolic factorization using AMD ordering
    symb = cp.symbolic(rmat, p=amd.order)
    ip = symb.p
    cliques = symb.cliques()
    R = np.zeros((len(cliques), n_dim))
    for i, clique in enumerate(cliques):
        for j in range(len(clique)):
            R[i, ip[cliques[i][j]]] = 1
    return R


def find_clique_index(variables, polynomial, clique_set):
    support = np.any(get_support(variables, polynomial), axis=0)
    support[np.nonzero(support)[0]] = 1
    for i, clique in enumerate(clique_set):
        if np.dot(support, clique) == len(np.nonzero(support)[0]):
            return i
    return -1


def find_variable_cliques(variables, objective=None, inequalities=None,
                          equalities=None, momentinequalities=None,
                          momentequalities=None):
    if objective is None and inequalities is None and equalities is None and \
            momentinequalities is None and momentequalities is None:
        raise Exception("There is nothing to extract the chordal structure " +
                        "from!")
    clique_set = generate_clique(variables, objective, inequalities,
                                 equalities, momentinequalities,
                                 momentequalities)
    variable_sets = []
    for clique in clique_set:
        variable_sets.append([variables[i] for i in np.nonzero(clique)[0]])
    return variable_sets


try:
    from cvxopt import spmatrix, amd
    import chompack as cp
    generate_clique = _generate_clique_alt
except ImportError:
    generate_clique = _generate_clique
