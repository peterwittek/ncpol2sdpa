# -*- coding: utf-8 -*-
"""
The module contains helper functions to calculate the chordal extension of the
correlative sparsity pattern matrix. It is largely based on the MATLAB version
in SparsePOP.

Created on Sun Nov 30 15:01:13 2014

@author: Peter Wittek
"""
import random
import numpy as np
from .nc_utils import get_support, flatten

def generate_clique(variables, obj, inequalities, equalities):
    n_dim = len(variables)
    rmat = np.eye(n_dim)
    #Objective: if x_i & x_j in monomial, rmat_ij = rand
    for support in get_support(variables, obj):
        nonzeros = np.nonzero(support)[0]
        value =  random.random()
        for i in nonzeros:
            for j in nonzeros:
                rmat[i,j] = value
    #Constraints: if x_i & x_j in support, rmat_ij = rand
    for polynomial in flatten([inequalities, equalities]):
        support = np.any(get_support(variables, polynomial), axis=0)
        nonzeros = np.nonzero(support)[0]
        value =  random.random()
        for i in nonzeros:
            for j in nonzeros:
                rmat[i,j] = value
    rmat = rmat + 5*n_dim*np.eye(n_dim);
    #TODO: approximate minimum degree ordering should go before the Cholesky
    #decomposition
    R = np.linalg.cholesky(rmat).T
    R[np.nonzero(R)] = 1
    remaining_indices = [0]
    for i in range(1, n_dim):
        check_set = R[i,i:n_dim]
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

def find_clique_index(variables, polynomial, clique_set):
    support = np.any(get_support(variables, polynomial), axis=0)
    support[np.nonzero(support)[0]] = 1
    for i, clique in enumerate(clique_set):
        if np.dot(support, clique) == len(np.nonzero(support)[0]):
            return i
    return -1
