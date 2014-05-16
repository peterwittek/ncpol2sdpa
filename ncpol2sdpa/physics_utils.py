# -*- coding: utf-8 -*-
"""
The module contains helper functions for physics applications.

Created on Fri May 16 14:27:47 2014

@author: Peter Wittek
"""
from sympy.physics.quantum.dagger import Dagger
from math import floor


def get_neighbors(index, lattice_dimension, periodic=False):
    """Get the neighbors of an operator in a lattice.

    Arguments:
    index -- linear index of operator
    lattice_dimension -- the size of the 2D lattice in either dimension

    Returns a list of neighbors in linear index.
    """

    neighbors = []
    coords = _linear2lattice(index, lattice_dimension)
    # if coords[0] > 0:
    #    neighbors.append(index - 1)
    if coords[0] < lattice_dimension - 1:
        neighbors.append(index + 1)
    elif periodic:
        neighbors.append(index - lattice_dimension + 1)
    # if coords[1] > 0:
    #    neighbors.append(index - lattice_dimension)
    if coords[1] < lattice_dimension - 1:
        neighbors.append(index + lattice_dimension)
    elif periodic:
        neighbors.append(index - (lattice_dimension - 1) * lattice_dimension)
    return neighbors


def _linear2lattice(index, dimension):
    """Helper function to map linear coordinates to a lattice."""
    coords = [0, 0]
    coords[0] = index % dimension
    coords[1] = int(floor(index / dimension))
    return coords


def bosonic_constraints(a):
    n_vars = len(a)
    monomial_substitutions = {}
    equalities = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            # [a_i,a_jT] = 0 for i\neq j
            monomial_substitutions[Dagger(a[j]) * a[i]] = a[i] * Dagger(a[j])
            # [a_i, a_j] = 0
            monomial_substitutions[a[j] * a[i]] = a[i] * a[j]
            # [a_iT, a_jT] = 0
            monomial_substitutions[Dagger(a[j]) * Dagger(a[i])] = \
                Dagger(a[i]) * Dagger(a[j])

    # [a_i,a_iT]=1
    for i in range(n_vars):
        equalities.append(a[i] * Dagger(a[i]) - Dagger(a[i]) * a[i] - 1.0)

    return monomial_substitutions, equalities


def fermionic_constraints(a):
    n_vars = len(a)
    monomial_substitutions = {}
    equalities = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            # {a_i,a_jT} = 0 for i\neq j
            monomial_substitutions[Dagger(a[j]) * a[i]] = - a[i] * Dagger(a[j])
            # {a_i, a_j} = 0
            monomial_substitutions[a[j] * a[i]] = - a[i] * a[j]
            # {a_iT, a_jT} = 0
            monomial_substitutions[Dagger(a[j]) * Dagger(a[i])] = \
                -Dagger(a[i]) * Dagger(a[j])

    # {a_i,a_iT} = 1
    for i in range(n_vars):
        equalities.append(a[i] * Dagger(a[i]) + Dagger(a[i]) * a[i] - 1.0)

    return monomial_substitutions, equalities
