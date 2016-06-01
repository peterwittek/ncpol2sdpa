# -*- coding: utf-8 -*-
"""
The module contains helper functions to work with PICOS.

Created on Wed Dec 10 18:33:34 2014

@author: Peter Wittek
"""
from __future__ import division, print_function
import numpy as np


def solve_with_cvxopt(sdp, solverparameters=None):
    """Helper function to convert the SDP problem to PICOS
    and call CVXOPT solver, and parse the output.

    :param sdp: The SDP relaxation to be solved.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    """
    P = convert_to_picos(sdp)
    P.set_option("solver", "cvxopt")
    P.set_option("verbose", sdp.verbose)
    if solverparameters is not None:
        for key, value in solverparameters.items():
            P.set_option(key, value)
    solution = P.solve()
    x_mat = [np.array(P.get_valued_variable('X'))]
    y_mat = [np.array(P.get_constraint(i).dual)
             for i in range(len(P.constraints))]
    return -solution["cvxopt_sol"]["primal objective"] + \
        sdp.constant_term, \
        -solution["cvxopt_sol"]["dual objective"] + \
        sdp.constant_term, \
        x_mat, y_mat, solution["status"]


def convert_to_picos(sdp, duplicate_moment_matrix=False):
    """Convert an SDP relaxation to a PICOS problem such that the exported
    .dat-s file is extremely sparse, there is not penalty imposed in terms of
    SDP variables or number of constraints. This conversion can be used for
    imposing extra constraints on the moment matrix, such as partial transpose.

    :param sdp: The SDP relaxation to convert.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    :param duplicate_moment_matrix: Optional parameter to add an unconstrained
                                    moment matrix to the problem with the same
                                    structure as the moment matrix with the PSD
                                    constraint.
    :type duplicate_moment_matrix: bool.

    :returns: :class:`picos.Problem`.
    """
    import picos as pic
    import cvxopt as cvx
    P = pic.Problem(verbose=sdp.verbose)
    block_size = sdp.block_struct[0]
    if sdp.F.dtype == np.float64:
        X = P.add_variable('X', (block_size, block_size), vtype="symmetric")
        if duplicate_moment_matrix:
            Y = P.add_variable('Y', (block_size, block_size), vtype="symmetric")
    else:
        X = P.add_variable('X', (block_size, block_size), vtype="hermitian")
        if duplicate_moment_matrix:
            Y = P.add_variable('X', (block_size, block_size), vtype="hermitian")
    row_offset = 0
    theoretical_n_vars = sdp.block_struct[0]**2
    eq_block_start = sdp.constraint_starting_block+sdp._n_inequalities
    block_idx = 0
    while (block_idx < len(sdp.block_struct)):
        block_size = sdp.block_struct[block_idx]
        x, Ix, Jx = [], [], []
        c, Ic, Jc = [], [], []
        for i, row in enumerate(sdp.F.rows[row_offset:row_offset +
                                                     block_size**2]):
            for j, column in enumerate(row):
                if column > 0:
                    x.append(sdp.F.data[row_offset+i][j])
                    Ix.append(i)
                    Jx.append(column-1)
                    i0 = (i//block_size)+(i % block_size)*block_size
                    if i != i0:
                        x.append(sdp.F.data[row_offset+i][j])
                        Ix.append(i0)
                        Jx.append(column-1)

                else:
                    c.append(sdp.F.data[row_offset+i][j])
                    Ic.append(i%block_size)
                    Jc.append(i//block_size)
        permutation = cvx.spmatrix(x, Ix, Jx, (block_size**2,
                                               theoretical_n_vars))
        constant = cvx.spmatrix(c, Ic, Jc, (block_size, block_size))
        if duplicate_moment_matrix:
            constraint = X
        else:
            constraint = X.copy()
        for k in constraint.factors:
            constraint.factors[k] = permutation
        constraint._size = (block_size, block_size)
        if block_idx < eq_block_start:
            P.add_constraint(constant + constraint >> 0)
        else:
            P.add_constraint(constant + constraint == 0)
            row_offset += block_size**2
            block_idx += 1
        if duplicate_moment_matrix and \
                block_size == sdp.block_struct[0]:
            for k in Y.factors:
                Y.factors[k] = permutation
        row_offset += block_size**2
        block_idx += 1
    if len(np.nonzero(sdp.obj_facvar)[0]) > 0:
        x, Ix, Jx = [], [], []
        for k, val in enumerate(sdp.obj_facvar):
            if val != 0:
                x.append(val)
                Ix.append(0)
                Jx.append(k)
        permutation = cvx.spmatrix(x, Ix, Jx)
        objective = X.copy()
        for k in objective.factors:
            objective.factors[k] = permutation
        objective._size = (1, 1)
        P.set_objective('min', objective)
    return P
