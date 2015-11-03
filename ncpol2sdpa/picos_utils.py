# -*- coding: utf-8 -*-
"""
The module contains helper functions to work with PICOS.

Created on Wed Dec 10 18:33:34 2014

@author: Peter Wittek
"""
from __future__ import print_function
import numpy as np


def solve_with_cvxopt(sdpRelaxation, solverparameters=None):
    """Helper function to convert the SDP problem to PICOS
    and call CVXOPT solver, and parse the output.

    :param sdpRelaxation: The SDP relaxation to be solved.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.
    """
    P = convert_to_picos(sdpRelaxation)
    P.set_option("solver", "cvxopt")
    P.set_option("verbose", sdpRelaxation.verbose)
    if solverparameters is not None:
        for key, value in solverparameters.items():
            P.set_option(key, value)
    solution = P.solve()
    x_mat = [np.array(P.get_valued_variable('X'))]
    y_mat = [np.array(P.get_constraint(i).dual)
             for i in range(len(P.constraints))]
    return -solution["cvxopt_sol"]["primal objective"] + \
        sdpRelaxation.constant_term, \
        -solution["cvxopt_sol"]["dual objective"] + \
        sdpRelaxation.constant_term, \
        x_mat, y_mat, solution["status"]


def convert_to_picos(sdpRelaxation, duplicate_moment_matrix=False):
    """Convert an SDP relaxation to a PICOS problem such that the exported
    .dat-s file is extremely sparse, there is not penalty imposed in terms of
    SDP variables or number of constraints. This conversion can be used for
    imposing extra constraints on the moment matrix, such as partial transpose.

    :param sdpRelaxation: The SDP relaxation to convert.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.
    :param duplicate_moment_matrix: Optional parameter to add an unconstrained
                                    moment matrix to the problem with the same
                                    structure as the moment matrix with the PSD
                                    constraint.
    :type duplicate_moment_matrix: bool.

    :returns: :class:`picos.Problem`.
    """
    import picos as pic
    import cvxopt as cvx
    P = pic.Problem(verbose=sdpRelaxation.verbose)
    block_size = sdpRelaxation.block_struct[0]
    if sdpRelaxation.F_struct.dtype == np.float64:
        X = P.add_variable('X', (block_size, block_size), vtype="symmetric")
        if duplicate_moment_matrix:
            Y = P.add_variable('Y', (block_size, block_size), vtype="symmetric")
    else:
        X = P.add_variable('X', (block_size, block_size), vtype="hermitian")
        if duplicate_moment_matrix:
            Y = P.add_variable('X', (block_size, block_size), vtype="hermitian")
    row_offset = 0
    theoretical_n_vars = sdpRelaxation.block_struct[0]**2
    for block_size in sdpRelaxation.block_struct:
        x, Ix, Jx = [], [], []
        c, Ic, Jc = [], [], []
        for i, row in enumerate(sdpRelaxation.F_struct.rows[row_offset:
                                                            row_offset +
                                                            block_size**2]):
            for j, column in enumerate(row):
                if column > 0:
                    x.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                    Ix.append(i)
                    Jx.append(column-1)
                    i0 = (i//block_size)+(i % block_size)*block_size
                    if i != i0:
                        x.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                        Ix.append(i0)
                        Jx.append(column-1)

                else:
                    c.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                    # Note that a constant term can only possibly occur in a
                    # top-left corner
                    Ic.append(0)
                    Jc.append(0)
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

        P.add_constraint(constant + constraint >> 0)
        if duplicate_moment_matrix and \
                block_size == sdpRelaxation.block_struct[0]:
            for k in Y.factors:
                Y.factors[k] = permutation
        row_offset += block_size**2
    x, Ix, Jx = [], [], []
    for k, val in enumerate(sdpRelaxation.obj_facvar):
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
