# -*- coding: utf-8 -*-
"""
The module contains helper functions to work with PICOS.

Created on Wed Dec 10 18:33:34 2014

@author: Peter Wittek
"""
from __future__ import print_function

def convert_to_picos_extra_moment_matrix(sdpRelaxation):
    """Convert an SDP relaxation to a PICOS problem, returning the moment
    matrix with positive semidefinite imposed, and an extra copy of the moment
    matrix with no constraints on it.

    :param sdpRelaxation: The SDP relaxation to convert.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.

    :returns: :class:`picos.Problem`.
    """
    import picos as pic
    import cvxopt as cvx
    P = pic.Problem()
    block_size = sdpRelaxation.block_struct[0]
    X = P.add_variable('X', (block_size, block_size))
    Y = P.add_variable('Y', (block_size, block_size))
    row_offset = 0
    for block_size in sdpRelaxation.block_struct:
        x, Ix, Jx = [], [], []
        c, Ic, Jc = [], [], []
        for i, row in enumerate(sdpRelaxation.F_struct.rows[row_offset:row_offset+block_size**2]):
            for j, column in enumerate(row):
                if column > 0:
                    x.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                    Ix.append(i)
                    Jx.append(column-1)
                    i0 = (i/block_size)+(i%block_size)*block_size
                    if i != i0:
                        x.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                        Ix.append(i0)
                        Jx.append(column-1)

                else:
                    c.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                    #Note that a constant term can only possibly occur in a
                    #top-left corner
                    Ic.append(0)
                    Jc.append(0)
        permutation = cvx.spmatrix(x, Ix, Jx)
        constant = cvx.spmatrix(c, Ic, Jc, (block_size, block_size))
        constraint = X#.copy()
        for k in constraint.factors:
            constraint.factors[k] = permutation
        constraint._size = (block_size, block_size)
        P.add_constraint(constant + constraint>>0)
        if block_size == sdpRelaxation.block_struct[0]:
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
    return P, X, Y

def convert_to_picos(sdpRelaxation):
    """Convert an SDP relaxation to a PICOS problem such that the exported
    .dat-s file is extremely sparse, there is not penalty imposed in terms of
    SDP variables or number of constraints. This conversion can be used for
    imposing extra constraints on the moment matrix, such as partial transpose.

    :param sdpRelaxation: The SDP relaxation to convert.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.

    :returns: :class:`picos.Problem`.
    """
    import picos as pic
    import cvxopt as cvx
    P = pic.Problem()
    block_size = sdpRelaxation.block_struct[0]
    X = P.add_variable('X', (block_size, block_size))
    row_offset = 0
    for block_size in sdpRelaxation.block_struct:
        x, Ix, Jx = [], [], []
        c, Ic, Jc = [], [], []
        for i, row in enumerate(sdpRelaxation.F_struct.rows[row_offset:row_offset+block_size**2]):
            for j, column in enumerate(row):
                if column > 0:
                    x.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                    Ix.append(i)
                    Jx.append(column-1)
                    i0 = (i/block_size)+(i%block_size)*block_size
                    if i != i0:
                        x.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                        Ix.append(i0)
                        Jx.append(column-1)

                else:
                    c.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                    #Note that a constant term can only possibly occur in a
                    #top-left corner
                    Ic.append(0)
                    Jc.append(0)
        permutation = cvx.spmatrix(x, Ix, Jx)
        constant = cvx.spmatrix(c, Ic, Jc, (block_size, block_size))
        constraint = X.copy()
        for k in constraint.factors:
            constraint.factors[k] = permutation
        constraint._size = (block_size, block_size)
        P.add_constraint(constant + constraint>>0)
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
