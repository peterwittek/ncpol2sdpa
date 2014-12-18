# -*- coding: utf-8 -*-
"""
The module contains helper functions to work with PICOS.

Created on Wed Dec 10 18:33:34 2014

@author: Peter Wittek
"""
from .sdpa_utils import convert_row_to_sdpa_index

def _row_to_affine_expression(row_vector, F_struct, row_offsets,
                             block_of_last_moment_matrix, block_struct, X):
    """Helper function to create an affine expression based on the variables
    in the moment matrices.
    """
    if row_vector.getnnz() > 0:
        affine_expression = 0
        row, columns = row_vector.nonzero()
        for k in columns:
            if k == 0:
                affine_expression += row_vector[row[0], k]
            else:
                row0 = F_struct[:row_offsets[block_of_last_moment_matrix+1],
                                k].nonzero()[0][0]
                block, i, j = convert_row_to_sdpa_index(block_struct,
                                                        row_offsets, row0)
                affine_expression += row_vector[row[0], k] * X[block][i, j]
        return affine_expression
    else:
        return None

def _objective_to_affine_expression(objective, F_struct, row_offsets,
                                   block_of_last_moment_matrix, block_struct,
                                   X):
    """Helper function to create an affine expression based on the variables
    in the moment matrices from the dense vector describing the objective.
    """
    affine_expression = 0
    for k, v in enumerate(objective):
        if v != 0:
            row0 = F_struct[0:row_offsets[block_of_last_moment_matrix+1],
                            k+1].nonzero()[0][0]
            block, i, j = convert_row_to_sdpa_index(block_struct, row_offsets,
                                                    row0)
            affine_expression += v * X[block][i, j]
    return affine_expression

def convert_to_picos_for_export(sdpRelaxation):
    """Convert an SDP relaxation to a PICOS problem in a way that PICOS cannot
    solve. The advantage is that the exported .dat-s file is extremely sparse,
    there is not penalty imposed in terms of SDP variables or number of
    constraints. This conversion can be used for imposing extra constraints
    on the moment matrix, such as partial transpose.

    :param sdpRelaxation: The SDP relaxation to convert.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.

    :returns: :class:`picos.Problem`.
    """
    import picos as pic
    import cvxopt as cvx
    P = pic.Problem()
    block_size = sdpRelaxation.block_struct[0]
    X = P.add_variable('X', (block_size, block_size), 'symmetric')
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
                else:
                    c.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                    #Note that a constant term can only possibly occur in a
                    #top-left corner
                    Ic.append(0)
                    Jc.append(0)
        permutation = cvx.spmatrix(x, Ix, Jx)
        constant = cvx.spmatrix(c, Ic, Jc, (block_size,block_size))
        constraint = X.copy()
        for k in constraint.factors:
            constraint.factors[k]=permutation
        constraint._size=(block_size, block_size)
        P.add_constraint(constant + constraint>>0)
        row_offset += block_size**2
    return P

def convert_to_picos(sdpRelaxation):
    """Convert an SDP relaxation to a PICOS problem.

    :param sdpRelaxation: The SDP relaxation to convert.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.

    :returns: :class:`picos.Problem`.
    """
    import picos as pic
    P = pic.Problem()
    row_offsets = [0]
    block_of_last_moment_matrix = 0
    for block, block_size in enumerate(sdpRelaxation.block_struct):
        if block > 0 and block_size < sdpRelaxation.block_struct[block]:
            block_of_last_moment_matrix = block - 1
        row_offsets.append(row_offsets[block]+block_size ** 2)
    X = []
    # First we work on the moment matrices
    for block in range(block_of_last_moment_matrix+1):
        block_size = sdpRelaxation.block_struct[block]
        X.append(P.add_variable('X%s' % block, (block_size, block_size),
                                'symmetric'))
        P.add_constraint(X[block] >> 0)
        start = row_offsets[block]
        end = row_offsets[block] + block_size ** 2
        # If there is a constant term, the moment matrix is normalized
        if sdpRelaxation.F_struct[start:end, 0].getnnz() > 0:
            P.add_constraint(X[block][0, 0] == 1)
        # Here we define the internal symmetries of the moment matrix
        for k in range(1, sdpRelaxation.n_vars+1):
            if sdpRelaxation.F_struct[start:end, k].getnnz() > 1:
                row0 = sdpRelaxation.F_struct[start:end, k].nonzero()[0][0]
                block_index, i1, j1 = \
                    convert_row_to_sdpa_index(
                        sdpRelaxation.block_struct,
                        row_offsets,
                        row0)
                for row in sdpRelaxation.F_struct[start:end, k].nonzero()[0][1:]:
                    block_index, i2, j2 = \
                        convert_row_to_sdpa_index(
                            sdpRelaxation.block_struct,
                            row_offsets,
                            row)
                    if not (i1 == i2 and j1 == j2):
                        P.add_constraint(X[block][i2, j2] == X[block][i1, j1])
    # Next we proceed to the constraints
    for block in range(block_of_last_moment_matrix+1,
                       len(sdpRelaxation.block_struct)):
        block_size = sdpRelaxation.block_struct[block]
        Y = P.add_variable('Y%s' % block, (block_size, block_size),
                           'symmetric')
        P.add_constraint(Y >> 0)
        start = row_offsets[block]
        end = row_offsets[block] + block_size ** 2
        for row in range(start, end):
            row_vector = sdpRelaxation.F_struct.getrow(row)
            affine_expression = \
              _row_to_affine_expression(row_vector, sdpRelaxation.F_struct,
                                       row_offsets,
                                       block_of_last_moment_matrix,
                                       sdpRelaxation.block_struct, X)
            if affine_expression != None:
                i, j = divmod(row-start, block_size)
                P.add_constraint(Y[i, j] == affine_expression)
    affine_expression = \
      _objective_to_affine_expression(sdpRelaxation.obj_facvar,
                                     sdpRelaxation.F_struct, row_offsets,
                                     block_of_last_moment_matrix,
                                     sdpRelaxation.block_struct, X)
    P.set_objective('min', affine_expression)
    return P
