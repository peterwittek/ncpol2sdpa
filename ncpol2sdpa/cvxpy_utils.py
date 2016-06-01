# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:39:15 2016

@author: Peter Wittek
"""

from __future__ import division, print_function
import numpy as np
try:
    from scipy.sparse import lil_matrix
except ImportError:
    from .sparse_utils import lil_matrix
from .sdpa_utils import convert_row_to_sdpa_index


def solve_with_cvxpy(sdp, solverparameters=None):
    """Helper function to convert the SDP problem to PICOS
    and call CVXOPT solver, and parse the output.

    :param sdp: The SDP relaxation to be solved.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    """
    problem = convert_to_cvxpy(sdp)
    if solverparameters is not None and 'solver' in solverparameters:
        solver = solverparameters.pop('solver')
        v = problem.solve(solver=solver, solver_specific_opts=solverparameters,
                          verbose=sdp.verbose)
    else:
        v = problem.solve(verbose=sdp.verbose)
    if v is None:
        status = "infeasible"
        x_mat, y_mat = [], []
    elif v == float("inf") or v == -float("inf"):
        status = "unbounded"
        x_mat, y_mat = [], []
    else:
        status = "optimal"
        x_pre = sdp.F[:, 1:sdp.n_vars+1].dot(problem.variables()[0].value)
        x_pre += sdp.F[:, 0]
        row_offsets = [0]
        cumulative_sum = 0
        for block_size in sdp.block_struct:
            cumulative_sum += block_size ** 2
            row_offsets.append(cumulative_sum)
        x_mat = []
        for bi, bs in enumerate(sdp.block_struct):
            x = x_pre[row_offsets[bi]:row_offsets[bi+1]].reshape((bs, bs))
            x += x.T - np.diag(np.array(x.diagonal())[0])
            x_mat.append(x)
        y_mat = [constraint.dual_value for constraint in problem.constraints]
    return v+sdp.constant_term, v+sdp.constant_term, x_mat, y_mat, status


def convert_to_cvxpy(sdp):
    """Convert an SDP relaxation to a CVXPY problem.

    :param sdp: The SDP relaxation to convert.
    :type sdp: :class:`ncpol2sdpa.sdp`.

    :returns: :class:`cvxpy.Problem`.
    """
    from cvxpy import Minimize, Problem, Variable
    row_offsets = [0]
    cumulative_sum = 0
    for block_size in sdp.block_struct:
        cumulative_sum += block_size ** 2
        row_offsets.append(cumulative_sum)
    x = Variable(sdp.n_vars)
    # The moment matrices are the first blocks of identical size
    constraints = []
    for idx, bs in enumerate(sdp.block_struct):
        nonzero_set = set()
        F = [lil_matrix((bs, bs)) for _ in range(sdp.n_vars+1)]
        for ri, row in enumerate(sdp.F.rows[row_offsets[idx]:
                                            row_offsets[idx+1]],
                                 row_offsets[idx]):
            block_index, i, j = convert_row_to_sdpa_index(sdp.block_struct,
                                                          row_offsets, ri)
            for col_index, k in enumerate(row):
                value = sdp.F.data[ri][col_index]
                F[k][i, j] = value
                F[k][j, i] = value
                nonzero_set.add(k)
        if bs > 1:
            sum_ = sum(F[k]*x[k-1] for k in nonzero_set if k > 0)
            if not isinstance(sum_, (int, float)):
                if F[0].getnnz() > 0:
                    sum_ += F[0]
                constraints.append(sum_ >> 0)
        else:
            sum_ = sum(F[k][0, 0]*x[k-1] for k in nonzero_set if k > 0)
            if not isinstance(sum_, (int, float)):
                sum_ += F[0][0, 0]
                constraints.append(sum_ >= 0)
    obj = sum(ci*xi for ci, xi in zip(sdp.obj_facvar, x) if ci != 0)
    problem = Problem(Minimize(obj), constraints)
    return problem
