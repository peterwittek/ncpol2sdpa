# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:05:24 2015

@author: Peter Wittek
"""
import numpy as np
from .nc_utils import pick_monomials_up_to_degree
from .sdpa_utils import solve_with_sdpa
from .mosek_utils import solve_with_mosek

def solve_sdp(sdpRelaxation, solver="sdpa", solverparameters=None):
    if solver is "sdpa":
        return solve_with_sdpa(sdpRelaxation, solverparameters)
    elif solver is "mosek":
        return solve_with_mosek(sdpRelaxation, solverparameters)

def find_rank_loop(sdpRelaxation, x_mat, base_level=0):
    """Helper function to detect rank loop in the solution matrix.

    :param sdpRelaxation: The SDP relaxation to be solved.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.
    :param x_mat: The solution of the moment matrix.
    :type x_mat: :class:`numpy.array`.
    :param base_level: Optional parameter for specifying the lower level
                       relaxation for which the rank loop should be tested
                       against.
    :type base_level: int.
    :returns: list of int -- the ranks of the solution matrix with in the
                             order of increasing degree.
    """
    ranks = []
    from numpy.linalg import matrix_rank
    if sdpRelaxation.hierarchy != "npa":
        raise Exception("The detection of rank loop is only implemented for \
                         the NPA hierarchy")
    if base_level == 0:
        levels = range(1, sdpRelaxation.level + 1)
    else:
        levels = [base_level]
    for level in levels:
        base_monomials = \
          pick_monomials_up_to_degree(sdpRelaxation.monomial_sets[0], level)
        ranks.append(matrix_rank(x_mat[:len(base_monomials),
                                       :len(base_monomials)]))
    ranks.append(matrix_rank(x_mat))
    return ranks

def sos_decomposition(sdpRelaxation, y_mat, threshold=0.0):
    """Given a solution of the dual problem, it returns the SOS
    decomposition. Currently limited to unconstrained problems.

    :param sdpRelaxation: The SDP relaxation to be solved.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.
    :param y_mat: The dual solution of the problem.
    :type y_mat: :class:`numpy.array`.
    :param threshold: Optional parameter for specifying the threshold value
                      below which the eigenvalues and entries of the
                      eigenvectors are disregarded.
    :type threshold: float.
    """
    sos = 0
    vals, vecs = np.linalg.eigh(y_mat[0])
    for j, val in enumerate(vals):
        if abs(val) > threshold:
            term = 0
            for i, entry in enumerate(vecs[:, j]):
                if abs(entry) > threshold:
                    term += np.sqrt(val)*entry*sdpRelaxation.monomial_sets[0][i]
            sos += term**2
    return sos
