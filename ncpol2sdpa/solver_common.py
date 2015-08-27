# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:05:24 2015

@author: Peter Wittek
"""
import numpy as np
import time
from sympy import expand, Number
try:
    from scipy.sparse import lil_matrix
except ImportError:
    from .sparse_utils import lil_matrix
from .nc_utils import pick_monomials_up_to_degree, simplify_polynomial, \
                      apply_substitutions, build_monomial
from .sdpa_utils import solve_with_sdpa, convert_row_to_sdpa_index, detect_sdpa
from .mosek_utils import solve_with_mosek
from .picos_utils import solve_with_cvxopt


def autodetect_solvers(solverparameters):
    solvers = []
    if detect_sdpa(solverparameters) is not None:
        solvers.append("sdpa")
    try:
        import mosek
    except ImportError:
        pass
    else:
        solvers.append("mosek")
    try:
        import picos
    except ImportError:
        pass
    else:
        solvers.append("cvxopt")
    return solvers


def solve_sdp(sdpRelaxation, solver=None, solverparameters=None):
    """Call a solver on the SDP relaxation. Upon successful solution, it
    returns the primal and dual objective values along with the solution
    matrices. It also sets these values in the `sdpRelaxation` object, along
    with some status information.

    :param sdpRelaxation: The SDP relaxation to be solved.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.
    :param solver: The solver to be called, either `None`, "sdpa", "mosek", or
                   "cvxopt". The default is `None`, which triggers autodetect.
    :type solver: str.
    :param solverparameters: Parameters to be passed to the solver. Actual
                             options depend on the solver:

                             SDPA:

                               - "executable":
                                 Specify the executable for SDPA. E.g.,
                                 "executable":"/usr/local/bin/sdpa", or
                                 "executable":"sdpa_gmp"
                               - "paramsfile": Specify the parameter file.

                             Mosek:
                             Refer to the Mosek documentation. All arguments
                             are passed on.

                             Cvxopt:
                             Refer to the PICOS documentation. All arguments
                             are passed on.
    :type solverparameters: dict of str.
    :returns: tuple of the primal and dual optimum, and the solutions for the
              primal and dual.
    :rtype: (float, float, list of `numpy.array`, list of `numpy.array`)
    """
    solvers = autodetect_solvers(solverparameters)
    solver = solver.lower() if solver is not None else solver
    if solvers == []:
        raise Exception("Could not find any SDP solver. Please install SDPA," +
                        " Mosek, or Picos with Cvxopt")
    elif solver is not None and solver not in solvers:
        print("Available solvers: " + str(solvers))
        raise Exception("Could not detect requested " + solver)
    elif solver is None:
        solver = solvers[0]
    primal, dual, x_mat, y_mat, status = None, None, None, None, None
    tstart = time.time()
    if solver == "sdpa":
        primal, dual, x_mat, y_mat, status = \
          solve_with_sdpa(sdpRelaxation, solverparameters)
    elif solver == "mosek":
        primal, dual, x_mat, y_mat, status = \
          solve_with_mosek(sdpRelaxation, solverparameters)
    elif solver == "cvxopt":
        primal, dual, x_mat, y_mat, status = \
          solve_with_cvxopt(sdpRelaxation, solverparameters)
    else:
        raise Exception("Unkown solver: " + solver)
    sdpRelaxation.solution_time = time.time() - tstart
    sdpRelaxation.primal = primal
    sdpRelaxation.dual = dual
    sdpRelaxation.x_mat = x_mat
    sdpRelaxation.y_mat = y_mat
    sdpRelaxation.status = status
    return primal, dual, x_mat, y_mat


def find_rank_loop(sdpRelaxation, x_mat=None, base_level=0):
    """Helper function to detect rank loop in the solution matrix.

    :param sdpRelaxation: The SDP relaxation.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.
    :param x_mat: Optional parameter providing the primal solution of the
                   moment matrix. If not provided, the solution is extracted
                  from the sdpRelaxation object.
   :type x_mat: :class:`numpy.array`.
    :param base_level: Optional parameter for specifying the lower level
                       relaxation for which the rank loop should be tested
                       against.
    :type base_level: int.
    :returns: list of int -- the ranks of the solution matrix with in the
                             order of increasing degree.
    """
    if sdpRelaxation.status == "unsolved" and x_mat is None:
        raise Exception("The SDP relaxation is unsolved and no primal " +
                        "solution is provided!")
    elif sdpRelaxation.status != "unsolved" and x_mat is None:
        x_mat = sdpRelaxation.x_mat[0]
    else:
        x_mat = sdpRelaxation.x_mat[0]
    if sdpRelaxation.status == "unsolved":
        raise Exception("The SDP relaxation is unsolved!")
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

    if x_mat.shape != (len(base_monomials), len(base_monomials)):
        ranks.append(matrix_rank(x_mat))
    return ranks


def sos_decomposition(sdpRelaxation, y_mat=None, threshold=0.0):
    """Given a solution of the dual problem, it returns the SOS
    decomposition. Currently limited to unconstrained problems.

    :param sdpRelaxation: The SDP relaxation to be solved.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.
    :param y_mat: Optional parameter providing the dual solution of the
                  moment matrix. If not provided, the solution is extracted
                  from the sdpRelaxation object.
    :type y_mat: :class:`numpy.array`.
    :param threshold: Optional parameter for specifying the threshold value
                      below which the eigenvalues and entries of the
                      eigenvectors are disregarded.
    :type threshold: float.
    :returns: The SOS decomposition
    :rtype: :class:`sympy.core.exp.Expr`.
    """
    if sdpRelaxation.status == "unsolved" and y_mat is None:
        raise Exception("The SDP relaxation is unsolved and dual solution " +
                        "is not provided!")
    elif sdpRelaxation.status != "unsolved" and y_mat is None:
        y_mat = sdpRelaxation.y_mat
    sos = 0
    vals, vecs = np.linalg.eigh(y_mat[0])
    for j, val in enumerate(vals):
        if abs(val) > threshold:
            term = 0
            for i, entry in enumerate(vecs[:, j]):
                if abs(entry) > threshold:
                    term += np.sqrt(val) * entry * \
                      sdpRelaxation.monomial_sets[0][i]
            sos += term**2
    return sos


def get_index_of_monomial(monomial, row_offsets, sdpRelaxation):
    k = sdpRelaxation.monomial_index[monomial]
    Fk = sdpRelaxation.F_struct.getcol(k)
    if not isinstance(Fk, lil_matrix):
        Fk = Fk.tolil()
    for row in range(len(Fk.rows)):
        if Fk.rows[row] != []:
            block, i, j = convert_row_to_sdpa_index(sdpRelaxation.block_struct,
                                                    row_offsets, row)
            return row, k, block, i, j


def get_recursive_xmat_value(k, row_offsets, sdpRelaxation, x_mat):
    Fk = sdpRelaxation.F_struct[:, k]
    for row in range(len(Fk.rows)):
        if Fk.rows[row] != []:
            block, i, j = convert_row_to_sdpa_index(sdpRelaxation.block_struct,
                                                    row_offsets, row)
            value = x_mat[block][i, j]
            for index in sdpRelaxation.F_struct.rows[row]:
                if k != index:
                    value -= sdpRelaxation.F_struct[row, index] * \
                               get_recursive_xmat_value(index, row_offsets,
                                                        sdpRelaxation)
            return value / sdpRelaxation.F_struct[row, k]


def get_xmat_value(monomial, sdpRelaxation, x_mat=None):
    """Given a solution of the primal problem and a monomial, it returns the
    value for the monomial in the solution matrix.

    :param monomial: The monomial for which the value is requested.
    :type monomial: :class:`sympy.core.exp.Expr`.
    :param sdpRelaxation: The SDP relaxation to be solved.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.
    :param x_mat: Optional parameter providing the primal solution of the
                  moment matrix. If not provided, the solution is extracted
                  from the sdpRelaxation object.
    :type x_mat: :class:`numpy.array`.
    :returns: The value of the monomial in the solved relaxation.
    :rtype: float.
    """
    if sdpRelaxation.status == "unsolved" and x_mat is None:
        raise Exception("The SDP relaxation is unsolved and no primal " +
                        "solution is provided!")
    elif sdpRelaxation.status != "unsolved" and x_mat is None:
        x_mat = sdpRelaxation.x_mat
    polynomial = expand(simplify_polynomial(monomial,
                                            sdpRelaxation.substitutions))
    if polynomial.is_Mul:
        elements = [polynomial]
    else:
        elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]
    row_offsets = [0]
    cumulative_sum = 0
    for block_size in sdpRelaxation.block_struct:
        cumulative_sum += block_size ** 2
        row_offsets.append(cumulative_sum)
    result = 0
    for element in elements:
        element, coeff = build_monomial(element)
        element = apply_substitutions(element, sdpRelaxation.substitutions)
        if isinstance(element, Number):
            result += coeff*float(element)
        else:
            row, k, block, i, j = get_index_of_monomial(element, row_offsets,
                                                        sdpRelaxation)
            value = x_mat[block][i, j]
            for index in sdpRelaxation.F_struct.rows[row]:
                if k != index:
                    value -= sdpRelaxation.F_struct[row, index] * \
                               get_recursive_xmat_value(index, row_offsets,
                                                        sdpRelaxation, x_mat)
            result += coeff * value / sdpRelaxation.F_struct[row, k]
    return result
