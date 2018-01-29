# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:05:24 2015

@author: Peter Wittek
"""
from __future__ import division, print_function
import numpy as np
from sympy import expand
import time
from .nc_utils import pick_monomials_up_to_degree, simplify_polynomial, \
                      apply_substitutions, separate_scalar_factor, \
                      is_number_type
from .cvxpy_utils import solve_with_cvxpy
from .sdpa_utils import solve_with_sdpa, convert_row_to_sdpa_index, detect_sdpa
from .mosek_utils import solve_with_mosek
from .picos_utils import solve_with_cvxopt


def autodetect_solvers(solverparameters):
    solvers = []
    if detect_sdpa(solverparameters) is not None:
        solvers.append("sdpa")
    try:
        import cvxpy
    except ImportError:
        pass
    else:
        solvers.append("cvxpy")
        solvers.append("scs")
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


def solve_sdp(sdp, solver=None, solverparameters=None):
    """Call a solver on the SDP relaxation. Upon successful solution, it
    returns the primal and dual objective values along with the solution
    matrices.

    :param sdpRelaxation: The SDP relaxation to be solved.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.
    :param solver: The solver to be called, either `None`, "sdpa", "mosek",
                   "cvxpy", "scs", or "cvxopt". The default is `None`,
                   which triggers autodetect.
    :type solver: str.
    :param solverparameters: Parameters to be passed to the solver. Actual
                             options depend on the solver:

                             SDPA:

                               - `"executable"`:
                                 Specify the executable for SDPA. E.g.,
                                 `"executable":"/usr/local/bin/sdpa"`, or
                                 `"executable":"sdpa_gmp"`
                               - `"paramsfile"`: Specify the parameter file

                             Mosek:
                             Refer to the Mosek documentation. All
                             arguments are passed on.

                             Cvxopt:
                             Refer to the PICOS documentation. All
                             arguments are passed on.

                             Cvxpy:
                             Refer to the Cvxpy documentation. All
                             arguments are passed on.

                             SCS:
                             Refer to the Cvxpy documentation. All
                             arguments are passed on.
    :type solverparameters: dict of str.
    :returns: tuple of the primal and dual optimum, and the solutions for the
              primal and dual.
    :rtype: (float, float, list of `numpy.array`, list of `numpy.array`)
    """
    solvers = autodetect_solvers(solverparameters)
    solver = solver.lower() if solver is not None else solver
    if solvers == []:
        raise Exception("Could not find any SDP solver. Please install SDPA," +
                        " Mosek, Cvxpy, or Picos with Cvxopt")
    elif solver is not None and solver not in solvers:
        print("Available solvers: " + str(solvers))
        if solver == "cvxopt":
            try:
                import cvxopt
            except ImportError:
                pass
            else:
                raise Exception("Cvxopt is detected, but Picos is not. "
                                "Please install Picos to use Cvxopt")
        raise Exception("Could not detect requested " + solver)
    elif solver is None:
        solver = solvers[0]
    primal, dual, x_mat, y_mat, status = None, None, None, None, None
    tstart = time.time()
    if solver == "sdpa":
        primal, dual, x_mat, y_mat, status = \
          solve_with_sdpa(sdp, solverparameters)
    elif solver == "cvxpy":
        primal, dual, x_mat, y_mat, status = \
          solve_with_cvxpy(sdp, solverparameters)
    elif solver == "scs":
        if solverparameters is None:
            solverparameters_ = {"solver": "SCS"}
        else:
            solverparameters_ = solverparameters.copy()
            solverparameters_["solver"] = "SCS"
        primal, dual, x_mat, y_mat, status = \
          solve_with_cvxpy(sdp, solverparameters_)
    elif solver == "mosek":
        primal, dual, x_mat, y_mat, status = \
          solve_with_mosek(sdp, solverparameters)
    elif solver == "cvxopt":
        primal, dual, x_mat, y_mat, status = \
          solve_with_cvxopt(sdp, solverparameters)
        # We have to compensate for the equality constraints
        for constraint in sdp.constraints[sdp._n_inequalities:]:
            idx = sdp._constraint_to_block_index[constraint]
            sdp._constraint_to_block_index[constraint] = (idx[0],)
    else:
        raise Exception("Unkown solver: " + solver)
    sdp.solution_time = time.time() - tstart
    sdp.primal = primal
    sdp.dual = dual
    sdp.x_mat = x_mat
    sdp.y_mat = y_mat
    sdp.status = status
    return primal, dual, x_mat, y_mat


def find_solution_ranks(sdp, xmat=None, baselevel=0):
    """Helper function to detect rank loop in the solution matrix.

    :param sdp: The SDP relaxation.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    :param x_mat: Optional parameter providing the primal solution of the
                  moment matrix. If not provided, the solution is extracted
                  from the sdp object.
    :type x_mat: :class:`numpy.array`.
    :param base_level: Optional parameter for specifying the lower level
                       relaxation for which the rank loop should be tested
                       against.
    :type base_level: int.
    :returns: list of int -- the ranks of the solution matrix with in the
              order of increasing degree.
    """
    if sdp.status == "unsolved" and xmat is None:
        raise Exception("The SDP relaxation is unsolved and no primal " +
                        "solution is provided!")
    elif sdp.status != "unsolved" and xmat is None:
        xmat = sdp.x_mat[0]
    else:
        xmat = sdp.x_mat[0]
    if sdp.status == "unsolved":
        raise Exception("The SDP relaxation is unsolved!")
    ranks = []
    from numpy.linalg import matrix_rank
    if baselevel == 0:
        levels = range(1, sdp.level + 1)
    else:
        levels = [baselevel]
    for level in levels:
        base_monomials = \
            pick_monomials_up_to_degree(sdp.monomial_sets[0], level)
        ranks.append(matrix_rank(xmat[:len(base_monomials),
                                      :len(base_monomials)]))
    if xmat.shape != (len(base_monomials), len(base_monomials)):
        ranks.append(matrix_rank(xmat))
    return ranks


def get_sos_decomposition(sdp, y_mat=None, threshold=0.0):
    """Given a solution of the dual problem, it returns the SOS
    decomposition.

    :param sdp: The SDP relaxation to be solved.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    :param y_mat: Optional parameter providing the dual solution of the
                  moment matrix. If not provided, the solution is extracted
                  from the sdp object.
    :type y_mat: :class:`numpy.array`.
    :param threshold: Optional parameter for specifying the threshold value
                      below which the eigenvalues and entries of the
                      eigenvectors are disregarded.
    :type threshold: float.
    :returns: The SOS decomposition of [sigma_0, sigma_1, ..., sigma_m]
    :rtype: list of :class:`sympy.core.exp.Expr`.
    """
    if len(sdp.monomial_sets) != 1:
        raise Exception("Cannot automatically match primal and dual " +
                        "variables.")
    elif len(sdp.y_mat[1:]) != len(sdp.constraints):
        raise Exception("Cannot automatically match constraints with blocks " +
                        "in the dual solution.")
    elif sdp.status == "unsolved" and y_mat is None:
        raise Exception("The SDP relaxation is unsolved and dual solution " +
                        "is not provided!")
    elif sdp.status != "unsolved" and y_mat is None:
        y_mat = sdp.y_mat
    sos = []
    for y_mat_block in y_mat:
        term = 0
        vals, vecs = np.linalg.eigh(y_mat_block)
        for j, val in enumerate(vals):
            if val < -0.001:
                raise Exception("Large negative eigenvalue: " + val +
                                ". Matrix cannot be positive.")
            elif val > 0:
                sub_term = 0
                for i, entry in enumerate(vecs[:, j]):
                    sub_term += entry * sdp.monomial_sets[0][i]
                term += val * sub_term**2
        term = expand(term)
        new_term = 0
        if term.is_Mul:
            elements = [term]
        else:
            elements = term.as_coeff_mul()[1][0].as_coeff_add()[1]
        for element in elements:
            _, coeff = separate_scalar_factor(element)
            if abs(coeff) > threshold:
                new_term += element
        sos.append(new_term)
    return sos


def get_facvar_of_monomial(monomial, sdp):
    results = sdp._get_index_of_monomial(monomial)
    if sdp._new_basis is None:
        facvar = np.zeros((sdp.n_vars + 1,))
    else:
        facvar = np.zeros((len(sdp._original_obj_facvar) + 1,))
    for (k, coeff) in results:
        facvar[k] += coeff
    if sdp._new_basis is not None:
        new_facvar = np.zeros((sdp.n_vars + 1,))
        new_facvar[0] = facvar[0]
        new_facvar[1:] = sdp._new_basis.T.dot(facvar[1:])
        facvar = new_facvar
    return facvar


def get_recursive_xmat_value(k, row_offsets, sdp, x_mat):
    Fk = sdp.F[:, k]
    for row in range(len(Fk.rows)):
        if Fk.rows[row] != []:
            block, i, j = convert_row_to_sdpa_index(sdp.block_struct,
                                                    row_offsets, row)
            value = x_mat[block][i, j]
            for index in sdp.F.rows[row]:
                if k != index:
                    value -= sdp.F[row, index] * \
                        get_recursive_xmat_value(index, row_offsets,
                                                 sdp, x_mat)
            return value / sdp.F[row, k]


def get_xmat_value(monomial, sdp, x_mat=None):
    if sdp.status == "unsolved" and x_mat is None:
        raise Exception("The SDP relaxation is unsolved and no primal " +
                        "solution is provided!")
    elif sdp.status != "unsolved" and x_mat is None:
        x_mat = sdp.x_mat
    polynomial = expand(simplify_polynomial(monomial,
                                            sdp.substitutions))
    if polynomial.is_Mul:
        elements = [polynomial]
    else:
        elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]
    row_offsets = [0]
    cumulative_sum = 0
    for block_size in sdp.block_struct:
        cumulative_sum += block_size ** 2
        row_offsets.append(cumulative_sum)
    linear_combination = get_facvar_of_monomial(0, sdp)
    for element in elements:
        element, coeff = separate_scalar_factor(element)
        if element in sdp.moment_substitutions:
            element = sdp.moment_substitutions[element]
        element = apply_substitutions(element, sdp.substitutions)
        linear_combination += coeff*get_facvar_of_monomial(element, sdp)
    value = linear_combination[0]
    linear_combination[0] = 0
    # This used to be a conditional when moments were never substituted:
    # if len(sdp.block_struct) > sdp.constraint_starting_block:
    for row in range(row_offsets[sdp.constraint_starting_block]):
        intersect = np.intersect1d(sdp.F.rows[row],
                                   np.nonzero(linear_combination)[0])
        if len(intersect) == 0:
            continue
        elif len(intersect) == 1:
            block, i, j = convert_row_to_sdpa_index(sdp.block_struct,
                                                    row_offsets, row)
            value += linear_combination[intersect[0]]*x_mat[block][i, j]
            linear_combination[intersect[0]] = 0
            for index in sdp.F.rows[row]:
                if index not in intersect:
                    value -= sdp.F[row, index] * \
                        get_recursive_xmat_value(index, row_offsets,
                                                 sdp, x_mat)
        else:
            # This can only happen because we changed the basis
            A = np.zeros((len(intersect), len(intersect)))
            b = np.zeros((len(intersect), ))
            in_ = 0
            rank0 = 0
            for row2 in range(row,
                              row_offsets[sdp.constraint_starting_block]):
                block, i, j = convert_row_to_sdpa_index(sdp.block_struct,
                                                        row_offsets, row2)
                is2 = np.intersect1d(sdp.F.rows[row2],
                                     np.nonzero(linear_combination)[0])
                if len(is2) == len(intersect):
                    col = 0
                    for it in intersect:
                        A[in_, col] = sdp.F[row2, it]
                        col += 1
                    rank1 = np.linalg.matrix_rank(A)
                    b[in_] = x_mat[block][i, j]
                    if rank1 > rank0:
                        rank0 = rank1
                        if in_ == len(intersect) - 1:
                            break
                        in_ += 1
            x = np.linalg.solve(A, b)
            for k, it in enumerate(intersect):
                value += x[k]*linear_combination[it]
                linear_combination[it] = 0
    return value


def extract_dual_value(sdp, monomial, blocks=None):
    """Given a solution of the dual problem and a monomial, it returns the
    inner product of the corresponding coefficient matrix and the dual
    solution. It can be restricted to certain blocks.

    :param sdp: The SDP relaxation.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    :param monomial: The monomial for which the value is requested.
    :type monomial: :class:`sympy.core.exp.Expr`.
    :param monomial: The monomial for which the value is requested.
    :type monomial: :class:`sympy.core.exp.Expr`.
    :param blocks: Optional parameter to specify the blocks to be included.
    :type blocks: list of `int`.
    :returns: The value of the monomial in the solved relaxation.
    :rtype: float.
    """
    if sdp.status == "unsolved":
        raise Exception("The SDP relaxation is unsolved!")
    if blocks is None:
        blocks = [i for i, _ in enumerate(sdp.block_struct)]
    if is_number_type(monomial):
        index = 0
    else:
        index = sdp.monomial_index[monomial]
    row_offsets = [0]
    cumulative_sum = 0
    for block_size in sdp.block_struct:
        cumulative_sum += block_size ** 2
        row_offsets.append(cumulative_sum)
    result = 0
    for row in range(len(sdp.F.rows)):
        if len(sdp.F.rows[row]) > 0:
            col_index = 0
            for k in sdp.F.rows[row]:
                if k != index:
                    continue
                value = sdp.F.data[row][col_index]
                col_index += 1
                block_index, i, j = convert_row_to_sdpa_index(
                    sdp.block_struct, row_offsets, row)
                if block_index in blocks:
                    result += -value*sdp.y_mat[block_index][i][j]
    return result
