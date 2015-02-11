# -*- coding: utf-8 -*-
"""
This file contains helper functions to work with SDPA.

Created on Fri May 16 13:52:58 2014

@author: Peter Wittek
"""
from bisect import bisect_left
from subprocess import call
import tempfile
import os
import numpy as np
from .nc_utils import pick_monomials_up_to_degree

def parse_solution_matrix(iterator):
    solution_matrix = []
    while True:
        sol_mat = None
        i = 0
        for row in iterator:
            if row.find('}') < 0:
                continue
            if row.startswith('}'):
                break
            numbers = row[row.rfind('{')+1:row.find('}')].strip().split(',')
            if sol_mat is None:
                sol_mat = np.empty((len(numbers), len(numbers)))
            for j, number in enumerate(numbers):
                sol_mat[i, j] = float(number)
            if row.find('}') != row.rfind('}'):
                break
            i += 1
        solution_matrix.append(sol_mat)
        if row.startswith('}'):
            break
    return solution_matrix

def read_sdpa_out(filename):
    """Helper function to parse the output file of SDPA.
    :param filename: The name of the SDPA output file.
    :type filename: str.
    """
    file_ = open(filename, 'r')
    for line in file_:
        if line.find("objValPrimal") > -1:
            primal = float((line.split())[2])
        if line.find("objValDual") > -1:
            dual = float((line.split())[2])
        if line.find("xMat =") > -1:
            x_mat = parse_solution_matrix(file_)
        if line.find("yMat =") > -1:
            y_mat = parse_solution_matrix(file_)
    file_.close()
    return primal, dual, x_mat, y_mat


def solve_sdp(sdpRelaxation, solutionmatrix=False,
              solverexecutable="sdpa"):
    """Helper function to write out the SDP problem to a temporary
    file, call the solver, and parse the output.

    :param sdpRelaxation: The SDP relaxation to be solved.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.
    :param solutionmatrix: Optional parameter for retrieving the solution
                           matrix.
    :type solutionmatrix: bool.
    :param solverexecutable: Optional paramater to specify the name of the
                             executable if sdpa is not in the path or has a
                             different name.
    :type solverexecutable: str.
    :returns: tuple of float -- the primal and dual solution of the SDP,
              respectively.
    """
    primal, dual = 0, 0
    tempfile_ = tempfile.NamedTemporaryFile()
    tmp_filename = tempfile_.name
    tempfile_.close()
    tmp_dats_filename = tmp_filename + ".dat-s"
    tmp_out_filename = tmp_filename + ".out"
    write_to_sdpa(sdpRelaxation, tmp_dats_filename)
    if sdpRelaxation.verbose<2:
      with open(os.devnull, "w") as fnull:
          call([solverexecutable, tmp_dats_filename, tmp_out_filename], stdout=fnull, stderr=fnull)
    else:
      call([solverexecutable, tmp_dats_filename, tmp_out_filename])
    primal, dual, x_mat, y_mat = read_sdpa_out(tmp_out_filename)
    if sdpRelaxation.verbose<2:
        os.remove(tmp_dats_filename)
        os.remove(tmp_out_filename)
    if solutionmatrix:
        return primal, dual, x_mat, y_mat
    else:
        return primal, dual

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
        levels = range(1, sdpRelaxation.level)
    else:
        levels = [base_level]
    for level in levels:
        base_monomials = \
          pick_monomials_up_to_degree(sdpRelaxation.monomial_sets[0], level)
        ranks.append(matrix_rank(x_mat[:len(base_monomials),
                                       :len(base_monomials)]))
    ranks.append(matrix_rank(x_mat))
    return ranks

def convert_row_to_sdpa_index(block_struct, row_offsets, row):
    """Helper function to map to sparse SDPA index values.
    """
    block_index = bisect_left(row_offsets[1:], row + 1)
    width = block_struct[block_index]
    row = row - row_offsets[block_index]
    i, j = divmod(row, width)
    return block_index, i, j


def write_to_sdpa(sdpRelaxation, filename):
    """Write the SDP relaxation to SDPA format.

    :param sdpRelaxation: The SDP relaxation to write.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.
    :param filename: The name of the file. It must have the suffix ".dat-s"
    :type filename: str.
    """
    # Coefficient matrices
    row_offsets = [0]
    cumulative_sum = 0
    for block_size in sdpRelaxation.block_struct:
        cumulative_sum += block_size ** 2
        row_offsets.append(cumulative_sum)
    lines = [[] for _ in range(sdpRelaxation.n_vars+1)]
    for row in range(len(sdpRelaxation.F_struct.rows)):
        if len(sdpRelaxation.F_struct.rows[row]) > 0:
            col_index = 0
            for k in sdpRelaxation.F_struct.rows[row]:
                value = sdpRelaxation.F_struct.data[row][col_index]
                col_index += 1
                block_index, i, j = convert_row_to_sdpa_index(
                    sdpRelaxation.block_struct, row_offsets, row)
                if k == 0:
                    value *= -1
                lines[k].append('{0}\t{1}\t{2}\t{3}\n'.format(
                    block_index + 1, i + 1, j + 1, value))
    file_ = open(filename, 'w')
    file_.write('"file ' + filename + ' generated by ncpol2sdpa"\n')
    file_.write(str(sdpRelaxation.n_vars) + ' = number of vars\n')
    file_.write(str(len(sdpRelaxation.block_struct)) + ' = number of blocs\n')
    # bloc structure
    file_.write(str(sdpRelaxation.block_struct).replace('[', '(')
                .replace(']', ')'))
    file_.write(' = BlocStructure\n')
    # c vector (objective)
    file_.write(str(list(sdpRelaxation.obj_facvar)).replace(
        '[', '{').replace(']', '}'))
    file_.write('\n')
    for k, line in enumerate(lines):
        for item in line:
            file_.write('{0}\t'.format(k)+item)
    file_.close()
