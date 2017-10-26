# -*- coding: utf-8 -*-
"""
This file contains helper functions to work with SDPA.

Created on Fri May 16 13:52:58 2014

@author: Peter Wittek
"""
from __future__ import division, print_function
from bisect import bisect_left
import numpy as np
import os
from subprocess import call
import tempfile
from .nc_utils import convert_monomial_to_string


def parse_solution_matrix(iterator):
    solution_matrix = []
    while True:
        sol_mat = None
        in_matrix = False
        i = 0
        for row in iterator:
            if row.find('}') < 0:
                continue
            if row.startswith('}'):
                break
            if row.find('{') != row.rfind('{'):
                in_matrix = True
            numbers = row[row.rfind('{')+1:row.find('}')].strip().split(',')
            if sol_mat is None:
                sol_mat = np.empty((len(numbers), len(numbers)))
            for j, number in enumerate(numbers):
                sol_mat[i, j] = float(number)
            if row.find('}') != row.rfind('}') or not in_matrix:
                break
            i += 1
        solution_matrix.append(sol_mat)
        if row.startswith('}'):
            break
    if len(solution_matrix) > 0 and solution_matrix[-1] is None:
        solution_matrix = solution_matrix[:-1]
    return solution_matrix


def read_sdpa_out(filename, solutionmatrix=False, status=False,
                  sdp=None):
    """Helper function to parse the output file of SDPA.

    :param filename: The name of the SDPA output file.
    :type filename: str.
    :param solutionmatrix: Optional parameter for retrieving the solution.
    :type solutionmatrix: bool.
    :param status: Optional parameter for retrieving the status.
    :type status: bool.
    :param sdp: Optional parameter to add the solution to a
                          relaxation.
    :type sdp: sdp.
    :returns: tuple of two floats and optionally two lists of `numpy.array` and
              a status string
    """
    primal = None
    dual = None
    x_mat = None
    y_mat = None
    status_string = None

    with open(filename, 'r') as file_:
        for line in file_:
            if line.find("objValPrimal") > -1:
                primal = float((line.split())[2])
            if line.find("objValDual") > -1:
                dual = float((line.split())[2])
            if solutionmatrix:
                if line.find("xMat =") > -1:
                    x_mat = parse_solution_matrix(file_)
                if line.find("yMat =") > -1:
                    y_mat = parse_solution_matrix(file_)
            if line.find("phase.value") > -1:
                if line.find("pdOPT") > -1:
                    status_string = 'optimal'
                elif line.find("pFEAS") > -1:
                    status_string = 'primal feasible'
                elif line.find("pdFEAS") > -1:
                    status_string = 'primal-dual feasible'
                elif line.find("dFEAS") > -1:
                    status_string = 'dual feasible'
                elif line.find("INF") > -1:
                    status_string = 'infeasible'
                elif line.find("UNBD") > -1:
                    status_string = 'unbounded'
                else:
                    status_string = 'unknown'

    for var in [primal, dual, status_string]:
        if var is None:
            status_string = 'invalid'
            break
    if solutionmatrix:
        for var in [x_mat, y_mat]:
            if var is None:
                status_string = 'invalid'
                break

    if sdp is not None:
        sdp.primal = primal
        sdp.dual = dual
        sdp.x_mat = x_mat
        sdp.y_mat = y_mat
        sdp.status = status_string
    if solutionmatrix and status:
        return primal, dual, x_mat, y_mat, status_string
    elif solutionmatrix:
        return primal, dual, x_mat, y_mat
    elif status:
        return primal, dual, status_string
    else:
        return primal, dual


def which(program):

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def detect_sdpa(solverparameters):
    solverexecutable = "sdpa"
    if solverparameters is not None and "executable" in solverparameters:
        solverexecutable = solverparameters["executable"]
    return which(solverexecutable)


def solve_with_sdpa(sdp, solverparameters=None):
    """Helper function to write out the SDP problem to a temporary
    file, call the solver, and parse the output.

    :param sdp: The SDP relaxation to be solved.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    :param solverparameters: Optional parameters to SDPA.
    :type solverparameters: dict of str.
    :returns: tuple of float and list -- the primal and dual solution of the
              SDP, respectively, and a status string.
    """
    solverexecutable = detect_sdpa(solverparameters)
    if solverexecutable is None:
        raise OSError("SDPA is not in the path or the executable provided is" +
                      " not correct")
    primal, dual = 0, 0
    tempfile_ = tempfile.NamedTemporaryFile()
    tmp_filename = tempfile_.name
    tempfile_.close()
    tmp_dats_filename = tmp_filename + ".dat-s"
    tmp_out_filename = tmp_filename + ".out"
    write_to_sdpa(sdp, tmp_dats_filename)
    command_line = [solverexecutable, "-ds", tmp_dats_filename,
                    "-o", tmp_out_filename]
    if solverparameters is not None:
        for key, value in list(solverparameters.items()):
            if key == "executable":
                continue
            elif key == "paramsfile":
                command_line.extend(["-p", value])
            else:
                raise ValueError("Unknown parameter for SDPA: " + key)
    if sdp.verbose < 1:
        with open(os.devnull, "w") as fnull:
            call(command_line, stdout=fnull, stderr=fnull)
    else:
        call(command_line)
    primal, dual, x_mat, y_mat, status = read_sdpa_out(tmp_out_filename, True,
                                                       True)
    if sdp.verbose < 2:
        os.remove(tmp_dats_filename)
        os.remove(tmp_out_filename)
    return primal+sdp.constant_term, \
        dual+sdp.constant_term, x_mat, y_mat, status


def convert_row_to_sdpa_index(block_struct, row_offsets, row):
    """Helper function to map to sparse SDPA index values.
    """
    block_index = bisect_left(row_offsets[1:], row + 1)
    width = block_struct[block_index]
    row = row - row_offsets[block_index]
    i, j = divmod(row, width)
    return block_index, i, j


def write_to_sdpa(sdp, filename):
    """Write the SDP relaxation to SDPA format.

    :param sdp: The SDP relaxation to write.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    :param filename: The name of the file. It must have the suffix ".dat-s"
    :type filename: str.
    """
    # Coefficient matrices
    row_offsets = [0]
    cumulative_sum = 0
    for block_size in sdp.block_struct:
        cumulative_sum += block_size ** 2
        row_offsets.append(cumulative_sum)
    multiplier = 1
    if sdp.F.dtype == np.complex128:
        multiplier = 2
    lines = [[] for _ in range(multiplier*sdp.n_vars+1)]
    for row in range(len(sdp.F.rows)):
        if len(sdp.F.rows[row]) > 0:
            col_index = 0
            block_index, i, j = convert_row_to_sdpa_index(sdp.block_struct,
                                                          row_offsets, row)
            for k in sdp.F.rows[row]:
                value = sdp.F.data[row][col_index]
                col_index += 1
                if k == 0:
                    value *= -1
                if sdp.F.dtype == np.float64:
                    lines[k].append('{0}\t{1}\t{2}\t{3}\n'.format(
                        block_index + 1, i + 1, j + 1, value))
                else:
                    bs = sdp.block_struct[block_index]
                    if value.real != 0:
                        lines[k].append('{0}\t{1}\t{2}\t{3}\n'.format(
                            block_index + 1, i + 1, j + 1, value.real))
                        lines[k].append('{0}\t{1}\t{2}\t{3}\n'.format(
                            block_index + 1, i + bs + 1, j + bs + 1,
                            value.real))
                    if value.imag != 0:
                        lines[k + sdp.n_vars].append(
                            '{0}\t{1}\t{2}\t{3}\n'.format(
                                block_index + 1, i + 1, j + bs + 1,
                                value.imag))
                        lines[k + sdp.n_vars].append(
                            '{0}\t{1}\t{2}\t{3}\n'.format(
                                block_index + 1, j + 1, i + bs + 1,
                                -value.imag))
    file_ = open(filename, 'w')
    file_.write('"file ' + filename + ' generated by ncpol2sdpa"\n')
    file_.write(str(multiplier*sdp.n_vars) + ' = number of vars\n')
    # bloc structure
    block_struct = [multiplier*blk_size
                    for blk_size in sdp.block_struct]
    file_.write(str(len(block_struct)) + ' = number of blocs\n')
    file_.write(str(block_struct).replace('[', '(')
                .replace(']', ')'))
    file_.write(' = BlocStructure\n')
    # c vector (objective)
    objective = \
        str(list(sdp.obj_facvar)).replace('[', '').replace(']', '')
    if multiplier == 2:
        objective += ', ' + objective
    file_.write('{'+objective+'}\n')
    for k, line in enumerate(lines):
        if line == []:
            continue
        for item in line:
            file_.write('{0}\t'.format(k)+item)
    file_.close()


def convert_to_human_readable(sdp):
    """Convert the SDP relaxation to a human-readable format.

    :param sdp: The SDP relaxation to write.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    :returns: tuple of the objective function in a string and a matrix of
              strings as the symbolic representation of the moment matrix
    """

    objective = ""
    indices_in_objective = []
    for i, tmp in enumerate(sdp.obj_facvar):
        candidates = [key for key, v in
                      sdp.monomial_index.items() if v == i+1]
        if len(candidates) > 0:
            monomial = convert_monomial_to_string(candidates[0])
        else:
            monomial = ""
        if tmp > 0:
            objective += "+"+str(tmp)+monomial
            indices_in_objective.append(i)
        elif tmp < 0:
            objective += str(tmp)+monomial
            indices_in_objective.append(i)

    matrix_size = 0
    cumulative_sum = 0
    row_offsets = [0]
    block_offset = [0]
    for bs in sdp.block_struct:
        matrix_size += abs(bs)
        cumulative_sum += bs ** 2
        row_offsets.append(cumulative_sum)
        block_offset.append(matrix_size)

    matrix = []
    for i in range(matrix_size):
        matrix_line = ["0"] * matrix_size
        matrix.append(matrix_line)

    for row in range(len(sdp.F.rows)):
        if len(sdp.F.rows[row]) > 0:
            col_index = 0
            for k in sdp.F.rows[row]:
                value = sdp.F.data[row][col_index]
                col_index += 1
                block_index, i, j = convert_row_to_sdpa_index(
                    sdp.block_struct, row_offsets, row)
                candidates = [key for key, v in
                              sdp.monomial_index.items()
                              if v == k]
                if len(candidates) > 0:
                    monomial = convert_monomial_to_string(candidates[0])
                else:
                    monomial = ""
                offset = block_offset[block_index]
                if matrix[offset+i][offset+j] == "0":
                    matrix[offset+i][offset+j] = ("%s%s" % (value, monomial))
                else:
                    if value.real > 0:
                        matrix[offset+i][offset+j] += ("+%s%s" % (value,
                                                                  monomial))
                    else:
                        matrix[offset+i][offset+j] += ("%s%s" % (value,
                                                                 monomial))
    return objective, matrix


def write_to_human_readable(sdp, filename):
    """Write the SDP relaxation to a human-readable format.

    :param sdp: The SDP relaxation to write.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    :param filename: The name of the file.
    :type filename: str.
    """
    objective, matrix = convert_to_human_readable(sdp)
    f = open(filename, 'w')
    f.write("Objective:" + objective + "\n")
    for matrix_line in matrix:
        f.write(str(list(matrix_line)).replace('[', '').replace(']', '')
                .replace('\'', ''))
        f.write('\n')
    f.close()
