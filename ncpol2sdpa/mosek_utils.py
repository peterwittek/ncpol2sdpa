# -*- coding: utf-8 -*-
"""
The module contains helper functions to work with MOSEK.

Created on Tue Nov 18 10:51:14 2014

@author: Peter Wittek
"""
import sys
from .sdpa_utils import convert_row_to_sdpa_index

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def convert_to_mosek_index(block_struct, row_offsets, block_offsets, 
                           dimensions, row):
    block_index, i, j = convert_row_to_sdpa_index(block_struct, row_offsets,
                                                  row)
    
    offset = block_offsets[block_index]
    ci = offset + i
    cj = offset + j
    return cj, ci # Note that MOSEK expect lower-triangular matrices

def convert_to_mosek_matrix(sdp_problem):
    barci = []
    barcj = []
    barcval = []
    barai = []
    baraj = []
    baraval = []
    for k in range(sdp_problem.n_vars):
        barai.append([])
        baraj.append([])
        baraval.append([])
    dimensions = sum(sdp_problem.block_struct)
    row_offsets = [0]
    block_offsets = [0]
    cumulative_sum = 0
    cumulative_square_sum = 0
    for block_size in sdp_problem.block_struct:
        cumulative_sum += block_size
        cumulative_square_sum += block_size ** 2
        row_offsets.append(cumulative_square_sum)
        block_offsets.append(cumulative_sum)
    for row in range(len(sdp_problem.F_struct.rows)):
        if len(sdp_problem.F_struct.rows[row]) > 0:
            col_index = 0
            for k in sdp_problem.F_struct.rows[row]:
                value = sdp_problem.F_struct.data[row][col_index]
                i, j = convert_to_mosek_index(sdp_problem.block_struct, row_offsets, block_offsets, dimensions, row)
                if k>0:
                    barai[k-1].append(i)
                    baraj[k-1].append(j)
                    baraval[k-1].append(-value)
                else:
                    barci.append(i)
                    barcj.append(j)
                    barcval.append(value)
                col_index += 1
    return barci, barcj, barcval, barai, baraj, baraval

def convert_to_mosek(sdp_problem):
    import mosek
    barci, barcj, barcval, barai, baraj, baraval = \
      convert_to_mosek_matrix(sdp_problem)
    bkc = [ mosek.boundkey.fx ]  * sdp_problem.n_vars
    blc = [-v for v in sdp_problem.obj_facvar]
    buc = [-v for v in sdp_problem.obj_facvar]

    env = mosek.Env()
    task = env.Task(0,0)
    task.set_Stream(mosek.streamtype.log, streamprinter)
    numvar = 0
    numcon = len(bkc)
    BARVARDIM = [sum(sdp_problem.block_struct)]
        
    task.appendvars(numvar)
    task.appendcons(numcon)
    task.appendbarvars(BARVARDIM)
    for i in range(numcon):
        task.putconbound(i, bkc[i], blc[i], buc[i])

    symc = \
        task.appendsparsesymmat(BARVARDIM[0], 
                                barci, 
                                barcj, 
                                barcval)
    task.putbarcj(0, [symc], [1.0])    

    for i in range(len(barai)):
        syma = task.appendsparsesymmat(BARVARDIM[0], 
                        barai[i], 
                        baraj[i], 
                        baraval[i])
        task.putbaraij(i, 0, [syma], [1.0])

    # Input the objective sense (minimize/maximize)
    task.putobjsense(mosek.objsense.minimize)

    return task    
