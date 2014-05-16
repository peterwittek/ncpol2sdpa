# -*- coding: utf-8 -*-
"""
The module contains helper functions to work with SDPA.

Created on Fri May 16 13:52:58 2014

@author: Peter Wittek
"""
from subprocess import call
import tempfile
from .sdp_relaxation import SdpRelaxation


def read_sdpa_out(filename):
    f = open(filename, 'r')
    for line in f:
        if line.find("objValPrimal") > -1:
            primal = float((line.split())[2])
        if line.find("objValDual") > -1:
            dual = float((line.split())[2])

    f.close()
    return primal, dual


def solve_sdp(sdp_problem):
    primal, dual = 0, 0
    if isinstance(sdp_problem, SdpRelaxation):
        tf = tempfile.NamedTemporaryFile()
        tmp_filename = tf.name
        tf.close()
        tmp_dats_filename = tmp_filename + ".dat-s"
        tmp_out_filename = tmp_filename + ".out"
        sdp_problem.write_to_sdpa(tmp_dats_filename)
        call(["sdpa", tmp_dats_filename, tmp_out_filename])
        primal, dual = read_sdpa_out(tmp_out_filename)
    else:
        out_filename = sdp_problem[:sdp_problem.find(".")] + ".out"
        call(["sdpa", sdp_problem, out_filename])
        primal, dual = read_sdpa_out(out_filename)
    return primal, dual
