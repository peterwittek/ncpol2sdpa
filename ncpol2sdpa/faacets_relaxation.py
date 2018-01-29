# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 11:41:52 2015

@author: wittek
"""
from __future__ import division, print_function
import glob
from math import copysign
import numpy as np
import os
from scipy.sparse import lil_matrix
from .sdp_relaxation import Relaxation


def collinsgisin_to_faacets(I):
    coefficients = []
    for j in range(len(I[0])):
        for i in range(len(I)):
            coefficients.append(I[i][j])
    return np.array(coefficients, dtype='d')


def configuration_to_faacets(A_configuration, B_configuration):
    a = str(A_configuration).replace('[', '(').replace(']', ')').replace(',',
                                                                         '')
    b = str(B_configuration).replace('[', '(').replace(']', ')').replace(',',
                                                                         '')
    return '[' + a + ' ' + b + ']'


def get_faacets_moment_matrix(A_configuration, B_configuration, coefficients):
    from jpype import startJVM, shutdownJVM, JPackage, getDefaultJVMPath, \
        JArray, JDouble
    # find the JAR file and start the JVM
    jarFiles = glob.glob('faacets-*.jar')
    assert len(jarFiles) == 1
    jarFile = os.path.join(os.getcwd(), jarFiles[0])
    startJVM(getDefaultJVMPath(), "-Djava.class.path="+jarFile)
    com = JPackage('com')

    # main code
    sc = com.faacets.Core.Scenario(configuration_to_faacets(A_configuration,
                                                            B_configuration))
    representation = com.faacets.Core.Representation('NCRepresentation')
    ope = com.faacets.SDP.OperatorElements(['', 'A', 'AB'])
    pts = com.faacets.SDP.PartialTransposes([])
    vec = com.faacets.Core.QVector(JArray(JDouble)(coefficients))
    expr = com.faacets.Core.Expr(sc, representation, vec)
    sdp = com.faacets.SDP.CorrSDP(sc, ope, pts, expr.symmetryGroup())
    M = np.array(sdp.indexArray())
    ncIndices = np.array(sdp.ncIndices())
    shutdownJVM()
    return M, ncIndices


class FaacetsRelaxation(Relaxation):

    """Class for wrapping around a Faacets relaxation.

    """

    def get_relaxation(self, A_configuration, B_configuration, I):
        """Get the sparse SDP relaxation of a Bell inequality.

        :param A_configuration: The definition of measurements of Alice.
        :type A_configuration: list of list of int.
        :param B_configuration: The definition of measurements of Bob.
        :type B_configuration: list of list of int.
        :param I: The matrix describing the Bell inequality in the
                  Collins-Gisin picture.
        :type I: list of list of int.
        """
        coefficients = collinsgisin_to_faacets(I)
        M, ncIndices = get_faacets_moment_matrix(A_configuration,
                                                 B_configuration, coefficients)
        self.n_vars = M.max() - 1
        bs = len(M)  # The block size
        self.block_struct = [bs]
        self.F = lil_matrix((bs**2, self.n_vars + 1))
        # Constructing the internal representation of the constraint matrices
        # See Section 2.1 in the SDPA manual and also Yalmip's internal
        # representation
        for i in range(bs):
            for j in range(i, bs):
                if M[i, j] != 0:
                    self.F[i*bs+j, abs(M[i, j])-1] = copysign(1, M[i, j])
        self.obj_facvar = [0 for _ in range(self.n_vars)]
        for i in range(1, len(ncIndices)):
            self.obj_facvar[abs(ncIndices[i])-2] += \
                copysign(1, ncIndices[i])*coefficients[i]
