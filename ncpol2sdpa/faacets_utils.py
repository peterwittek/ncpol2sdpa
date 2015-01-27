# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 11:41:52 2015

@author: wittek
"""
import numpy as np
import glob
import os

def collinsgisin_to_faacets(I):
    coefficients = []
    for j in range(len(I[0])):
        for i in range(len(I)):
            coefficients.append(I[i][j])
    return np.array(coefficients, dtype='d')

def configuration_to_faacets(A_configuration, B_configuration):
    tmpA = str(A_configuration).replace('[','(').replace(']',')').replace(',','')
    tmpB = str(B_configuration).replace('[','(').replace(']',')').replace(',','')
    return '[' + tmpA + ' ' + tmpB +']'

def get_faacets_moment_matrix(A_configuration, B_configuration, coefficients):
    from jpype import startJVM, shutdownJVM, JPackage, getDefaultJVMPath, JArray, JDouble
    # find the JAR file and start the JVM
    jarFiles = glob.glob('faacets-*.jar')
    assert(len(jarFiles) == 1)
    jarFile = os.path.join(os.getcwd(), jarFiles[0])
    startJVM(getDefaultJVMPath(), "-Djava.class.path="+jarFile)
    com = JPackage('com')

    # main code
    sc = com.faacets.Core.Scenario(configuration_to_faacets(A_configuration, B_configuration))
    repr = com.faacets.Core.Representation('NCRepresentation')
    ope = com.faacets.SDP.OperatorElements(['', 'A', 'AB'])
    pts = com.faacets.SDP.PartialTransposes([])
    CHSH = JArray(JDouble)(coefficients)

    vec = com.faacets.Core.QVector(CHSH)
    expr = com.faacets.Core.Expr(sc, repr, vec)
    sdp = com.faacets.SDP.CorrSDP(sc, ope, pts, expr.symmetryGroup())
    M = np.array(sdp.indexArray())
    ncIndices = np.array(sdp.ncIndices())
    shutdownJVM()
    return M, ncIndices
