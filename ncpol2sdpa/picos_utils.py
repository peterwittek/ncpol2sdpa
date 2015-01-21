# -*- coding: utf-8 -*-
"""
The module contains helper functions to work with PICOS.

Created on Wed Dec 10 18:33:34 2014

@author: Peter Wittek
"""
from math import sqrt
from .sdpa_utils import convert_row_to_sdpa_index

def _row_to_affine_expression(row_vector, F_struct, row_offsets,
                              block_of_last_moment_matrix, block_struct, X):
    """Helper function to create an affine expression based on the variables
    in the moment matrices.
    """
    if row_vector.getnnz() > 0:
        affine_expression = 0
        row, columns = row_vector.nonzero()
        for k in columns:
            if k == 0:
                affine_expression += row_vector[row[0], k]
            else:
                row0 = F_struct[:row_offsets[block_of_last_moment_matrix+1],
                                k].nonzero()[0][0]
                block, i, j = convert_row_to_sdpa_index(block_struct,
                                                        row_offsets, row0)
                affine_expression += row_vector[row[0], k] * X[block][i, j]
        return affine_expression
    else:
        return None

def _objective_to_affine_expression(objective, F_struct, row_offsets,
                                    block_of_last_moment_matrix, block_struct,
                                    X):
    """Helper function to create an affine expression based on the variables
    in the moment matrices from the dense vector describing the objective.
    """
    affine_expression = 0
    for k, v in enumerate(objective):
        if v != 0:
            row0 = F_struct[0:row_offsets[block_of_last_moment_matrix+1],
                            k+1].nonzero()[0][0]
            block, i, j = convert_row_to_sdpa_index(block_struct, row_offsets,
                                                    row0)
            affine_expression += v * X[block][i, j]
    return affine_expression

def inplace_partial_transpose(affine_expression):
    import cvxopt as cvx
    import picos as pic
    if isinstance(affine_expression, pic.Variable):
        raise Exception('inplace_transpose should not be called on a Variable object')
    for k in affine_expression.factors:
        I0 = []
        bsize = affine_expression.size[0]
        subsize = int(sqrt(bsize))
        J = affine_expression.factors[k].J
        V = affine_expression.factors[k].V
        for i in affine_expression.factors[k].I:
            row, column = divmod(i, bsize)
            row_block, lrow = divmod(row, subsize)
            column_block, lcolumn = divmod(column, subsize)
            row = column_block*subsize + lrow
            column = row_block*subsize + lcolumn
            I0.append(row*bsize + column)
        J = affine_expression.factors[k].J
        V = affine_expression.factors[k].V
        affine_expression.factors[k] = cvx.spmatrix(V, I0, J, affine_expression.factors[k].size)
    if affine_expression.constant is not None:
        affine_expression.constant = cvx.matrix(affine_expression.constant,
                                                affine_expression.size).T[:]
    affine_expression._size = (affine_expression.size[1],
                               affine_expression.size[0])
    if (('*' in affine_expression.affstring()) or ('/' in affine_expression.affstring())
            or ('+' in affine_expression.affstring()) or ('-' in affine_expression.affstring())):
        affine_expression.string = '( '+affine_expression.string+' ).Tx'
    else:
        affine_expression.string += '.Tx'

def partial_transpose(affine_expression):
    """Returns the partial transpose of a moment matrix
    """
    selfcopy = affine_expression.copy()
    inplace_partial_transpose(selfcopy)
    return selfcopy


def convert_to_picos_extra_moment_matrix(sdpRelaxation):
    """Convert an SDP relaxation to a PICOS problem, returning the moment
    matrix with positive semidefinite imposed, and an extra copy of the moment
    matrix with no constraints on it.

    :param sdpRelaxation: The SDP relaxation to convert.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.

    :returns: :class:`picos.Problem`.
    """
    import picos as pic
    import cvxopt as cvx
    P = pic.Problem()
    block_size = sdpRelaxation.block_struct[0]
    X = P.add_variable('X', (block_size, block_size))
    Y = P.add_variable('Y', (block_size, block_size))
    row_offset = 0
    for block_size in sdpRelaxation.block_struct:
        x, Ix, Jx = [], [], []
        c, Ic, Jc = [], [], []
        for i, row in enumerate(sdpRelaxation.F_struct.rows[row_offset:row_offset+block_size**2]):
            for j, column in enumerate(row):
                if column > 0:
                    x.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                    Ix.append(i)
                    Jx.append(column-1)
                    i0 = (i/block_size)+(i%block_size)*block_size
                    if i != i0:
                        x.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                        Ix.append(i0)
                        Jx.append(column-1)

                else:
                    c.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                    #Note that a constant term can only possibly occur in a
                    #top-left corner
                    Ic.append(0)
                    Jc.append(0)
        permutation = cvx.spmatrix(x, Ix, Jx)
        constant = cvx.spmatrix(c, Ic, Jc, (block_size, block_size))
        constraint = X#.copy()
        for k in constraint.factors:
            constraint.factors[k] = permutation
        constraint._size = (block_size, block_size)
        P.add_constraint(constant + constraint>>0)
        if block_size == sdpRelaxation.block_struct[0]:
            for k in Y.factors:
                Y.factors[k] = permutation
        row_offset += block_size**2
    x, Ix, Jx = [], [], []
    for k, val in enumerate(sdpRelaxation.obj_facvar):
        if val != 0:
            x.append(val)
            Ix.append(0)
            Jx.append(k)
    permutation = cvx.spmatrix(x, Ix, Jx)
    objective = X.copy()
    for k in objective.factors:
        objective.factors[k] = permutation
    objective._size = (1, 1)
    P.set_objective('min', objective)
    return P, X, Y

def write_picos_to_sdpa(problem, filename, extended=False):
    """
    Write a problem to sdpa format

    :param problem: The PICOS problem to convert.
    :type problem: :class:`picos.Problem`.
    :param filename: The name of the file. It must have the suffix ".dat-s"
    :type filename: str.
    :param extended: Whether to use extended format.
    :type problem: bool.

    """
    import cvxopt as cvx
    import picos as pic
    #--------------------#
    # makes the instance #
    #--------------------#
    if not any(problem.cvxoptVars.values()):
        problem._make_cvxopt_instance()

    dims = {}
    dims['s'] = [int(sqrt(Gsi.size[0])) for Gsi in problem.cvxoptVars['Gs']]
    dims['l'] = problem.cvxoptVars['Gl'].size[0]
    dims['q'] = [Gqi.size[0] for Gqi in problem.cvxoptVars['Gq']]
    G = problem.cvxoptVars['Gl']
    h = problem.cvxoptVars['hl']

    # handle the equalities as 2 ineq for smcp
    if problem.cvxoptVars['A'].size[0] > 0:
        G = cvx.sparse([G, problem.cvxoptVars['A']])
        G = cvx.sparse([G, -problem.cvxoptVars['A']])
        h = cvx.matrix([h, problem.cvxoptVars['b']])
        h = cvx.matrix([h, -problem.cvxoptVars['b']])
        dims['l'] += (2*problem.cvxoptVars['A'].size[0])

    for i in range(len(dims['q'])):
        G = cvx.sparse([G, problem.cvxoptVars['Gq'][i]])
        h = cvx.matrix([h, problem.cvxoptVars['hq'][i]])


    for i in range(len(dims['s'])):
        G = cvx.sparse([G, problem.cvxoptVars['Gs'][i]])
        h = cvx.matrix([h, problem.cvxoptVars['hs'][i]])

    #Remove the lines in A and b corresponding to 0==0
    JP = list(set(problem.cvxoptVars['A'].I))
    IP = range(len(JP))
    VP = [1]*len(JP)

    #is there a constraint of the form 0==a(a not 0) ?
    if any([b for (i, b) in enumerate(problem.cvxoptVars['b']) if i not in JP]):
        raise Exception('infeasible constraint of the form 0=a')

    P = cvx.spmatrix(VP, IP, JP, (len(IP), problem.cvxoptVars['A'].size[0]))
    problem.cvxoptVars['A'] = P*problem.cvxoptVars['A']
    problem.cvxoptVars['b'] = P*problem.cvxoptVars['b']
    c = problem.cvxoptVars['c']

    #-----------------------------------------------------------#
    # make A,B,and blockstruct.                                 #
    # This code is a modification of the conelp function in smcp#
    #-----------------------------------------------------------#
    from cvxopt import sparse, spmatrix
    Nl = dims['l']
    Nq = dims['q']
    Ns = dims['s']
    if not Nl:
        Nl = 0

    P_m = G.size[1]

    P_b = -c
    P_blockstruct = []
    if Nl:
        P_blockstruct.append(-Nl)
    if extended:
        for i in Nq:
            P_blockstruct.append(i*1j)
    else:
        for i in Nq:
            P_blockstruct.append(i)
    for i in Ns:
        P_blockstruct.append(i)

    def to_lower_triangular_spmatrix(n, u):
        I = []
        J = []
        V = []
        for k, index in enumerate(u.I):
            j, i = divmod(index, n)
            if j <= i:
                I.append(i)
                J.append(j)
                V.append(u.V[k])
        return spmatrix(V, I, J, (n, n))


    #write data
    #add extension
    if extended:
        if filename[-7:] != '.dat-sx':
            filename += '.dat-sx'
    else:
        if filename[-6:] != '.dat-s':
            filename += '.dat-s'
    #check lp compatibility
    if (problem.numberQuadConstraints + problem.numberLSEConstraints) > 0:
        if problem.options['convert_quad_to_socp_if_needed']:
            pcop = problem.copy()
            pcop.convert_quad_to_socp()
            pcop._write_sdpa(filename, extended)
            return
        else:
            raise pic.QuadAsSocpError('Problem should not have quad or gp constraints. '+
                                      'Try to convert the problem to an SOCP with the function convert_quad_to_socp()')
    #open file
    f = open(filename, 'w')
    f.write('"file '+filename+' generated by picos"\n')
    print('writing problem in '+filename+'...')
    f.write(str(problem.numberOfVars)+' = number of vars\n')
    f.write(str(len(P_blockstruct))+' = number of blocs\n')
    #bloc structure
    f.write(str(P_blockstruct).replace('[', '(').replace(']', ')'))
    f.write(' = BlocStructure\n')
    #c vector (objective)
    f.write(str(list(-P_b)).replace('[', '{').replace(']', '}'))
    f.write('\n')
    #coefs
    from itertools import izip
    for k in range(P_m+1):
        if k != 0:
            v = sparse(G[:, k-1])
        else:
            v = +sparse(h)

        ptr = 0
        Ak = []
        # lin. constraints
        if Nl:
            u = v[:Nl]
            I = u.I
            Ak.append(spmatrix(u.V, I, I, (Nl, Nl)))
            ptr += Nl

        # SOC constraints
        for nq in Nq:
            u0 = v[ptr]
            u1 = v[ptr+1:ptr+nq]
            tmp = spmatrix(u1.V, [nq-1 for j in xrange(len(u1))], u1.I, (nq, nq))
            if not u0 == 0.0:
                tmp += spmatrix(u0, xrange(nq), xrange(nq), (nq, nq))
            Ak.append(tmp)
            ptr += nq

        # SDP constraints
        for ns in Ns:
            Ak.append(to_lower_triangular_spmatrix(ns, v[ptr:ptr+ns**2]))
            ptr += ns**2

        for b, B in enumerate(Ak):
            for i, j, v in izip(B.I, B.J, B.V):
                f.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(k, b+1, j+1, i+1, -v))

    #binaries an integers in extended format
    if extended:
        #general integers
        f.write("Generals\n")
        for _, v in problem.variables.iteritems():
            if v.vtype == 'integer':
                for i in xrange(v.startIndex, v.endIndex):
                    f.write(str(i+1)+'\n')
            if v.vtype == 'semiint' or v.vtype == 'semicont':
                raise Exception('semiint and semicont variables not handled by this LP writer')
        #binary variables
        f.write("Binaries\n")
        for _, v in problem.variables.iteritems():
            if v.vtype == 'binary':
                for i in xrange(v.startIndex, v.endIndex):
                    f.write(str(i+1)+'\n')

    f.close()


def _convert_to_picos_compact(sdpRelaxation):
    """Convert an SDP relaxation to a PICOS problem such that the exported
    .dat-s file is extremely sparse, there is not penalty imposed in terms of
    SDP variables or number of constraints. This conversion can be used for
    imposing extra constraints on the moment matrix, such as partial transpose.

    :param sdpRelaxation: The SDP relaxation to convert.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.

    :returns: :class:`picos.Problem`.
    """
    import picos as pic
    import cvxopt as cvx
    P = pic.Problem()
    block_size = sdpRelaxation.block_struct[0]
    X = P.add_variable('X', (block_size, block_size))
    row_offset = 0
    for block_size in sdpRelaxation.block_struct:
        x, Ix, Jx = [], [], []
        c, Ic, Jc = [], [], []
        for i, row in enumerate(sdpRelaxation.F_struct.rows[row_offset:row_offset+block_size**2]):
            for j, column in enumerate(row):
                if column > 0:
                    x.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                    Ix.append(i)
                    Jx.append(column-1)
                    i0 = (i/block_size)+(i%block_size)*block_size
                    if i != i0:
                        x.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                        Ix.append(i0)
                        Jx.append(column-1)

                else:
                    c.append(sdpRelaxation.F_struct.data[row_offset+i][j])
                    #Note that a constant term can only possibly occur in a
                    #top-left corner
                    Ic.append(0)
                    Jc.append(0)
        permutation = cvx.spmatrix(x, Ix, Jx)
        constant = cvx.spmatrix(c, Ic, Jc, (block_size, block_size))
        constraint = X.copy()
        for k in constraint.factors:
            constraint.factors[k] = permutation
        constraint._size = (block_size, block_size)
        P.add_constraint(constant + constraint>>0)
        row_offset += block_size**2
    x, Ix, Jx = [], [], []
    for k, val in enumerate(sdpRelaxation.obj_facvar):
        if val != 0:
            x.append(val)
            Ix.append(0)
            Jx.append(k)
    permutation = cvx.spmatrix(x, Ix, Jx)
    objective = X.copy()
    for k in objective.factors:
        objective.factors[k] = permutation
    objective._size = (1, 1)
    P.set_objective('min', objective)
    return P

def convert_to_picos(sdpRelaxation):
    """Convert an SDP relaxation to a PICOS problem.

    :param sdpRelaxation: The SDP relaxation to convert.
    :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.

    :returns: :class:`picos.Problem`.
    """
    if sdpRelaxation.hierarchy != "nieto-silleras":
        return _convert_to_picos_compact(sdpRelaxation)
    import picos as pic
    P = pic.Problem()
    row_offsets = [0]
    block_of_last_moment_matrix = 0
    for block, block_size in enumerate(sdpRelaxation.block_struct):
        if block > 0 and block_size < sdpRelaxation.block_struct[block]:
            block_of_last_moment_matrix = block - 1
        row_offsets.append(row_offsets[block]+block_size ** 2)
    X = []
    # First we work on the moment matrices
    for block in range(block_of_last_moment_matrix+1):
        block_size = sdpRelaxation.block_struct[block]
        X.append(P.add_variable('X%s' % block, (block_size, block_size),
                                'symmetric'))
        P.add_constraint(X[block] >> 0)
        start = row_offsets[block]
        end = row_offsets[block] + block_size ** 2
        # If there is a constant term, the moment matrix is normalized
        if sdpRelaxation.F_struct[start:end, 0].getnnz() > 0:
            P.add_constraint(X[block][0, 0] == 1)
        # Here we define the internal symmetries of the moment matrix
        for k in range(1, sdpRelaxation.n_vars+1):
            if sdpRelaxation.F_struct[start:end, k].getnnz() > 1:
                row0 = sdpRelaxation.F_struct[start:end, k].nonzero()[0][0]
                _, i1, j1 = \
                    convert_row_to_sdpa_index(
                        sdpRelaxation.block_struct,
                        row_offsets,
                        row0)
                for row in sdpRelaxation.F_struct[start:end, k].nonzero()[0][1:]:
                    _, i2, j2 = \
                        convert_row_to_sdpa_index(
                            sdpRelaxation.block_struct,
                            row_offsets,
                            row)
                    if not (i1 == i2 and j1 == j2):
                        P.add_constraint(X[block][i2, j2] == X[block][i1, j1])
    # Next we proceed to the constraints
    for block in range(block_of_last_moment_matrix+1,
                       len(sdpRelaxation.block_struct)):
        block_size = sdpRelaxation.block_struct[block]
        Y = P.add_variable('Y%s' % block, (block_size, block_size),
                           'symmetric')
        P.add_constraint(Y >> 0)
        start = row_offsets[block]
        end = row_offsets[block] + block_size ** 2
        for row in range(start, end):
            row_vector = sdpRelaxation.F_struct.getrow(row)
            affine_expression = \
              _row_to_affine_expression(row_vector, sdpRelaxation.F_struct,
                                        row_offsets,
                                        block_of_last_moment_matrix,
                                        sdpRelaxation.block_struct, X)
            if affine_expression is not None:
                i, j = divmod(row-start, block_size)
                P.add_constraint(Y[i, j] == affine_expression)
    affine_expression = \
      _objective_to_affine_expression(sdpRelaxation.obj_facvar,
                                      sdpRelaxation.F_struct, row_offsets,
                                      block_of_last_moment_matrix,
                                      sdpRelaxation.block_struct, X)
    P.set_objective('min', affine_expression)
    return P
