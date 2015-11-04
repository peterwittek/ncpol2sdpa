# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:24:48 2015

@author: Peter Wittek
"""
from .sdp_relaxation import SdpRelaxation


class MoroderHierarchy(SdpRelaxation):

    """Class for obtaining a step in the Moroder hierarchy
    (`doi:10.1103/PhysRevLett.111.030501 <http://dx.doi.org/10.1103/PhysRevLett.111.030501>`_)
    :param variables: Commutative or noncommutative, Hermitian or nonhermiatian
                      variables, possibly a list of list of variables if the
                      hierarchy is not NPA.
    :type variables: list of :class:`sympy.physics.quantum.operator.Operator`
                     or
                     :class:`sympy.physics.quantum.operator.HermitianOperator`
                     or a list of list.
    :param nonrelaxed: Optional variables which are not to be relaxed.
    :type nonrelaxed: list of :class:`sympy.physics.quantum.operator.Operator`
                     or
                     :class:`sympy.physics.quantum.operator.HermitianOperator`
                     or a list of list.
    :param verbose: Optional parameter for level of verbosity:

                       * 0: quiet
                       * 1: verbose
                       * 2: debug level
    :type verbose: int.
    :param normalized: Optional parameter for changing the normalization of
                       states over which the optimization happens. Turn it off
                       if further processing is done on the SDP matrix before
                       solving it.
    :type normalized: bool.
    :param ppt: Optional parameter to impose a partial positivity constraint
                on the moment matrix.
    :type ppt: bool.
    """

    def __init__(self, variables, nonrelaxed=None, verbose=0, normalized=True,
                 ppt=False):
        super(MoroderHierarchy, self).__init__(variables, nonrelaxed, verbose,
                                               normalized)
        self.ppt = ppt

    def _generate_all_moment_matrix_blocks(self, n_vars, block_index):
        processed_entries = 0
        n_vars, block_index, _ = \
            self._generate_moment_matrix(n_vars, block_index,
                                         processed_entries,
                                         self.monomial_sets[0],
                                         self.monomial_sets[1],
                                         ppt=self.ppt)
        return n_vars, block_index

    def _calculate_block_structure(self, inequalities, equalities, bounds,
                                   psd, extramomentmatrix, removeequalities,
                                   localizing_monomial_sets):
        """Calculates the block_struct array for the output file.
        """
        block_struct = []
        if self.verbose > 0:
            print("Calculating block structure...")
        block_struct.append(len(self.monomial_sets[0]) *
                            len(self.monomial_sets[1]))
        if extramomentmatrix is not None:
            for _ in extramomentmatrix:
                block_struct.append(len(self.monomial_sets[0]) *
                                    len(self.monomial_sets[1]))
        super(MoroderHierarchy, self).\
            _calculate_block_structure(inequalities, equalities, bounds,
                                       psd, extramomentmatrix,
                                       removeequalities,
                                       localizing_monomial_sets,
                                       block_struct=block_struct)

    def _estimate_n_vars(self):
        self.n_vars = 0
        if self.nonrelaxed is not None:
            self.n_vars = len(self.nonrelaxed)
        n_monomials = len(self.monomial_sets[0])*len(self.monomial_sets[1])
        self.n_vars += int(n_monomials * (n_monomials + 1) / 2)
