# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:24:48 2015

@author: Peter Wittek
"""
from __future__ import division, print_function
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
    :param parameters: Optional symbolic variables for which moments are not
                       generated.
    :type parameters: list of :class:`sympy.physics.quantum.operator.Operator`
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

    Attributes:
      - `monomial_sets`: The monomial sets that generate the moment matrix
      blocks.

      - `monomial_index`: Dictionary that maps monomials to SDP variables.

      - `constraints`: The complete set of constraints after preprocesssing.

      - `primal`: The primal optimal value.

      - `dual`: The dual optimal value.

      - `x_mat`: The primal solution matrix.

      - `y_mat`: The dual solution matrix.

      - `solution_time`: The amount of time taken to solve the relaxation.

      - `status`: The solution status of the relaxation.

    """

    def __init__(self, variables, parameters=None, verbose=0, normalized=True,
                 ppt=False, parallel=False):
        super(MoroderHierarchy, self).__init__(variables, parameters, verbose,
                                               normalized, parallel)
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

    def _calculate_block_structure(self, inequalities, equalities,
                                   momentinequalities, momentequalities,
                                   extramomentmatrix, removeequalities,
                                   block_struct=None):
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
            _calculate_block_structure(inequalities, equalities,
                                       momentinequalities, momentequalities,
                                       extramomentmatrix,
                                       removeequalities,
                                       block_struct=block_struct)

    def _estimate_n_vars(self):
        self.n_vars = 0
        if self.parameters is not None:
            self.n_vars = len(self.parameters)
        n_monomials = len(self.monomial_sets[0])*len(self.monomial_sets[1])
        self.n_vars += int(n_monomials * (n_monomials + 1) / 2)
