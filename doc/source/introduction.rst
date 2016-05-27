************
Introduction
************
Ncpol2sdpa solves global polynomial optimization problems of either commutative variables or noncommutative operators through a semidefinite programming (SDP) relaxation. The optimization problem can be unconstrained or constrained by equalities and inequalities, and also by constraints on the moments. The objective is to be able to solve large scale optimization problems. Example applications include:

- When the polynomial optimization problem is defined over commutative variables, the generated SDP hierarchy is identical to `Lasserre's <http://dx.doi.org/10.1137/S1052623400366802>`_. In this case, the functionality resembles the MATLAB toolboxes `Gloptipoly <http://homepages.laas.fr/henrion/software/gloptipoly/>`_, and, with the chordal extension, `SparsePOP <http://sparsepop.sourceforge.net/>`_.
- `Relaxations <http://nbviewer.ipython.org/github/peterwittek/ipython-notebooks/blob/master/Parameteric%20and%20Bilevel%20Polynomial%20Optimization%20Problems.ipynb>`_ of `parametric <http://dx.doi.org/10.1137/090759240>`_ and `bilevel <http://arxiv.org/abs/1506.02099>`_ polynomial optimization problems.
- When the polynomials are over noncommutative operators, the generated SDP is a step in the Navascués-Pironio-Acín (NPA) hierarchy. The most notable example is calculating the `maximum quantum violation <http:/dx.doi.org/10.1103/PhysRevLett.98.010401>`_ of `Bell inequalities <http://peterwittek.com/2014/06/quantum-bound-on-the-chsh-inequality-using-sdp/>`_, also in `multipartite scenarios <http://peterwittek.github.io/multipartite_entanglement/>`_.
- `Nieto-Silleras <http://dx.doi.org/10.1088/1367-2630/16/1/013035>`_ hierarchy for `quantifying randomness <http://peterwittek.com/2014/11/the-nieto-silleras-and-moroder-hierarchies-in-ncpol2sdpa/>`_ and for `calculating maximum guessing probability <http://nbviewer.ipython.org/github/peterwittek/ipython-notebooks/blob/master/Optimal%20randomness%20generation%20from%20entangled%20quantum%20states.ipynb>`_.
- `Moroder <http://dx.doi.org/10.1103/PhysRevLett.111.030501>`_ hierarchy to enable PPT-style and other additional constraints.
- Sums-of-square (SOS) decomposition based on the dual solution.
- `Ground-state energy problems <http://dx.doi.org/10.1137/090760155>`_: bosonic and `fermionic systems <http://nbviewer.ipython.org/github/peterwittek/ipython-notebooks/blob/master/Comparing_DMRG_ED_and_SDP.ipynb>`_, Pauli spin operators. This methodology closely resembles the reduced density matrix (RDM) method.
- `Hierarchy for quantum steering <http://dx.doi.org/10.1103/physrevlett.115.210401>`_.

The implementation has an intuitive syntax for entering problems and it scales for a larger number of noncommutative variables using a sparse representation of the SDP problem. Further details are found in the following paper:

- Peter Wittek. Algorithm 950: Ncpol2sdpa---Sparse Semidefinite Programming Relaxations for Polynomial Optimization Problems of Noncommuting Variables. *ACM Transactions on Mathematical Software*, 41(3), 21, 2015. DOI: `10.1145/2699464 <http://dx.doi.org/10.1145/2699464>`_. arXiv:`1308.6029 <http://arxiv.org/abs/1308.6029>`_.

The module was used for calculations in the following papers:

- Antonio Acín, Stefano Pironio, Tamás Vértesi, and Peter Wittek. Optimal randomness certification from one entangled bit. *Physical Review A*, 93, 040102, 2016. DOI:`10.1103/PhysRevA.93.040102 <http://dx.doi.org/10.1103/PhysRevA.93.040102>`_.  arXiv:`1505.03837 <http://arxiv.org/abs/1505.03837>`_. `Notebook <http://nbviewer.ipython.org/github/peterwittek/ipython-notebooks/blob/master/Optimal%20randomness%20generation%20from%20entangled%20quantum%20states.ipynb>`_.

- Ivan Šupić, Matty J. Hoban. Self-testing through EPR-steering. arXiv:`1601.01552 <http://arxiv.org/abs/1601.01552>`_.

- Peter Wittek, Sándor Darányi, Gustaf Nelhans. Ruling Out Static Latent Homophily in Citation Networks. arXiv:`1605.08185 <http://arxiv.org/abs/1605.08185>`_. `Notebook <https://nbviewer.jupyter.org/github/peterwittek/ipython-notebooks/blob/master/Citation_Network_SDP.ipynb>`_.

Copyright and License
=====================
Ncpol2sdpa is free software; you can redistribute it and/or modify it under the terms of the `GNU General Public License <http://www.gnu.org/licenses/gpl-3.0.html>`_ as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

Ncpol2sdpa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the `GNU General Public License <http://www.gnu.org/licenses/gpl-3.0.html>`_ for more details.


Acknowledgment
==============
This work is supported by the European Commission Seventh Framework Programme under Grant Agreement Number FP7-601138 `PERICLES <http://pericles-project.eu/>`_, by the `Red Espanola de Supercomputacion <http://www.bsc.es/RES>`_ grants number FI-2013-1-0008 and  FI-2013-3-0004, and by the `Swedish National Infrastructure for Computing <http://www.snic.se/>`_ projects SNIC 2014/2-7 and SNIC 2015/1-162.
