Ncpol2sdpa
==========
Ncpol2sdpa solves global polynomial optimization problems of either commutative variables or noncommutative operators through a semidefinite programming (SDP) relaxation. The optimization problem can be unconstrained or constrained by equalities and inequalities, and also by constraints on the moments. The objective is to be able to solve large scale optimization problems. Example applications include:

- When the polynomial optimization problem is defined over commutative variables, the generated SDP hierarchy is identical to `Lasserre's <http://dx.doi.org/10.1137/S1052623400366802>`_. In this case, the functionality resembles the MATLAB toolboxes `Gloptipoly <http://homepages.laas.fr/henrion/software/gloptipoly/>`_, and, with the chordal extension, `SparsePOP <http://sparsepop.sourceforge.net/>`_.
- `Relaxations <http://nbviewer.ipython.org/github/peterwittek/ipython-notebooks/blob/master/Parameteric%20and%20Bilevel%20Polynomial%20Optimization%20Problems.ipynb>`_ of `parametric <http://dx.doi.org/10.1137/090759240>`_ and `bilevel <http://arxiv.org/abs/1506.02099>`_ polynomial optimization problems.
- When the polynomials are over noncommutative operators, the generated SDP is a step in the Navascués-Pironio-Acín (NPA) hierarchy. The most notable example is calculating the `maximum quantum violation <http:/dx.doi.org/10.1103/PhysRevLett.98.010401>`_ of `Bell inequalities <http://peterwittek.com/2014/06/quantum-bound-on-the-chsh-inequality-using-sdp/>`_, also in `multipartite scenarios <http://peterwittek.github.io/multipartite_entanglement/>`_.
- `Nieto-Silleras <http://dx.doi.org/10.1088/1367-2630/16/1/013035>`_ hierarchy for `quantifying randomness <http://peterwittek.com/2014/11/the-nieto-silleras-and-moroder-hierarchies-in-ncpol2sdpa/>`_ and for `calculating maximum guessing probability <http://nbviewer.ipython.org/github/peterwittek/ipython-notebooks/blob/master/Optimal%20randomness%20generation%20from%20entangled%20quantum%20states.ipynb>`_.
- `Moroder <http://dx.doi.org/10.1103/PhysRevLett.111.030501>`_ hierarchy to enable PPT-style and other additional constraints.
- Sums-of-square (SOS) decomposition based on the dual solution.
- `Ground-state energy problems <http://dx.doi.org/10.1137/090760155>`_: bosonic and `fermionic systems <http://nbviewer.ipython.org/github/peterwittek/ipython-notebooks/blob/master/Comparing_DMRG_ED_and_SDP.ipynb>`_, Pauli spin operators. This methodology closely resembles the reduced density matrix (RDM) method.
- `Hierarchy for quantum steering <http://dx.doi.org/10.1103/physrevlett.115.210401>`_.

The implementation has an intuitive syntax for entering problems and it scales for a larger number of noncommutative variables using a sparse representation of the SDP problem.  Further details are found in the following paper:

- Peter Wittek. Algorithm 950: Ncpol2sdpa---Sparse Semidefinite Programming Relaxations for Polynomial Optimization Problems of Noncommuting Variables. *ACM Transactions on Mathematical Software*, 41(3), 21, 2015. DOI: `10.1145/2699464 <http://dx.doi.org/10.1145/2699464>`_. arXiv:`1308.6029 <http://arxiv.org/abs/1308.6029>`_.

The module was used for calculations in the following papers:

- Antonio Acín, Stefano Pironio, Tamás Vértesi, and Peter Wittek. Optimal randomness certification from one entangled bit. *Physical Review A*, 93, 040102, 2016. DOI:`10.1103/PhysRevA.93.040102 <https://dx.doi.org/10.1103/PhysRevA.93.040102>`_.  arXiv:`1505.03837 <http://arxiv.org/abs/1505.03837>`_. `Notebook <https://nbviewer.ipython.org/github/peterwittek/ipython-notebooks/blob/master/Optimal%20randomness%20generation%20from%20entangled%20quantum%20states.ipynb>`_.

- Ivan Šupić, Matty J. Hoban. Self-testing through EPR-steering. arXiv:`1601.01552 <http://arxiv.org/abs/1601.01552>`_.

- Peter Wittek, Sándor Darányi, Gustaf Nelhans. Ruling Out Static Latent Homophily in Citation Networks. arXiv:`1605.08185 <http://arxiv.org/abs/1605.08185>`_. `Notebook <https://nbviewer.jupyter.org/github/peterwittek/ipython-notebooks/blob/master/Citation_Network_SDP.ipynb>`_.

Dependencies
============
The implementation requires `SymPy <http://sympy.org/>`_ and `Numpy <http://www.numpy.org/>`_. The code is compatible with both Python 2 and 3. While the default CPython interpreter is sufficient for small to medium-scale problems, execution time becomes excessive for larger problems. The code is compatible with Pypy. Using it yields a 10-20x speedup. If you use Pypy, you will need the `Pypy fork of Numpy <https://bitbucket.org/pypy/numpy/>`_.

By default, Ncpol2sdpa does not require a solver, but then it will not be able to solve a generated relaxation either. Install any supported solver and it will be detected automatically.

Optional dependencies include:

- `SDPA <http://sdpa.sourceforge.net/>`_ is a possible target solver.
- `SciPy <http://scipy.org/>`_ yields faster execution with the default CPython interpreter.
- `PICOS <http://picos.zib.de/>`_ is necessary for using the Cvxopt solver and for converting the problem to a PICOS instance.
- `MOSEK <http://www.mosek.com/>`_ Python module is necessary to work with the MOSEK solver.
- `CVXPY <http://cvxpy.org/>`_ is required for converting the problem to or by solving it by CVXPY.
- `Cvxopt <http://cvxopt.org/>`_ is required by both Chompack and PICOS.
- `Chompack <http://chompack.readthedocs.io/en/latest/>`_ improves the sparsity of the chordal graph extension.

Usage
=====
Documentation is available on `Read the Docs <http://ncpol2sdpa.readthedocs.io/>`_. The following code replicates the toy example from Pironio, S.; Navascués, M. & Acín, A. Convergent relaxations of polynomial optimization problems with noncommuting variables SIAM Journal on Optimization, SIAM, 2010, 20, 2157-2180.

.. code:: python

    from ncpol2sdpa import generate_operators, SdpRelaxation

    # Number of operators
    n_vars = 2
    # Level of relaxation
    level = 2

    # Get Hermitian operators
    X = generate_operators('X', n_vars, hermitian=True)

    # Define the objective function
    obj = X[0] * X[1] + X[1] * X[0]

    # Inequality constraints
    inequalities = [-X[1] ** 2 + X[1] + 0.5 >= 0]

    # Simple monomial substitutions
    substitutions = {X[0]**2: X[0]}

    # Obtain SDP relaxation
    sdpRelaxation = SdpRelaxation(X)
    sdpRelaxation.get_relaxation(level, objective=obj, inequalities=inequalities,
                                 substitutions=substitutions)
    sdpRelaxation.solve()
    print(sdpRelaxation.primal, sdpRelaxation.dual, sdpRelaxation.status)

Further examples are found in the documentation.

Installation
============
The code is available on PyPI, hence it can be installed by

``$ sudo pip install ncpol2sdpa``

If you want the latest git version, follow the standard procedure for installing Python modules after cloning the repository:

``$ sudo python setup.py install``

Acknowledgment
==============
This work is supported by the European Commission Seventh Framework Programme under Grant Agreement Number FP7-601138 `PERICLES <http://pericles-project.eu/>`_, by the `Red Espanola de Supercomputacion <http://www.bsc.es/RES>`_ grants number FI-2013-1-0008 and  FI-2013-3-0004, and by the `Swedish National Infrastructure for Computing <http://www.snic.se/>`_ projects SNIC 2014/2-7 and SNIC 2015/1-162.
