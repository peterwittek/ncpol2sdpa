Revision History
****************
**Version 1.11.0 (2016-06-23)**
  - New: Experimental new parallel computation of the moment matrix and the constraints.
  - New: CVXPY conversion with ``convert_to_cvxpy``. CVXPY is now also a valid solver.
  - New: The method ``get_dual`` returns the block in the dual solution corresponding to the requested constraint.
  - Changed: Deprecated optional parameter ``bounds`` was removed.
  - Fixed: Moments are correctly returned even if equalities are removed.
  - Fixed: Constants in PICOS conversion are added correctly irrespective of where they are in the matrices.
  - Fixed: PICOS conversion handles feasibility problems correctly.
  - Fixed: The optional parameter ``removeequalities=True`` handles equalities of SDP variables correctly.

**Version 1.10.3 (2016-02-26)**
  - Fixed: Problem with unexpanded moment equality constraints resolved.

**Version 1.10.2 (2016-02-03)**
  - New: Very efficient substitutions of moment equalities if one side of the equality is the moment of a monomial, and the other side is a constant.

**Version 1.10.1 (2016-01-29)**
  - Fixed: The moment equalities are removed correctly if asked.

**Version 1.10 (2015-12-08)**

  - New: The function ``generate_operators`` returns a list of operators from the ``sympy.physics.quantum`` submodule. This is the old behaviour of ``generate_variables``.
  - New: The ``SdpRelaxation`` class is now subscriptable. You can retrieve the value of polynomials in the solved relaxation in such way. Internally, it calls ``get_xmat_value`` with ``self``.
  - New: The convenience method ``solve()`` was added to the class ``SdpRelaxation``.
  - New: The convenience method ``write_to_file()`` was added to the class ``SdpRelaxation``.
  - New: The convenience method ``save_monomial_index()`` was added to the class ``SdpRelaxation``.
  - New: The convenience method ``find_solution_ranks()`` was added to the class ``SdpRelaxation``. It replaces the previous stand-alone ``find_rank_loop()`` function.
  - New: The conversion routines ``convert_to_picos`` and ``convert_to_mosek`` are also part of the class ``sdpRelaxation``.
  - New: The new method ``extract_dual_value()`` was added to the class ``SdpRelaxation`` to calculate the inner product of the coefficient matrix of an SDP variable with the dual solution.
  - New: The class ``RdmHierarchy`` was added to generate SDPs of the reduced density matrix method. Initial support for 1D spinless, translational invariant systems is included.
  - New: Better support for the steering hierarchy in a new class ``SteeringHierarchy``.
  - Changed: The function ``generate_variables`` now returns a list of ``sympy.Symbol`` variables if commutative variables are requested, and the default is commutative.
  - Changed: Many unnecessary user-facing functions were removed.
  - Changed: The SOS decomposition is now requested with ``get_sos_decomposition`` from the class ``SdpRelaxation``, and it returns a list of the SOS polynomials.
  - Changed: The optional parameter ``bounds`` for ``get_relaxation`` is deprececated, use the optional parameters ``momentinequalities`` and ``momentequalities`` instead.
  - Changed: Removed ``convert_to_picos_extra_moment_matrix`` and added optional parameter ``duplicate_moment_matrix`` to ``convert_to_picos`` to achieve the same effect.
  - Changed: The chordal extension is now requested as an optional parameter ``chordal_extension=True`` passed to the ``get_relaxation`` method, and not by specifying it as a hierarchy type in the constructor.
  - Changed: The Moroder hierarchy is now a class.
  - Changed: Small improvements in speed in the substitution routines; unit tests for the substitution routines.
  - Changed: The ``read_sdpa_out`` routine takes an optional argument for a relaxation, and adds the solution to this object if requested.
  - Changed: Instead of an examples folder, all examples were migrated to the documentation.
  - Changed: The symbolic variables which are not to be relaxed are now supplied to the constructor with the optional parameter ``parameters``.
  - Changed: Redundant positive-semidefinite constraint type removed.
  - Fixed: PICOS and MOSEK conversion works for complex matrices too (`issue #10 <https://github.com/peterwittek/ncpol2sdpa/issues/10>`_).
  - Fixed: The moment symmetries are correctly calculated for both Hermitian and non-Hermitian variables (`issue #9 <https://github.com/peterwittek/ncpol2sdpa/issues/9>`_)

**Version 1.9 (2015-08-28)**

  - New: Defining the constraints now also allows using for the symbols ``<``, ``<=``, ``>=``, ``>``. Additionally, the function ``Eq`` from SymPy can be used to defined equalities.
  - New: The function ``solve_sdp`` also accepts ``solver="cvxopt"`` to use CVXOPT for solving a relaxation (requires PICOS and CVXOPT).
  - New: ``convert_to_human_readable`` function returns the objective function and the moment matrix as a string and a matrix of strings to give a symbolic representation of the problem.
  - New: ``get_next_neighbors`` function retrieves the forward neighbors at a given distance of a site or set of sites in a lattice.
  - New: Much faster substitutions if the right-hand side of the substitution never contains variables that are not in the left-hand side.
  - New: Non-unique variables are considered only once in each variable set.
  - New: When using ``solve_sdp`` to solve the relaxation, the solution, its status, and the time it takes to solve are now part of the class ``SdpRelaxation``.
  - New: The class ``Probability`` provides an intuitive way to define quantum probabilities and Bell inequalities.
  - New: The function ``solve_sdp`` autodetects available solvers and complains if there is none.
  - New: The optional parameter ``solverparameters`` to the function ``solve_sdp`` can contain a dictionary of options, with a different set for each of the target solvers.
  - New: Regression testing framework added.
  - Changed: The functions ``find_rank_loop``, ``sos_decomposition``, and ``get_xmat_value`` are no longer required an ``x_mat`` or ``y_mat`` parameter to pass the primal or dual solution. These values are extracted from the solved relaxation. The respective parameters became optional.
  - Changed: Constant term in objective function is added to the primal and dual values when using the ``solve_sdp`` function.
  - Changed: The primal and dual values of the Mosek solution change their signs when using the ``solve_sdp`` function.
  - Changed: The verbosity parameter also controls the console output of every solver.
  - Changed: Faacets relaxations got their own class ``FaacetsRelaxation``.
  - Fixed: Localizing matrices are built correctly when substitution rules contain polynomials and when the identity operator is not part of the monomial sets.
  - Fixed: The function ``get_xmat_value`` also works in Pypy.

**Version 1.8 (2015-05-25)**

  - New: Complex moment matrices are embedded to as real matrices in the SDPA export and the ``solve_sdp`` function.
  - New: Localizing monomials can be fine-tuned by supplying them to ``get_relaxation`` through the optional parameter ``localizing_monomials``.
  - New: ``solve_sdp`` can also solve a problem with Mosek.
  - New: The function ``get_xmat_value`` returns the matching value for a monomial from a solution matrix, given the relaxation and the solution.
  - Changed: ``solve_sdp`` no longer accepts parameters ``solutionmatrix`` and ``solverexecutable``. All parameters are now passed via the solverparameters dictionary.
  - Changed: Legacy Picos code removed. Requirement is now Picos >=1.0.2.
  - Fixed: Determining degree of polynomial also works when coefficient is complex.

**Version 1.7 (2015-03-23)**

  - New: the function ``find_rank_loop`` aids the detection of a rank loop.
  - New: the function ``write_to_human_readable`` writes the relaxation in a human-readable format.
  - New: the function ``read_sdpa_out`` is now exposed to the user, primarily to help in detecting rank loops.
  - New: the function ``save_monomial_index`` allows saving the monomial index of a relaxation.
  - New: support for obtaining the SOS decomposition from a dual solution through the function ``sos_decomposition``.
  - New: optional parameter ``psd=[matrix1, matrix2, ..., matrixn]`` can be passed to ``get_relaxation`` and ``process_constraints`` which contain symbolic matrices that should be positive semidefinite.
  - New: solution matrices can be returned by ``solve_sdp`` by passing the optional
    parameter ``solutionmatrix=True``. It does not work for diagonal blocks.
  - New: basic interface for `Faacets <https://github.com/denisrosset/faacets-core>`_ via the function ``get_faacets_relaxation``.
  - New: PPT constraint can be imposed directly in the Moroder hierarchy by passing the extra parameter ``ppt=True`` to the constructor.
  - New: Passing the optional parameter ``extramomentmatrices=...`` to ``get_relaxation`` allows defining new moment matrices either freely or based on the first one. Basic relations of the elements between the moment matrices can be imposed as strings passed through ``inequalites=...``.
  - Changed: Nieto-Silleras hierarchy is no longer supported through an option. Now constraints have to be manually defined.
  - Changed: Monomials are not saved automatically with ``verbose=2``.
  - Fixed: wider range of substitutions supported, including a polynomial on the right-hands side of the substitution.
  - Fixed: constraints for fermionic and bosonic systems and Pauli operators.

**Version 1.6 (2014-12-22)**

  - Syntax for passing parameters changed. Only the level of the relaxation is compulsory for obtaining a relaxation.
  - Extra parameter for bounds on the variables was added. Syntax is identical to the inequalities. The difference is that the inequalities in the bounds will not be relaxed by localizing matrices.
  - Support for chordal graph extension in the commutative case (doi:`10.1137/050623802 <http://dx.doi.org/10.1137/050623802>`_). Pass ``hierarchy="npa_chordal"`` to the constructor.
  - It is possible to pass variables which will not be relaxed. Pass ``nonrelaxed=[variables]`` to the constructor.
  - It is possible to change the constraints once the moment matrix is generated. Refer to the new function ``process_constraints``.
  - Extra parameter ``nsextraobjvars=[]`` was added for passing additional variables to the Nieto-Silleras hierarchy. This is important because the top-left elements of the blocks of moment matrices in the relaxation are not one: they add up to one. Hence specifying the last element of a measurement becomes possible with this option. The number of elements in this must match the number of behaviours.
  - PICOS conversion routines were separated and reworked to ensure sparsity.
  - Moved documentation to Sphinx.
  - SciPy dependency made optional.

**Version 1.5 (2014-11-27)**

  - Support for Moroder hierarchy (doi:`10.1103/PhysRevLett.111.030501 <http://dx.doi.org/10.1103/PhysRevLett.111.030501>`_).
  - Further symmetries are discovered when all variables are Hermitian.
  - Normalization can be turned off.

**Version 1.4 (2014-11-18)**

  - Pypy support restored with limitations.
  - Direct export to and optimization by MOSEK.
  - Added helper function to add constraints on Pauli operators.
  - Handling of complex coefficients improved.
  - Added PICOS compatibility layer, enabling solving a problem by a larger range of solvers.
  - Bug fixes: Python 3 compatibility restored.

**Version 1.3 (2014-11-03)**

  - Much smaller SDPs are generated when using the helper functions for quantum correlations by not considering the last projector in the measurements and thus removing the sum-to-identity constraint; positive semidefinite condition is not influenced by this.
  - Helper functions for fermionic systems and projective measurements are simplified.
  - Support for the Nieto-Silleras (doi:`10.1088/1367-2630/16/1/013035 <http://dx.doi.org/10.1088/1367-2630/16/1/013035>`_) hierarchy for level 1+ relaxations.

**Version 1.2.4 (2014-06-13)**

  - Bug fixes: mixed commutative and noncommutative variable monomials are handled correctly in substitutions, constant integer objective functions are accepted.

**Version 1.2.3 (2014-06-04)**

  - CHSH inequality added as an example.
  - Allows supplying extra monomials to a given level of relaxation.
  - Added functions to make it easier to work with Bell inequalities.
  - Bug fixes: constant separation works correctly for integers, max-cut example fixed.

**Version 1.2.2 (2014-05-27)**

  - Much faster SDPA writer for problems with many blocks.
  - Removal of equalities does not happen by default.

**Version 1.2.1 (2014-05-22)**

  - Size of localizing matrices adjusts to individual inequalities.
  - Internal structure for storing monomials reorganized.
  - Checks for maximum order in the constraints added.
  - Fermionic constraints corrected.

**Version 1.2 (2014-05-16)**

  - Fast replace was updated and made default.
  - Numpy and SciPy are now dependencies.
  - Replaced internal data structures by SciPy sparse matrices.
  - Pypy is no longer supported.
  - Equality constraints are removed by a QR decomposition and basis transformation.
  - Functions added to support calling SDPA from Python.
  - Helper functions added to help phrasing physics problems.
  - More commutative examples added for comparison to Gloptipoly.
  - Internal module structure reorganized.

**Version 1.1 (2014-05-12)**

  - Commutative variables also work.
  - Major rework of how the moment matrix is generated.

**Version 1.0 (2014-04-29)**

  - Initial release.
