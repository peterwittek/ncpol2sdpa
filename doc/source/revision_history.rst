Revision History
****************

**Since version 1.5**

  - Syntax for passing parameters changed. Only the level of the relaxation is
    compulsory for obtaining a relaxation.
  - Extra parameter for bounds on the variables was added. Syntax is identical 
    to the inequalities. The difference is that the inequalities in the bounds 
    will not be relaxed by localizing matrices.
  - Extra parameter for passing additional variables to the Nieto-Silleras
    hierarchy. This is important because the top-left element of the blocks of 
    moment matrices in the relaxation is not one: they add up to one. Hence
    specifying the last element of a measurement becomes possible with this 
    option.
  - Support for chordal graph extension in the commutative case
    (doi:`10.1137/050623802 <http://dx.doi.org/10.1137/050623802>`_). Pass ``hierarchy="npa_chordal"`` to the constructor.
  - PICOS conversion routines were separated.
  - Moved documentation to Sphinx.

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
