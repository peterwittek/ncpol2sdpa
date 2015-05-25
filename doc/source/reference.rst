******************
Function Reference
******************

SdpRelaxation Class
===================
.. autoclass:: ncpol2sdpa.SdpRelaxation
   :members: get_relaxation, get_faacets_relaxation, set_objective, process_constraints

Functions to Help Define Polynomial Optimization Problems
=========================================================
.. autofunction:: ncpol2sdpa.generate_variables
.. autofunction:: ncpol2sdpa.get_ncmonomials
.. autofunction:: ncpol2sdpa.ncdegree
.. autofunction:: ncpol2sdpa.flatten

Functions to Export, Solve, and Study Relaxations
=================================================
.. autofunction:: ncpol2sdpa.solve_sdp
.. autofunction:: ncpol2sdpa.find_rank_loop
.. autofunction:: ncpol2sdpa.sos_decomposition
.. autofunction:: ncpol2sdpa.get_xmat_value
.. autofunction:: ncpol2sdpa.write_to_sdpa
.. autofunction:: ncpol2sdpa.read_sdpa_out
.. autofunction:: ncpol2sdpa.convert_to_mosek
.. autofunction:: ncpol2sdpa.convert_to_picos
.. autofunction:: ncpol2sdpa.save_monomial_index
.. autofunction:: ncpol2sdpa.write_to_human_readable

Functions to Define Physics Problems
====================================
.. autofunction:: ncpol2sdpa.bosonic_constraints
.. autofunction:: ncpol2sdpa.fermionic_constraints
.. autofunction:: ncpol2sdpa.pauli_constraints
.. autofunction:: ncpol2sdpa.get_neighbors
.. autofunction:: ncpol2sdpa.correlator
.. autofunction:: ncpol2sdpa.generate_measurements
.. autofunction:: ncpol2sdpa.projective_measurement_constraints
.. autofunction:: ncpol2sdpa.maximum_violation
.. autofunction:: ncpol2sdpa.define_objective_with_I
