******************
Function Reference
******************

SdpRelaxation Class
===================
.. autoclass:: ncpol2sdpa.SdpRelaxation
   :members: get_relaxation, set_objective, process_constraints, solve, __getitem__, write_to_file, save_monomial_index, get_sos_decomposition, find_solution_ranks, convert_to_picos, convert_to_mosek, extract_dual_value

MoroderHierarchy Class
=======================
.. autoclass:: ncpol2sdpa.SteeringHierarchy
   :members: get_relaxation, set_objective, process_constraints, solve, __getitem__, write_to_file, save_monomial_index, convert_to_picos, convert_to_mosek

SteeringHierarchy Class
=======================
.. autoclass:: ncpol2sdpa.SteeringHierarchy
   :members: get_relaxation, set_objective, process_constraints, solve, __getitem__, write_to_file, save_monomial_index, convert_to_picos, convert_to_mosek

FaacetsRelaxation Class
=======================
.. autoclass:: ncpol2sdpa.FaacetsRelaxation
   :members: get_relaxation, solve

Functions to Help Define Polynomial Optimization Problems
=========================================================
.. autofunction:: ncpol2sdpa.generate_operators
.. autofunction:: ncpol2sdpa.generate_variables
.. autofunction:: ncpol2sdpa.get_monomials
.. autofunction:: ncpol2sdpa.flatten

Functions to Study Output of Solver
=================================================
.. autofunction:: ncpol2sdpa.read_sdpa_out

Functions and Classes to Define Physics Problems
================================================
.. autoclass:: ncpol2sdpa.Probability
   :members: __call__, get_all_operators
.. autofunction:: ncpol2sdpa.bosonic_constraints
.. autofunction:: ncpol2sdpa.fermionic_constraints
.. autofunction:: ncpol2sdpa.pauli_constraints
.. autofunction:: ncpol2sdpa.get_neighbors
.. autofunction:: ncpol2sdpa.get_next_neighbors
.. autofunction:: ncpol2sdpa.correlator
.. autofunction:: ncpol2sdpa.generate_measurements
.. autofunction:: ncpol2sdpa.projective_measurement_constraints
.. autofunction:: ncpol2sdpa.maximum_violation
.. autofunction:: ncpol2sdpa.define_objective_with_I
