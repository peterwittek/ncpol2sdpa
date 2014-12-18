******************
Function Reference
******************

SdpRelaxation Class
===================
.. autoclass:: ncpol2sdpa.SdpRelaxation
   :members: get_relaxation, set_objective

Functions to Work with SDPA, PICOS, and MOSEK
=============================================
.. autofunction:: ncpol2sdpa.solve_sdp
.. autofunction:: ncpol2sdpa.write_to_sdpa
.. autofunction:: ncpol2sdpa.convert_to_mosek
.. autofunction:: ncpol2sdpa.convert_to_picos
.. autofunction:: ncpol2sdpa.convert_to_picos_for_export

Functions to Help Define Polynomial Optimization Problems
=========================================================
.. autofunction:: ncpol2sdpa.generate_variables
.. autofunction:: ncpol2sdpa.get_ncmonomials
.. autofunction:: ncpol2sdpa.ncdegree
.. autofunction:: ncpol2sdpa.flatten

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
