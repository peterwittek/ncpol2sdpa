"""
Ncpol2SDPA
=====

Provides
 1. A converter from a polynomial optimization problems of commuting and
    noncommuting variables to a semidefinite programming relaxation.
 2. Helper functions to define physics problems.

"""

__all__ = ['SdpRelaxation',
           'generate_variables',
           'get_ncmonomials',
           'ncdegree',
           'bosonic_constraints',
           'fermionic_constraints',
           'projective_measurement_constraints',
           'correlator',
           'maximum_violation',
           'generate_measurements',
           'define_objective_with_I',
           'flatten',
           'get_neighbors',
           'solve_sdp']

from .sdp_relaxation import SdpRelaxation
from .nc_utils import generate_variables, get_ncmonomials, ncdegree, flatten
from .sdpa_utils import solve_sdp, write_to_sdpa
from .physics_utils import bosonic_constraints, fermionic_constraints, \
    pauli_constraints, get_neighbors, correlator, generate_measurements, \
    projective_measurement_constraints, \
    maximum_violation, define_objective_with_I
