"""Ncpol2SDPA
=====

Provides
 1. A converter from a polynomial optimization problems of commuting and
    noncommuting variables to a semidefinite programming relaxation.
 2. Helper functions to define physics problems.

"""
from .faacets_relaxation import FaacetsRelaxation
from .sdp_relaxation import SdpRelaxation
from .steering_hierarchy import SteeringHierarchy
from .moroder_hierarchy import MoroderHierarchy
from .chordal_extension import find_variable_cliques
from .nc_utils import generate_variable, generate_variables, get_ncmonomials, \
                      ncdegree, flatten, save_monomial_index
from .sdpa_utils import write_to_sdpa, read_sdpa_out, \
                        convert_to_human_readable, write_to_human_readable
from .solver_common import solve_sdp, find_rank_loop, sos_decomposition, \
                           get_xmat_value
from .physics_utils import bosonic_constraints, fermionic_constraints, \
    pauli_constraints, get_neighbors, get_next_neighbors, correlator, \
    generate_measurements, projective_measurement_constraints, \
    maximum_violation, define_objective_with_I, Probability
from .mosek_utils import convert_to_mosek
from .picos_utils import convert_to_picos

__all__ = ['SdpRelaxation',
           'SteeringHierarchy',
           'MoroderHierarchy',
           'FaacetsRelaxation',
           'generate_variable',
           'generate_variables',
           'get_ncmonomials',
           'find_variable_cliques',
           'ncdegree',
           'save_monomial_index',
           'bosonic_constraints',
           'fermionic_constraints',
           'projective_measurement_constraints',
           'correlator',
           'maximum_violation',
           'generate_measurements',
           'define_objective_with_I',
           'Probability',
           'flatten',
           'get_neighbors',
           'get_next_neighbors',
           'solve_sdp',
           'get_xmat_value',
           'write_to_sdpa',
           'find_rank_loop',
           'read_sdpa_out',
           'pauli_constraints',
           'convert_to_human_readable',
           'convert_to_mosek',
           'convert_to_picos',
           'write_to_human_readable',
           'sos_decomposition']
