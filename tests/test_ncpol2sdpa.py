import unittest
from test import test_support
import numpy as np
from sympy.physics.quantum.dagger import Dagger
from ncpol2sdpa import SdpRelaxation, solve_sdp, generate_variables, flatten, \
                       projective_measurement_constraints, Probability, \
                       define_objective_with_I, maximum_violation, \
                       bosonic_constraints, convert_to_picos_extra_moment_matrix
from sympy.core.cache import clear_cache

class Chsh(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_maximum_violation(self):
      
        def expectation_values(measurement, outcomes):
            exp_values = []
            for k in range(len(measurement)):
                exp_value = 0
                for j in range(len(measurement[k])):
                    exp_value += outcomes[k][j] * measurement[k][j]
                exp_values.append(exp_value)
            return exp_values

        E = generate_variables(8, name='E', hermitian=True)
        M, outcomes = [], []
        for i in range(4):
            M.append([E[2 * i], E[2 * i + 1]])
            outcomes.append([1, -1])
        A = [M[0], M[1]]
        B = [M[2], M[3]]
        substitutions = projective_measurement_constraints(A, B)
        C = expectation_values(M, outcomes)
        chsh = -(C[0] * C[2] + C[0] * C[3] + C[1] * C[2] - C[1] * C[3])
        sdpRelaxation = SdpRelaxation(E, verbose=0)
        sdpRelaxation.get_relaxation(1, objective=chsh,
                                     substitutions=substitutions)
        solve_sdp(sdpRelaxation)
        self.assertTrue(abs(sdpRelaxation.primal+2*np.sqrt(2))<10e-5)

class ChshMixedLevel(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_maximum_violation(self):
        I = [[ 0,   -1,    0 ],
             [-1,    1,    1 ],
             [ 0,    1,   -1 ]]
        P = Probability([2, 2], [2, 2])
        sdpRelaxation = SdpRelaxation(P.get_all_operators())
        sdpRelaxation.get_relaxation(1, objective=define_objective_with_I(I, P),
                                     substitutions=P.substitutions,
                                     extramonomials=P.get_extra_monomials('AB'))
        solve_sdp(sdpRelaxation)
        self.assertTrue(abs(sdpRelaxation.primal+(np.sqrt(2)-1)/2)<10e-5)

class ElegantBell(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_maximum_violation(self):
        I = [[0, -1.5,  0.5,  0.5,  0.5],
             [0,    1,    1,   -1,   -1],
             [0,    1,   -1,    1,   -1],
             [0,    1,   -1,   -1,    1]]
        self.assertTrue(abs(maximum_violation([2, 2, 2], [2, 2, 2, 2], I, 2)[0]
                            +np.sqrt(3))<10e-5)


class ExampleCommutative(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_solving_with_sdpa(self):
        x = generate_variables(2, commutative=True)
        sdpRelaxation = SdpRelaxation(x)
        sdpRelaxation.get_relaxation(2, objective=x[0]*x[1] + x[1]*x[0],
                                          inequalities=[-x[1]**2 + x[1] + 0.5],
                                          substitutions={x[0]**2:x[0]})
        solve_sdp(sdpRelaxation, solver="sdpa")
        self.assertTrue(abs(sdpRelaxation.primal+0.7320505301965234)<10e-5)

class ExampleNoncommutative(unittest.TestCase):

    def setUp(self):
        X = generate_variables(2, hermitian=True)
        self.sdpRelaxation = SdpRelaxation(X)
        self.sdpRelaxation.get_relaxation(2, objective=X[0]*X[1] + X[1]*X[0],
                                          inequalities=[-X[1]**2 + X[1] + 0.5],
                                          substitutions={X[0]**2:X[0]})

    def tearDown(self):
        clear_cache()

    def test_solving_with_sdpa(self):
        solve_sdp(self.sdpRelaxation, solver="sdpa")
        self.assertTrue(abs(self.sdpRelaxation.primal+0.75)<10e-5)

    def test_solving_with_mosek(self):
        solve_sdp(self.sdpRelaxation, solver="mosek")
        self.assertTrue(abs(self.sdpRelaxation.primal+0.75)<10e-5)

    def test_solving_with_cvxopt(self):
        solve_sdp(self.sdpRelaxation, solver="cvxopt")
        self.assertTrue(abs(self.sdpRelaxation.primal+0.75)<10e-5)

class Gloptipoly(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_solving(self):
        x = generate_variables(2, commutative=True)
        g0 = 4 * x[0] ** 2 + x[0] * x[1] - 4 * x[1] ** 2 - \
            2.1 * x[0] ** 4 + 4 * x[1] ** 4 + x[0] ** 6 / 3
        sdpRelaxation = SdpRelaxation(x)
        sdpRelaxation.get_relaxation(3, objective=g0)
        solve_sdp(sdpRelaxation)
        self.assertTrue(abs(sdpRelaxation.primal+1.0316282672706911)<10e-5)

class HarmonicOscillator(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_ground_state_energy(self):
        N = 3
        a = generate_variables(N, name='a')
        substitutions = bosonic_constraints(a)
        hamiltonian = sum(Dagger(a[i]) * a[i] for i in range(N))
        sdpRelaxation = SdpRelaxation(a, verbose=0)
        sdpRelaxation.get_relaxation(1, objective=hamiltonian,
                                     substitutions=substitutions)
        solve_sdp(sdpRelaxation)
        self.assertTrue(abs(sdpRelaxation.primal)<10e-5)

class MaxCut(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_max_cut(self):
        W = np.diag(np.ones(8), 1) + np.diag(np.ones(7), 2) + np.diag([1, 1], 7) + \
            np.diag([1], 8)
        W = W + W.T
        Q = (np.diag(np.dot(np.ones(len(W)).T, W)) - W) / 4
        x = generate_variables(len(W), commutative=True)
        equalities = [xi ** 2 - 1 for xi in x]
        objective = -np.dot(x, np.dot(Q, np.transpose(x)))
        sdpRelaxation = SdpRelaxation(x)
        sdpRelaxation.get_relaxation(1, objective=objective, 
                                     equalities=equalities,
                                     removeequalities=True)
        solve_sdp(sdpRelaxation)
        self.assertTrue(abs(sdpRelaxation.primal+4.5)<10e-5)

class Moroder(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_violation(self):
        I = [[ 0,   -1,    0 ],
             [-1,    1,    1 ], 
             [ 0,    1,   -1 ]]
        P = Probability([2, 2], [2, 2])
        objective = define_objective_with_I(I, P)
        sdpRelaxation = SdpRelaxation([flatten(P.parties[0]), flatten(P.parties[1])], 
                                       verbose=0, hierarchy="moroder", normalized=False)
        sdpRelaxation.get_relaxation(1, objective=objective,
                                     substitutions=P.substitutions)
        Problem, X, Y = convert_to_picos_extra_moment_matrix(sdpRelaxation)
        Z = Problem.add_variable('Z', (sdpRelaxation.block_struct[0],
                                 sdpRelaxation.block_struct[0]))
        Problem.add_constraint(Y.partial_transpose()>>0)
        Problem.add_constraint(Z.partial_transpose()>>0)
        Problem.add_constraint(X - Y + Z == 0)
        Problem.add_constraint(Z[0,0] == 1)
        solution = Problem.solve(verbose=0)
        self.assertTrue(abs(solution["obj"]-0.728)<10e-3)

class NietoSilleras(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_guessing_probability(self):
        p = [0.5, 0.5, 0.5, 0.5, 0.4267766952966368, 0.4267766952966368, 
             0.4267766952966368, 0.07322330470336313]
        P = Probability([2, 2], [2, 2])
        bounds = [
          P([0],[0],'A')-p[0],
          P([0],[1],'A')-p[1],
          P([0],[0],'B')-p[2],
          P([0],[1],'B')-p[3],
          P([0,0],[0,0])-p[4],
          P([0,0],[0,1])-p[5],
          P([0,0],[1,0])-p[6],
          P([0,0],[1,1])-p[7]]
        bounds.extend([-bound for bound in bounds])
        bounds.append("-0[0,0]+1.0")
        bounds.append("0[0,0]-1.0")        
        sdpRelaxation = SdpRelaxation(P.get_all_operators(), 
                                      normalized=False, verbose=0)
        sdpRelaxation.get_relaxation(1, objective=-P([0],[0],'A'), 
                                     bounds=bounds,
                                     substitutions=P.substitutions)
        solve_sdp(sdpRelaxation)
        self.assertTrue(abs(sdpRelaxation.primal+0.5)<10e-5)

class SparsePop(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_chordal_extension(self):
        X = generate_variables(3, commutative=True)
        inequalities = [1-X[0]**2-X[1]**2, 1-X[1]**2-X[2]**2]
        sdpRelaxation = SdpRelaxation(X, hierarchy="npa_chordal")
        sdpRelaxation.get_relaxation(2, objective=X[1] - 2*X[0]*X[1] + X[1]*X[2],
                                     inequalities=inequalities)
        solve_sdp(sdpRelaxation)
        self.assertTrue(abs(sdpRelaxation.primal+2.2443690631722637)<10e-5)

def test_main():
    test_support.run_unittest(Chsh, ChshMixedLevel, ElegantBell, 
                              ExampleCommutative, ExampleNoncommutative, 
                              Gloptipoly, HarmonicOscillator, MaxCut, Moroder,
                              NietoSilleras, SparsePop)

if __name__ == '__main__':
    test_main()
