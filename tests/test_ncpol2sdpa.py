import unittest
from test import test_support
import numpy as np
from sympy import S, expand
from sympy.physics.quantum.dagger import Dagger
from ncpol2sdpa import bosonic_constraints, \
                       define_objective_with_I, fermionic_constraints, \
                       flatten, generate_operators, generate_variables, \
                       get_neighbors, maximum_violation, MoroderHierarchy, \
                       Probability, projective_measurement_constraints,  \
                       SdpRelaxation
from ncpol2sdpa.nc_utils import fast_substitute, apply_substitutions
from sympy.core.cache import clear_cache


class ApplySubstitutions(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_apply_substitutions(self):

        def apply_correct_substitutions(monomial, substitutions):
            if isinstance(monomial, int) or isinstance(monomial, float):
                return monomial
            original_monomial = monomial
            changed = True
            while changed:
                for lhs, rhs in substitutions.items():
                    monomial = monomial.subs(lhs, rhs)
                if original_monomial == monomial:
                    changed = False
                original_monomial = monomial
            return monomial

        length, h, U, t = 2, 3.8, -6, 1
        fu = generate_operators('fu', length)
        fd = generate_operators('fd', length)
        _b = flatten([fu, fd])
        hamiltonian = 0
        for j in range(length):
            hamiltonian += U * (Dagger(fu[j])*Dagger(fd[j]) * fd[j]*fu[j])
            hamiltonian += -h/2*(Dagger(fu[j])*fu[j] - Dagger(fd[j])*fd[j])
            for k in get_neighbors(j, len(fu), width=1):
                hamiltonian += -t*Dagger(fu[j])*fu[k]-t*Dagger(fu[k])*fu[j]
                hamiltonian += -t*Dagger(fd[j])*fd[k]-t*Dagger(fd[k])*fd[j]
        substitutions = fermionic_constraints(_b)
        monomials = expand(hamiltonian).as_coeff_mul()[1][0].as_coeff_add()[1]
        substituted_hamiltonian = sum([apply_substitutions(monomial,
                                                           substitutions)
                                       for monomial in monomials])
        correct_hamiltonian = sum([apply_correct_substitutions(monomial,
                                                               substitutions)
                                   for monomial in monomials])
        self.assertTrue(substituted_hamiltonian == expand(correct_hamiltonian))


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

        E = generate_operators('E', 8, hermitian=True)
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
        sdpRelaxation.solve()
        self.assertTrue(abs(sdpRelaxation.primal + 2*np.sqrt(2)) < 10e-5)


class ChshMixedLevel(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_maximum_violation(self):
        I = [[0, -1, 0], [-1, 1, 1], [0, 1, -1]]
        P = Probability([2, 2], [2, 2])
        relaxation = SdpRelaxation(P.get_all_operators())
        relaxation.get_relaxation(1,
                                  objective=define_objective_with_I(I, P),
                                  substitutions=P.substitutions,
                                  extramonomials=P.get_extra_monomials('AB'))
        relaxation.solve()
        self.assertTrue(abs(relaxation.primal + (np.sqrt(2)-1)/2) < 10e-5)


class ElegantBell(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_maximum_violation(self):
        I = [[0, -1.5,  0.5,  0.5,  0.5],
             [0,    1,    1,   -1,   -1],
             [0,    1,   -1,    1,   -1],
             [0,    1,   -1,   -1,    1]]
        violation = maximum_violation([2, 2, 2], [2, 2, 2, 2], I, 1)[0]
        self.assertTrue(abs(violation + np.sqrt(3)) < 10e-5)


class ExampleCommutative(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_solving_with_sdpa(self):
        x = generate_variables('x', 2, commutative=True)
        sdpRelaxation = SdpRelaxation(x)
        sdpRelaxation.get_relaxation(2, objective=x[0]*x[1] + x[1]*x[0],
                                     inequalities=[-x[1]**2 + x[1] + 0.5],
                                     substitutions={x[0]**2: x[0]})
        sdpRelaxation.solve(solver="sdpa")
        self.assertTrue(abs(sdpRelaxation.primal + 0.7320505301965234) < 10e-5)


class ExampleNoncommutative(unittest.TestCase):

    def setUp(self):
        X = generate_operators('x', 2, hermitian=True)
        self.sdpRelaxation = SdpRelaxation(X)
        self.sdpRelaxation.get_relaxation(2, objective=X[0]*X[1] + X[1]*X[0],
                                          inequalities=[-X[1]**2 + X[1] + 0.5],
                                          substitutions={X[0]**2: X[0]})

    def tearDown(self):
        clear_cache()

    def test_solving_with_sdpa(self):
        self.sdpRelaxation.solve(solver="sdpa")
        self.assertTrue(abs(self.sdpRelaxation.primal + 0.75) < 10e-5)

    def test_solving_with_mosek(self):
        self.sdpRelaxation.solve(solver="mosek")
        self.assertTrue(abs(self.sdpRelaxation.primal + 0.75) < 10e-5)

    def test_solving_with_cvxopt(self):
        self.sdpRelaxation.solve(solver="cvxopt")
        self.assertTrue(abs(self.sdpRelaxation.primal + 0.75) < 10e-5)

    def test_solving_with_cvxpy(self):
        self.sdpRelaxation.solve(solver="cvxpy")
        self.assertTrue(abs(self.sdpRelaxation.primal + 0.75) < 10e-5)


class FastSubstitute(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_fast_substitute(self):
        f = generate_operators('f', 2)
        substitutions = {}
        substitutions[Dagger(f[0])*f[0]] = -f[0]*Dagger(f[0])
        monomial = Dagger(f[0])*f[0]
        lhs = Dagger(f[0])*f[0]
        rhs = -f[0]*Dagger(f[0])
        self.assertTrue(fast_substitute(monomial, lhs, rhs) ==
                        monomial.subs(lhs, rhs))
        monomial = Dagger(f[0])*f[0]**2
        self.assertTrue(fast_substitute(monomial, lhs, rhs) ==
                        monomial.subs(lhs, rhs))
        monomial = Dagger(f[0])**2*f[0]
        self.assertTrue(fast_substitute(monomial, lhs, rhs) ==
                        monomial.subs(lhs, rhs))
        monomial = Dagger(f[0])**2*f[0]
        lhs = Dagger(f[0])**2
        rhs = -f[0]*Dagger(f[0])
        self.assertTrue(fast_substitute(monomial, lhs, rhs) ==
                        monomial.subs(lhs, rhs))
        g = generate_operators('g', 2)
        monomial = 2*g[0]**3*g[1]*Dagger(f[0])**2*f[0]
        self.assertTrue(fast_substitute(monomial, lhs, rhs) ==
                        monomial.subs(lhs, rhs))
        monomial = S.One
        self.assertTrue(fast_substitute(monomial, lhs, rhs) ==
                        monomial.subs(lhs, rhs))
        monomial = 5
        self.assertTrue(fast_substitute(monomial, lhs, rhs) ==
                        monomial)
        monomial = 2*g[0]**3*g[1]*Dagger(f[0])**2*f[0] + f[1]
        self.assertTrue(fast_substitute(monomial, lhs, rhs) ==
                        monomial.subs(lhs, rhs))
        monomial = f[1]*Dagger(f[0])**2*f[0]
        lhs = f[1]
        rhs = 1.0 + f[0]
        self.assertTrue(fast_substitute(monomial, lhs, rhs) ==
                        expand(monomial.subs(lhs, rhs)))
        monomial = f[1]**2*Dagger(f[0])**2*f[0]
        result = fast_substitute(fast_substitute(monomial, lhs, rhs), lhs, rhs)
        self.assertTrue(result == expand(monomial.subs(lhs, rhs)))


class Gloptipoly(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_solving(self):
        x = generate_variables('x', 2, commutative=True)
        g0 = 4 * x[0] ** 2 + x[0] * x[1] - 4 * x[1] ** 2 - \
            2.1 * x[0] ** 4 + 4 * x[1] ** 4 + x[0] ** 6 / 3
        sdpRelaxation = SdpRelaxation(x)
        sdpRelaxation.get_relaxation(3, objective=g0)
        sdpRelaxation.solve()
        self.assertTrue(abs(sdpRelaxation.primal + 1.0316282672706911) < 10e-5)


class HarmonicOscillator(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_ground_state_energy(self):
        N = 3
        a = generate_operators('a', N)
        substitutions = bosonic_constraints(a)
        hamiltonian = sum(Dagger(a[i]) * a[i] for i in range(N))
        sdpRelaxation = SdpRelaxation(a, verbose=0)
        sdpRelaxation.get_relaxation(1, objective=hamiltonian,
                                     substitutions=substitutions)
        sdpRelaxation.solve()
        self.assertTrue(abs(sdpRelaxation.primal) < 10e-5)


class Magnetization(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_ground_state(self):
        length, n, h, U, t = 2, 0.8, 3.8, -6, 1
        fu = generate_operators('fu', length)
        fd = generate_operators('fd', length)
        _b = flatten([fu, fd])
        monomials = [[ci for ci in _b]]
        monomials[-1].extend([Dagger(ci) for ci in _b])
        monomials.append([cj*ci for ci in _b for cj in _b])
        monomials.append([Dagger(cj)*ci for ci in _b for cj in _b])
        monomials[-1].extend([cj*Dagger(ci)
                              for ci in _b for cj in _b])
        monomials.append([Dagger(cj)*Dagger(ci)
                          for ci in _b for cj in _b])
        hamiltonian = 0
        for j in range(length):
            hamiltonian += U * (Dagger(fu[j])*Dagger(fd[j]) * fd[j]*fu[j])
            hamiltonian += -h/2*(Dagger(fu[j])*fu[j] - Dagger(fd[j])*fd[j])
            for k in get_neighbors(j, len(fu), width=1):
                hamiltonian += -t*Dagger(fu[j])*fu[k]-t*Dagger(fu[k])*fu[j]
                hamiltonian += -t*Dagger(fd[j])*fd[k]-t*Dagger(fd[k])*fd[j]
        momentequalities = [n-sum(Dagger(br)*br for br in _b)]
        sdpRelaxation = SdpRelaxation(_b, verbose=0)
        sdpRelaxation.get_relaxation(-1,
                                     objective=hamiltonian,
                                     momentequalities=momentequalities,
                                     substitutions=fermionic_constraints(_b),
                                     extramonomials=monomials)
        sdpRelaxation.solve()
        s = 0.5*(sum((Dagger(u)*u) for u in fu) -
                 sum((Dagger(d)*d) for d in fd))
        magnetization = sdpRelaxation[s]
        self.assertTrue(abs(magnetization-0.021325317328560453) < 10e-5)


class MaxCut(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_max_cut(self):
        W = np.diag(np.ones(8), 1) + np.diag(np.ones(7), 2) + \
            np.diag([1, 1], 7) + np.diag([1], 8)
        W = W + W.T
        Q = (np.diag(np.dot(np.ones(len(W)).T, W)) - W) / 4
        x = generate_variables('x', len(W), commutative=True)
        equalities = [xi ** 2 - 1 for xi in x]
        objective = -np.dot(x, np.dot(Q, np.transpose(x)))
        sdpRelaxation = SdpRelaxation(x)
        sdpRelaxation.get_relaxation(1, objective=objective,
                                     equalities=equalities,
                                     removeequalities=True)
        sdpRelaxation.solve()
        self.assertTrue(abs(sdpRelaxation.primal + 13.5) < 10e-5)


class Moroder(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_violation(self):
        I = [[0,   -1,    0],
             [-1,    1,    1],
             [0,    1,   -1]]
        P = Probability([2, 2], [2, 2])
        objective = define_objective_with_I(I, P)
        sdpRelaxation = MoroderHierarchy([flatten(P.parties[0]),
                                          flatten(P.parties[1])],
                                         verbose=0, normalized=False)
        sdpRelaxation.get_relaxation(1, objective=objective,
                                     substitutions=P.substitutions)
        Problem = sdpRelaxation.convert_to_picos(duplicate_moment_matrix=True)
        X = Problem.get_variable('X')
        Y = Problem.get_variable('Y')
        Z = Problem.add_variable('Z', (sdpRelaxation.block_struct[0],
                                 sdpRelaxation.block_struct[0]))
        Problem.add_constraint(Y.partial_transpose() >> 0)
        Problem.add_constraint(Z.partial_transpose() >> 0)
        Problem.add_constraint(X - Y + Z == 0)
        Problem.add_constraint(Z[0, 0] == 1)
        solution = Problem.solve(verbose=0)
        self.assertTrue(abs(solution["obj"] - 0.139) < 10e-3)


class NietoSilleras(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_guessing_probability(self):
        p = [0.5, 0.5, 0.5, 0.5, 0.4267766952966368, 0.4267766952966368,
             0.4267766952966368, 0.07322330470336313]
        P = Probability([2, 2], [2, 2])
        behaviour_constraint = [
          P([0], [0], 'A')-p[0],
          P([0], [1], 'A')-p[1],
          P([0], [0], 'B')-p[2],
          P([0], [1], 'B')-p[3],
          P([0, 0], [0, 0])-p[4],
          P([0, 0], [0, 1])-p[5],
          P([0, 0], [1, 0])-p[6],
          P([0, 0], [1, 1])-p[7]]
        behaviour_constraint.append("-0[0,0]+1.0")
        sdpRelaxation = SdpRelaxation(P.get_all_operators(),
                                      normalized=False, verbose=0)
        sdpRelaxation.get_relaxation(1, objective=-P([0], [0], 'A'),
                                     momentequalities=behaviour_constraint,
                                     substitutions=P.substitutions)
        sdpRelaxation.solve()
        self.assertTrue(abs(sdpRelaxation.primal + 0.5) < 10e-5)


class SparsePop(unittest.TestCase):

    def tearDown(self):
        clear_cache()

    def test_chordal_extension(self):
        X = generate_variables('x', 3, commutative=True)
        inequalities = [1-X[0]**2-X[1]**2, 1-X[1]**2-X[2]**2]
        sdpRelaxation = SdpRelaxation(X)
        sdpRelaxation.get_relaxation(2,
                                     objective=X[1] - 2*X[0]*X[1] + X[1]*X[2],
                                     inequalities=inequalities,
                                     chordal_extension=True)
        sdpRelaxation.solve()
        self.assertTrue(abs(sdpRelaxation.primal + 2.2443690631722637) < 10e-5)


def test_main():
    test_support.run_unittest(ApplySubstitutions, Chsh, ChshMixedLevel,
                              ElegantBell, ExampleCommutative, FastSubstitute,
                              ExampleNoncommutative, Gloptipoly,
                              HarmonicOscillator, MaxCut, Magnetization,
                              Moroder, NietoSilleras, SparsePop)

if __name__ == '__main__':
    test_main()
