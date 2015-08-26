import unittest
from test import test_support
from ncpol2sdpa import SdpRelaxation, solve_sdp, generate_variables
from sympy.core.cache import clear_cache

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
        self.failUnless(abs(self.sdpRelaxation.primal+0.75)<10e-5)

    def test_solving_with_mosek(self):
        solve_sdp(self.sdpRelaxation, solver="mosek")
        self.failUnless(abs(self.sdpRelaxation.primal+0.75)<10e-5)

    def test_solving_with_cvxopt(self):
        solve_sdp(self.sdpRelaxation, solver="cvxopt")
        self.failUnless(abs(self.sdpRelaxation.primal+0.75)<10e-5)

class ExampleCommutative(unittest.TestCase):

    def setUp(self):
        x = generate_variables(2, commutative=True)
        self.sdpRelaxation = SdpRelaxation(x)
        self.sdpRelaxation.get_relaxation(2, objective=x[0]*x[1] + x[1]*x[0],
                                          inequalities=[-x[1]**2 + x[1] + 0.5],
                                          substitutions={x[0]**2:x[0]})

    def tearDown(self):
        clear_cache()

    def test_solving_with_sdpa(self):
        solve_sdp(self.sdpRelaxation, solver="sdpa")
        self.failUnless(abs(self.sdpRelaxation.primal+0.7320505301965234)<10e-5)


def test_main():
    test_support.run_unittest(ExampleNoncommutative,
                              ExampleCommutative)

if __name__ == '__main__':
    test_main()
