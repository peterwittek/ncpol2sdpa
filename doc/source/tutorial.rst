********
Tutorial
********

The implementation follows an object-oriented design. The core object is
SdpRelaxation. There are three steps to generate the relaxation:

* Instantiate the SdpRelaxation object.

* Get the relaxation.

* Write the relaxation to a file or solve the problem.

The second step is the most time consuming, often running for hours as
the number of variables increases. Once the solution is obtained, it can
be studied further with some helper functions.

To instantiate the SdpRelaxation object, you need to specify the
variables. You can use any SymPy symbolic variable, as long as the adjoint
operator is well-defined. The library also has helper functions to generate
commutative or noncommutative variables or operators.

Getting the relaxation requires at least the level of relaxation, and the
matching method, `SdpRelaxation.get_relaxation`, will generate the moment
matrix. Additional elements of the problem, such as the objective function,
inequalities, equalities, and constraints on the moments.

The last step in is to either solve or export the relaxation. The function
`solve_sdp` or the class method `SdpRelaxation.solve` autodetects the possible
solvers: SDPA, MOSEK, and CVXOPT. Alternatively, the method ``write_to_file``
exports the file to sparse SDPA format, which can be solved externally on a
supercomputer, in MATLAB, or by any other means that accepts this input format.


Defining a Polynomial Optimization Problem of Commuting Variables
=================================================================

Consider the following polynomial optimization problem:

.. math:: \min_{x\in \mathbb{R}^2}2x_1x_2

such that

.. math:: -x_2^2+x_2+0.5\geq 0

.. math:: x_1^2-x_1=0.

The equality constraint is a simple projection. We either substitute it with two
inequalities or treat the equality as a monomial substitution. The second option
leads to a sparser SDP relaxation. The code samples below take this approach.
In this case, the monomial basis is
:math:`\{1, x_1, x_2, x_1x_2, x_2^2\}`. The corresponding level-2
relaxation is written as

.. math:: \min_{y}2y_{12}

such that

.. math::

   \left[ \begin{array}{c|cc|cc}1 & y_{1} & y_{2} & y_{12} & y_{22}\\
   \hline{}y_{1} & y_{1} & y_{12} & y_{12} & y_{122}\\
   y_{2} & y_{12} & y_{22} & y_{122} & y_{222}\\
   \hline{}y_{12} & y_{12} & y_{122} & y_{122} & y_{1222}\\
   y_{22} & y_{122} & y_{222} & y_{1222} & y_{2222}\end{array} \right] \succeq{}0

.. math::

   \left[ \begin{array}{c|cc}-y_{22}+y_{2}+0.5 & -y_{122}+y_{12}+0.5y_{1} & -y_{222}+y_{22}+0.5y_{2}\\
   \hline{}-y_{122}+y_{12}+0.5y_{1} & -y_{122}+y_{12}+0.5y_{1} & -y_{1222}+y_{122}+0.5y_{12}\\
   -y_{222}+y_{22}+0.5y_{2} & -y_{1222}+y_{122}+0.5y_{12} & -y_{2222}+y_{222}+0.5y_{22}
   \end{array}\right]\succeq{}0.

Apart from the matrices being symmetric, notice other regular patterns
between the elements -- these are recognized in the relaxation and the same SDP
variables are used for matching moments. To generate the relaxation, first we
set up a few helper variables, including the symbolic variables used to define
the polynomial objective function and constraint. The symbolic manipulations
are based on SymPy.

::

    from ncpol2sdpa import *

    n_vars = 2 # Number of variables
    level = 2  # Requested level of relaxation
    x = generate_variables('x', n_vars)

By default, the generated variables are commutative. Alternatively, you can use
standard SymPy symbols, but it is worth declaring them as real. With these
variables, we can define the objective and the inequality constraint.

::

    obj = x[0]*x[1] + x[1]*x[0]
    inequalities = [-x[1]**2 + x[1] + 0.5>=0]

We can also write all inequality-type constraints assuming to be in the form :math:`\ge 0` as

::

    inequalities = [-x[1]**2 + x[1] + 0.5]

This is more convenient when we have a large number of constraints.

The equality, as discussed, is entered as a substitution rule:

::

    substitutions = {x[0]**2 : x[0]}


Generating and Solving the Relaxation
=====================================
After we defined the problem, we need to initialize the SDP relaxation object
with the variables, and request generating the relaxation given the constraints:

::

    sdp = SdpRelaxation(x)
    sdp.get_relaxation(level, objective=obj, inequalities=inequalities,
                       substitutions=substitutions)

For large problems, getting the relaxation can take a long time. Once we have
the relaxation, we can try to solve it solve it. Currently three solvers are
supported fully: SDPA, MOSEK, and CVXOPT. If any of them are available, we
obtain the solution by calling the ``solve`` method:

::

    sdp.solve()
    print(sdp.primal, sdp.dual, sdp.status)

This gives a solution close to the optimum around -0.7321. The solution and some
status information and the time it takes to solve it become part of the
relaxation object.

If no solver is detected, or you want more control over the parameters
of the solver, or you want to solve the problem in MATLAB, you export the
relaxation to SDPA format:

::

    sdp.write_to_file('example.dat-s')

You can also specify a solver if you wish. For instance, if you want to use
the arbitrary-precision solver that you have available in the path, along with a
matching parameter file, you can call

::

    sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp",
                                               "paramsfile"="params.gmp.sdpa"})

If you have multiple solvers available, you might want to specify which exactly
you want to use. For CVXOPT, call

::

    sdp.solve(solver='cvxopt')
    print(sdp.primal, sdp.dual)

This solution also requires PICOS on top of CXOPT. Alternatively, if you have
MOSEK installed and it is callable from your Python distribution, you can
request to use it:

    sdp.solve(solver='mosek')
    print(sdp.primal, sdp.dual)


Analyzing the Solution
======================
We can study individual entries of the solution matrix by providing the monomial
we are interested in. For example:

::

    sdp[X[0]*X[1]]

The sums-of-square (SOS) decomposition is extracted from the dual solution:

::

    sigma = sdp.get_sos_decomposition()

If we solve the SDP with the arbitrary-precision solver ``sdpa_gmp``,
we can find a rank loop at level two, indicating that convergence has
been achieved.

::

    sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp",
                                               "paramsfile"="params.gmp.sdpa"})
    sdp.find_solution_ranks()

The output for this problem is ``[2, 3]``, not showing a rank loop at this level
of relaxation.


Debugging the SDP Relaxation
============================
It often happens that solving a relaxation does not yield the expected results.
To help understand what goes wrong, Ncpol2sdpa provides a function to write the
relaxation in a comma separated file, in which the individual cells contain the
respective monomials. The first line of the file is the objective function.

::

    sdp.write_to_file("examples.csv")

Furthermore, the library can write out which SDP variable corresponds to which
monomial by calling

::

    sdp.save_monomial_index("monomials.txt")

Defining and Solving an Optimization Problem of Noncommuting Variables
======================================================================
Consider a slight variation of the problem discussed in the previous sections:
change the algebra of the variables from commutative to Hermitian noncommutative, and use
the following objective function:

.. math:: \min_{x\in \mathbb{R}^2}x_1x_2+x_2x_1

The constraints remain identical:

.. math:: -x_2^2+x_2+0.5\geq 0

.. math:: x_1^2-x_1=0.

Defining the problem, generating the relaxation, and solving it follow a similar
pattern, but we request operators instead of variables.

::

    X = generate_operators('X', n_vars, hermitian=True)
    obj_nc = X[0] * X[1] + X[1] * X[0]
    inequalities_nc = [-X[1] ** 2 + X[1] + 0.5]
    substitutions_nc = {X[0]**2 : X[0]}
    sdp_nc = SdpRelaxation(X)
    sdp_nc.get_relaxation(level, objective=obj_nc, inequalities=inequalities_nc,
                          substitutions=substitutions_nc)
    sdp_nc.solve()


This gives a solution very close to the analytical -3/4. Let us export the
problem again:

::

    sdp.write_to_file("examplenc.dat-s")

Solving this with the arbitrary-precision solver, we discover a rank loop:

::

    sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp",
                                               "paramsfile"="params.gmp.sdpa"})
    sdp.find_solution_ranks()

The output is ``[2, 2]``, indicating a rank loop and showing that the
noncommutative case of the relaxation converges faster.
