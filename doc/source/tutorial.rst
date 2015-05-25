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
noncommuting variables:

::

    X = generate_variables(2, 'X')
    sdpRelaxation = SdpRelaxation(X)

Getting the relaxation requires at least the level of relaxation:

::

    sdpRelaxation.get_relaxation(level)

This will generate the moment matrix. Additional elements of the
problem, such as the objective function, inequalities, equalities, and
bounds on the variables.

The last step in is to write out the relaxation to a sparse SDPA file.
The method (``write_to_sdpa``) takes one parameter, the file name.
Alternatively, if SDPA is in the search path, then it can be solved by
invoking a helper function (``solve_sdp``). Alternatively, MOSEK is
also supported for obtaining a solution by passing the parameter 
``solver='mosek'`` to this function. Using a converter to PICOS, 
it is also possible to solve the problem with a range of other solvers, 
including CVXOPT.


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
:math:`\{1, x_1, x_2, x_1x_2, x_2x_1, x_2^2\}`. The corresponding level-2 
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
    x = generate_variables(n_vars, commutative=True, name='x')

Above we must declare the variables as commutative. By default, the generated
variables are noncommutative and non-Hermitian. With these variables, we can 
define the objective and the inequality constraint. Notice that all 
inequality-type constraints are assumed to be in the form :math:`\ge 0`.

::

    obj = x[0]*x[1] + x[1]*x[0]
    inequalities = [-x[1]**2 + x[1] + 0.5]

The equality, as discussed, is entered as a substitution rule:

::

    substitutions = {x[0]**2 : x[0]}


Generating and Solving the Relaxation
=====================================
After we defined the problem, we need to initialize the SDP relaxation object 
with the variables, and request generating the relaxation given the constraints:

::

    sdpRelaxation = SdpRelaxation(x)
    sdpRelaxation.get_relaxation(level, objective=obj, inequalities=inequalities,
                                 substitutions=substitutions)
  
For large problems, getting the relaxation can take a long time. Once we have 
the relaxation, we can try to solve it solve it. Currently two solvers are 
supported fully: SDPA and MOSEK. SDPA is the default and it has to be in the 
path. If it is, we obtain the solution by calling the ``solve_sdp`` function:

::

    primal, dual, x_mat, y_mat = solve_sdp(sdpRelaxation)
    print(primal, dual)

This gives a solution close to the optimum around -0.7321. If the solver is not
in the path, or you want more control over the parameters of the solver, or you
want to solve the problem in MATLAB, you export the relaxation to SDPA format:
  
::

    write_to_sdpa(sdpRelaxation, 'example.dat-s')

Alternatively, if you have MOSEK installed and it is callable from your Python
distribution, you can request to use it:

    primal, dual, x_mat, y_mat = solve_sdp(sdpRelaxation, solver='mosek')
    print(primal, dual)


Analyzing the Solution
======================
We can study individual entries of the solution matrix by providing the monomial
we are interested in. For example:

::
  
    get_xmat_value(X[0]*X[1], sdpRelaxation, x_mat)

The sums-of-square (SOS) decomposition is extracted from the dual solution:

::

    sos_decomposition(sdpRelaxation, y_mat, threshold=0.001)

If we solve the SDP with the arbitrary-precision solver ``sdpa_gmp``, 
we can find a rank loop at level two, indicating that convergence has 
been achieved. Assuming that you exported the file and solved the SDP outside
Python, we read the solution file and analyse the ranks:

::

    primal, dual, x_mat, y_mat = read_sdpa_out("example.out", True)
    find_rank_loop(sdpRelaxation, x_mat[0])

The output for this problem is ``[2, 3]``, not showing a rank loop at this level
of relaxation.


Debugging the SDP Relaxation
============================
It often happens that solving a relaxation does not yield the expected results.
To help understand what goes wrong, Ncpol2sdpa provides a function to write the 
relaxation in a comma separated file, in which the individual cells contain the 
respective monomials. The first line of the file is the objective function.

::

    write_to_human_readable(sdpRelaxation, "examples.csv")
    
Furthermore, the library can write out which SDP variable corresponds to which 
monomial by calling

::

    save_monomial_index("monomials.txt", sdpRelaxation.monomial_index)

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
pattern:

::

    X = generate_variables(n_vars, hermitian=True, name='X')
    obj_nc = X[0] * X[1] + X[1] * X[0]
    inequalities_nc = [-X[1] ** 2 + X[1] + 0.5]
    substitutions_nc = {X[0]**2 : X[0]}
    sdpRelaxation_nc = SdpRelaxation(X)
    sdpRelaxation_nc.get_relaxation(level, objective=obj_nc, 
                                    inequalities=inequalities_nc,
                                    substitutions=substitutions_nc)
    primal_nc, dual_nc, x_mat_nc, y_mat_nc = solve_sdp(sdpRelaxation_nc)


This gives a solution very close to the analytical -3/4. Let us export the
problem again:

::
    
    write_to_sdpa(sdpRelaxation, 'examplenc.dat-s')
    
Solving this with the arbitrary-precision solver, we discover a rank loop:

::

    primal_nc, dual_nc, x_mat_nc, y_mat_nc = read_sdpa_out("data/examplenc.out", True)
    find_rank_loop(sdpRelaxation_nc, x_mat_nc[0])

The output is ``[2, 2]``, indicating a rank loop and showing that the 
noncommutative case of the relaxation converges faster.
