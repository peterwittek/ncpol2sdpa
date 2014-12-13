********
Examples
********
The implementation follows an object-oriented design. The core object is
SdpRelaxation. There are three steps to generate the relaxation:

* Instantiate the SdpRelaxation object.

* Get the relaxation.

* Write the relaxation to a file or solve the problem.

The second step is the most time consuming, often running for hours as
the number of variables increases.

To instantiate the SdpRelaxation object, you need to specify the
noncommuting variables:

::

    X = ... # Define noncommuting variables
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
also supported for writing a problem and solving it. Using a converter
to PICOS, it is also possible to solve the problem with a range of
other solvers, including CVXOPT.


Example 1: Toy Example
==================================================

Consider the following polynomial optimization problem (Pironio,
Navascués, and Acín 2010):

.. math:: \min_{x\in \mathbb{R}^2}x_1x_2+x_2x_1

such that

.. math:: -x_2^2+x_2+0.5\geq 0

.. math:: x_1^2-x_1=0.

Entering the objective function and the inequality constraint is easy.
The equality constraint is a simple projection. We either substitute two
inequalities to replace the equality, or treat the equality as a
monomial substitution. The second option leads to a sparser SDP
relaxation. The code samples below take this approach. In this case, the
monomial basis is :math:`\{1, x_1, x_2, x_1x_2, x_2x_1, x_2^2\}`. The
corresponding relaxation is written as

.. math:: \min_{y}y_{12}+y_{21}

such that

.. math::

   \left[\begin{array}{c|cc|ccc}
   1 & y_{1} & y_{2} & y_{12} & y_{21} & y_{22}\\
   \hline{}
   y_{1} & y_{1} & y_{12} & y_{12} & y_{121} & y_{122}\\
   y_{2} & y_{21} & y_{22} & y_{212} & y_{221} & y_{222}\\
   \hline{}
   y_{21} & y_{21} & y_{212} & y_{212} & y_{2121} & y_{2122} \\
   y_{12} & y_{121} & y_{122} & y_{1212} & y_{1221} & y_{1222}\\
   y_{22} & y_{221} & y_{222} & y_{2212} & y_{2221} & y_{2222}
   \end{array} \right] \succeq{}0

.. math::

   \left[ \begin{array}{c|cc}
   -y_{22}+y_{2}+0.5 & -y_{221}+y_{21}+0.5y_{1} & -y_{222}+y_{22}+0.5y_{2}\\
   \hline{}
   -y_{221}+y_{21}+0.5y_{1} & -y_{1221}+y_{121}+0.5y_{1} & -y_{1222}+y_{122}+0.5y_{12}\\
   -y_{222}+y_{22}+0.5y_{2} & -y_{1222}+y_{122}+0.5y_{12} & -y_{2222}+y_{222}+0.5y_{22}
   \end{array}\right]\succeq{}0.

Apart from the matrices being symmetric, notice other regular patterns
between the elements. These are taken care of as additional constraints
in the implementation. The optimum for the objective function is
:math:`-3/4`. The implementation reads as follows:

::

    from ncpol2sdpa import *

    # Number of Hermitian variables
    n_vars = 2
    # Level of relaxation
    level = 2

    # Get Hermitian variables
    X = generate_variables(n_vars, hermitian=True)

    # Define the objective function
    obj = X[0] * X[1] + X[1] * X[0]

    # Inequality constraints
    inequalities = [-X[1] ** 2 + X[1] + 0.5]

    # Simple monomial substitutions
    monomial_substitution = {}
    monomial_substitution[X[0] ** 2] = X[0]

    # Obtain SDP relaxation
    sdpRelaxation = SdpRelaxation(X)
    sdpRelaxation.get_relaxation(level, objective=obj, inequalities=inequalities,
                                 substitutions=monomial_substitution)
    write_to_sdpa(sdpRelaxation, 'examplenc.dat-s')

Any flavour of the SDPA family of solvers will solve the exported
problem:

::

    $ sdpa examplenc.dat-s examplenc.out

If the SDPA solver is in the search path, we can invoke the solver from
Python:

::

    primal, dual = solve_sdp(sdpRelaxation)

The relevant part of the output shows the optimum for the objective
function:

::

    objValPrimal = -7.5000001721851994e-01
    objValDual   = -7.5000007373829902e-01

This is close to the analytical optimum of :math:`-3/4`.

Example 2: Using MOSEK or PICOS
==================================================

Apart from SDPA, MOSEK also enjoys full support. Using the preliminaries
of the problem outlined in Section [example1], once we have the
relaxation, we can convert it to a MOSEK task and solve it:

::

    task = convert_to_mosek(sdpRelaxation)
    task.optimize()
    task.solutionsummary(mosek.streamtype.msg)

Please ensure that the MOSEK installation is operational.

A compatibility layer with PICOS allows calling a wider ranger of
solvers. Assuming that the PICOS dependencies are in ``PYTHONPATH``, we
can pass an argument to the function ``get_relaxation`` to generate a
PICOS optimization problem. Using the same example as before, we change
the relevant function call to:

::

    P = convert_to_picos(sdpRelaxation)

This returns a PICOS problem, and with that, we can solve it with any
solver that PICOS supports:

::

    P.solve()

Example 3: Mixed-Level Relaxation of a Bell Inequality
======================================================

It is often the case that moving to a higher-order relaxation is
computationally prohibitive. For these cases, it is possible to inject
extra monomials to a lower level relaxation. We refer to this case as a
mixed-level relaxation.

As an example, we consider the CHSH inequality in the probability
picture at level 1+AB relaxation.

::

    level = 1
    A_configuration = [2, 2]
    B_configuration = [2, 2]
    I = [[ 0,   -1,    0 ],
         [-1,    1,    1 ], 
         [ 0,    1,   -1 ]]
    A = generate_measurements(A_configuration, 'A')
    B = generate_measurements(B_configuration, 'B')
    monomial_substitutions = projective_measurement_constraints(A, B)
    objective = define_objective_with_I(I, A, B)

Then we need to generate the monomials we would like to add to the
relaxation.

::

    AB = [Ai*Bj for Ai in flatten(A) for Bj in flatten(B)]  

We have to tell when we ask for the relaxation that these extra
monomials should be considered:

::

    sdpRelaxation = SdpRelaxation(flatten([A, B]))
    sdpRelaxation.get_relaxation(level, objective=objective,
                                 substitutions=monomial_substitutions,
                                 extramonomials=AB)

Example 4: Bosonic System
==================================================

The system Hamiltonian describes :math:`N` harmonic oscillators with a
parameter :math:`\omega`. It is the result of second quantization and it
is subject to bosonic constraints on the ladder operators :math:`a_{k}`
and :math:`a_{k}^{\dagger}` (see, for instance, Section 22.2 in M.
Fayngold and Fayngold (2013)). The Hamiltonian is written as

.. math:: H = \hbar \omega\sum_{i}\left(a_{i}^{\dagger}a_{i}+\frac{1}{2}\right).

Here :math:`^{\dagger}` stands for the adjoint operation. The
constraints on the ladder operators are given as

.. math::

   \begin{aligned}
   [a_{i},a_{j}^{\dagger}] &=  \delta_{ij} \\
   [a_{i},a_{j}]  &=  0 \nonumber \\
   [a_{i}^{\dagger},a_{j}^{\dagger}] &=  0,\nonumber\end{aligned}

where :math:`[.,.]` stands for the commutation operator
:math:`[a,b]=ab-ba`.

Clearly, most of the constraints are monomial substitutions, except
:math:`[a_{i},a_{i}^{\dagger}]=1`, which needs to be defined as an
equality. The Python code for generating the SDP relaxation is provided
below. We set :math:`\omega=1`, and we also set Planck’s constant
:math:`\hbar` to one, to obtain numerical results that are easier to
interpret.

::

    from sympy.physics.quantum.dagger import Dagger

    # level of relaxation
    level = 1

    # Number of variables
    N = 4

    # Parameters for the Hamiltonian
    hbar, omega = 1, 1

    # Define ladder operators
    a = generate_variables(N, name='a')

    hamiltonian = 0
    for i in range(N):
        hamiltonian += hbar*omega*(Dagger(a[i])*a[i]+0.5)

    monomial_substitutions, equalities = bosonic_constraints(a)
    inequalities = []

    time0 = time.time()

    print("Obtaining SDP relaxation...")
    verbose = 1
    sdpRelaxation = SdpRelaxation(a)
    sdpRelaxation.get_relaxation(level, objective=hamiltonian,
                                 equalities=equalities,
                                 substitutions=substitutions,
                                 removeequalities=True)
    write_to_sdpa(sdpRelaxation, 'harmonic_oscillator.dat-s')                      

Solving the SDP for :math:`N=4`, for instance, gives the following
result:

::

    objValPrimal = +1.9999998358414430e+00
    objValDual   = +1.9999993671869802e+00

This is very close to the analytic result of 2. The result is similarly
precise for arbitrary numbers of oscillators.

It is remarkable that we get the correct value at the first level of
relaxation, but this property is typical for bosonic systems (Navascués
et al. 2013).

Example 5: Using the Nieto-Silleras Hierarchy
==================================================

One of the newer approaches to the SDP relaxations takes all joint
probabilities into consideration when looking for a maximum guessing
probability, and not just the ones included in a particular Bell
inequality (Nieto-Silleras, Pironio, and Silman 2014; Bancal, Sheridan,
and Scarani 2014). Ncpol2sdpa can generate the respective hierarchy.

To deal with the joint probabilities necessary for setting constraints,
we also rely on QuTiP (Johansson, Nation, and Nori 2013):

::

    from math import sqrt
    from qutip import tensor, basis, sigmax, sigmay, expect, qeye

We will work in a CHSH scenario where we are trying to find the maximum
guessing probability of the first projector of Alice’s first
measurement. We generate the joint probability distribution on the
maximally entangled state with the measurements that give the maximum
quantum violation of the CHSH inequality:

::

    def joint_probabilities():
        psi = (tensor(basis(2,0),basis(2,0)) + 
               tensor(basis(2,1),basis(2,1))).unit()
        A_0 = sigmax()
        A_1 = sigmay()
        B_0 = (-sigmay()+sigmax())/sqrt(2)
        B_1 = (sigmay()+sigmax())/sqrt(2)

        A_00 = (qeye(2) + A_0)/2
        A_10 = (qeye(2) + A_1)/2
        B_00 = (qeye(2) + B_0)/2
        B_10 = (qeye(2) + B_1)/2

        p=[]
        p.append(expect(tensor(A_00, qeye(2)), psi))
        p.append(expect(tensor(A_10, qeye(2)), psi))
        p.append(expect(tensor(qeye(2), B_00), psi))
        p.append(expect(tensor(qeye(2), B_10), psi))

        p.append(expect(tensor(A_00, B_00), psi))
        p.append(expect(tensor(A_00, B_10), psi))
        p.append(expect(tensor(A_10, B_00), psi))
        p.append(expect(tensor(A_10, B_10), psi))
        return p

Next we need the basic configuration of the projectors. We also set the
level of the SDP relaxation and the objective.

::

    level = 1
    A_configuration = [2, 2]
    B_configuration = [2, 2]
    P_A = generate_measurements(A_configuration, 'P_A')
    P_B = generate_measurements(B_configuration, 'P_B')
    monomial_substitutions = projective_measurement_constraints(
        P_A, P_B)
    objective = -P_A[0][0]

We must define further constraints, namely that the joint probabilities
must match:

::

    probabilities = joint_probabilities()
    equalities = []
    k=0
    for i in range(len(A_configuration)):
        equalities.append(P_A[i][0] - probabilities[k])
        k += 1
    for i in range(len(B_configuration)):
        equalities.append(P_B[i][0] - probabilities[k])
        k += 1
    for i in range(len(A_configuration)):
        for j in range(len(B_configuration)):
            equalities.append(P_A[i][0]*P_B[j][0] - probabilities[k])
            k += 1

From here, the solution follows the usual pathway, indicating that we
are requesting the Nieto-Silleras hierarchy:

::

    sdpRelaxation = SdpRelaxation([flatten([P_A, P_B])], verbose=2,
                                   hierarchy="nieto-silleras")
    sdpRelaxation.get_relaxation(level, objective=objective, 
                                 equalities=equalities,
                                 substitutions=monomial_substitutions)

    print(solve_sdp(sdpRelaxation))

Example 6: Using the Moroder Hierarchy
==================================================

This type of hierarchy allows for a wider range of constraints of the
optimization problems, including ones that are not of polynomial
form (Moroder et al. 2013). These constraints are hard to impose using
SymPy and the sparse structures in Ncpol2Sdpa. For this reason, we
separate two steps: generating the SDP and post-processing the SDP to
impose extra constraints. This second step can be done in MATLAB, for
instance.

Then we set up the problem with specifically with the CHSH inequality in
the probability picture as the objective function. This part is
identical to the one discussed in Section [mixedlevel].

::

    level = 1
    A_configuration = [2, 2]
    B_configuration = [2, 2]
    I = [[ 0,   -1,    0 ],
         [-1,    1,    1 ], 
         [ 0,    1,   -1 ]]
    A = generate_measurements(A_configuration, 'A')
    B = generate_measurements(B_configuration, 'B')
    monomial_substitutions = projective_measurement_constraints(A, B)
    objective = define_objective_with_I(I, A, B)

When obtaining the relaxation for this kind of problem, it can prove
useful to disable the normalization of the top-left element of the
moment matrix. Naturally, before solving the problem this should be set
to zero, but further processing of the SDP matrix can be easier without
this constraint set a priori. Hence we write:

::

    sdpRelaxation = SdpRelaxation([flatten(A), flatten(B)], verbose=2,
                                   hierarchy="moroder", normalized=False)
    sdpRelaxation.get_relaxation(level, objective=objective,
                                 substitutions=monomial_substitutions)
    write_to_sdpa(sdpRelaxation, "chsh-moroder.dat-s")  

For instance, reading this file with SeDuMi’s ``fromsdpa``
function (Sturm 1999), we can impose the positivity of the partial trace
of the moment matrix, or decompose the moment matrix in various forms.

Example 7: Sparse Relaxation with Chordal Extension
===================================================
This method replicates the behaviour of SparsePOP (Waki et. al, 2008). It is 
invoked by defining the hierarchy as ``"npa_chordal"``. The following is a 
simple example:

::

    level = 2
    X = generate_variables(3, commutative=True)

    obj = X[1] - 2*X[0]*X[1] + X[1]*X[2]
    inequalities = [1-X[0]**2-X[1]**2, 1-X[1]**2-X[2]**2]

    sdpRelaxation = SdpRelaxation(X, hierarchy="npa_chordal")
    sdpRelaxation.get_relaxation(level, objective=obj, inequalities=inequalities)
    print(solve_sdp(sdpRelaxation))
