Features
========

Problems
--------------------------------

This package supports two types of optimization problems:

* Linear problems

    .. math::

        \text{Maximize}~& c' x \\
        \text{Subject to} & \\
            A x &= b \\
            x_l \leq x & \leq x_u

* Quadratic problems

    .. math::

        \text{Maximize}~& x' Q x + c' x \\
        \text{Subject to} & \\
            A x & = b \\
            x_l \leq x & \leq x_u


Objectives
--------------------------------

This package supports two types of objectives:

* linear
* quadratic


Variable types
--------------------------------

This package supports five types of variables:

* binary
* integer
* continuous
* semi-integer
* semi-continuous
* partially-integer


Constraints
--------------------------------

This package supports one type of constraint:

* linear


Solvers
--------------------------------

This package supports several solvers:

* Open-source

    * `Cbc <https://projects.coin-or.org/cbc>`_ via `CyLP <https://github.com/coin-or/CyLP>`_
    * `CVXOPT <http://cvxopt.org>`_ via `CVXPY <https://cvxgrp.github.io>`_
    * `GLPK <https://www.gnu.org/software/glpk>`_ via `optlang <http://optlang.readthedocs.io>`_    
    * `quadprog <https://github.com/rmcgibbo/quadprog>`_
    * `SciPy <https://docs.scipy.org>`_    

* Non-commercial with free academic licenses

    * `MINOS <https://web.stanford.edu/group/SOL/minos.htm>`_ via `solveME <https://github.com/SBRG/solvemepy>`_
    * `SoPlex <http://soplex.zib.de>`_ via `soplex_cython <https://github.com/SBRG/soplex_cython>`_

* Commercial with free academic licenses

    * `FICO XPRESS <http://www.fico.com/en/products/fico-xpress-optimization>`_
    * `IBM CPLEX <https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer>`_
    * `Gurobi <http://www.gurobi.com/products/gurobi-optimizer>`_
    * `MOSEL Optimizer <https://www.mosek.com>`_

However, as described below, some of the solvers only support some of these features.

Objective types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Cbc: only supports linear objectives
* GLPK: only supports linear objectives
* quadprog: only supports quadratic objectives


Variable types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Cbc: only supports binary, integer, and continuous variables
* CVXOPT: only supports continuous variables
* FICO XPRESS: supports all variable types
* GLPK: only supports binary, integer, and continuous variables
* Gurobi: doesn't support partially integer variables
* IBM CPLEX: doesn't support partially integer variables
* MINOS: only supports continuous variables
* MOSEK Optimizer: only supports binary, integer, and continuous variables
* quadprog: only supports continuous variables
* Scipy: only supports continuous variables


Constraint types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cbc, GLPK, and quadprog only support linear constraints. Only SciPy's COBLYA and SLSQP method support contraints.


Python versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* CPLEX supports Python 2.7, 3.5, and 3.6


Licensing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* FICO XPRESS: licenses are tied to machines, or a license server must be used
* Gurobi: licenses are tied to machines, or a license server must be used
* IBM CPLEX: No license file or activation is needed
* MINOS: free academic licenses can be obtained from Michael Saunders at Stanford
* MOSEL Optimizer: license files can be used on multiple machines
* SoPlex: free academic licenses can be automatically obtained from the SoPlex website
