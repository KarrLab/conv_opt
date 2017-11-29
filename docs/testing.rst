Testing
=======

The package can be tested by running these commands:

    .. code-block:: bash

        pip install pytest
        python -m pytest tests

We tested the package with the following package versions:

* OS: Mint Linux 18.1 (based on Ubuntu 16.04)
* Python: 2.7.14, 3.6.3
* NumPy: 1.13.3
* SymPy: 1.1.1
* Solvers and solver interfaces:

    * Cbc: 2.8.5, CyLP: 0.7.4
    * CVXOPT: 1.1.9, CVXPY: 0.4.11
    * GLPK: 4.57, swiglpk: 1.4.4, optlang: 1.2.5
    * FICO XPRESS: 8.4.0
    * Gurobi: 7.5.1
    * IBM CPLEX: 12.7.1.0
    * MOSEK Optimizer: 8.1.33
    * quadprog: 0.1.6
    * SciPy: 1.0.0
