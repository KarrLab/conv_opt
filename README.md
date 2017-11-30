[![PyPI package](https://img.shields.io/pypi/v/conv_opt.svg)](https://pypi.python.org/pypi/conv_opt)
[![Documentation](https://readthedocs.org/projects/conv_opt/badge/?version=latest)](http://conv_opt.readthedocs.org)
[![Test results](https://circleci.com/gh/KarrLab/conv_opt.svg?style=shield)](https://circleci.com/gh/KarrLab/conv_opt)
[![Test coverage](https://coveralls.io/repos/github/KarrLab/conv_opt/badge.svg)](https://coveralls.io/github/KarrLab/conv_opt)
[![Code analysis](https://api.codeclimate.com/v1/badges/f61deab196a9dbf42555/maintainability)](https://codeclimate.com/github/KarrLab/conv_opt)
[![License](https://img.shields.io/github/license/KarrLab/conv_opt.svg)](LICENSE)
![Analytics](https://ga-beacon.appspot.com/UA-86759801-1/conv_opt/README.md?pixel)

# conv_opt

`conv_opt` is a high-level Python package for solving linear and quadratic optimization problems using
multiple open-source and commercials solvers including [Cbc](https://projects.coin-or.org/cbc),
[CVXOPT](http://cvxopt.org), [FICO XPRESS](http://www.fico.com/en/products/fico-xpress-optimization),
[GLPK](https://www.gnu.org/software/glpk), [Gurobi](http://www.gurobi.com/products/gurobi-optimizer),
[IBM CPLEX](https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer),
[Mosek](https://www.mosek.com), [quadprog](https://github.com/rmcgibbo/quadprog), and 
[SciPy](https://docs.scipy.org).

## Installation

1. Install Python and pip
2. Install this package. This will install the package, as well as the CVXOPT, GLPK, quadprog, and SciPy solvers.
   ```
   pip install git+git://github.com/KarrLab/conv_opt#egg=conv_opt
   ```
3. Optionally, install the Cbc/CyLP, FICO XPRESS, IBM CPLEX, Gurobi, and Mosek solvers. Please see our detailed [instructions](http://intro-to-wc-modeling.readthedocs.io/en/latest/installation.html).

## Documentation
Please see the [API documentation](http://conv_opt.readthedocs.io).

## License
The build utilities are released under the [MIT license](LICENSE).

## Development team
This package was developed by the [Karr Lab](http://www.karrlab.org) at the Icahn School of Medicine at Mount Sinai in New York, USA.

## Questions and comments
Please contact the [Karr Lab](http://www.karrlab.org) with any questions or comments.
