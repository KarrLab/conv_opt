""" Open-source, easily installable solvers """
from .cvxopt import CvxoptModel
from .cvxpy import CvxpyModel
from .glpk import GlpkModel
from .optlang import OptlangModel
from .quadprog import QuadprogModel
from .scipy import ScipyModel

""" Optional solvers """

# IBM CPLEX
try:
    from .cplex import CplexModel
except ImportError:
    pass

# CyLP/CBC
try:
    from .cylp import CylpModel
except ImportError:
    pass

# Gurobi
try:
    from .gurobi import GurobiModel
except ImportError:
    pass

# MOSEK
try:
    from .mosek import MosekModel
except ImportError:
    pass

# FICO XPRESS
try:
    from .xpress import XpressModel
except ImportError:
    pass
