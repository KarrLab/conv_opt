""" Open-source, easily installable solvers """
from .glpk import GlpkModel
from .optlang import OptlangModel
from .quadprog import QuadprogModel
from .scipy import ScipyModel

""" Optional solvers """

# CVXOPT
try:
    from .cvxopt import CvxoptModel
except:  # pragma: no cover
    pass  # pragma: no cover

# CVXPY
try:
    from .cvxpy import CvxpyModel
except:  # pragma: no cover
    pass  # pragma: no cover

# IBM CPLEX
try:
    from .cplex import CplexModel
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

# CyLP/CBC
try:
    from .cylp import CylpModel
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

# Gurobi
try:
    from .gurobi import GurobiModel
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

# MINOS
try:
    from .minos import MinosModel
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

# MOSEK
try:
    from .mosek import MosekModel
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

# SoPlex
try:
    from .soplex import SoplexModel
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

# FICO XPRESS
try:
    from .xpress import XpressModel
except ImportError:  # pragma: no cover
    pass  # pragma: no cover
