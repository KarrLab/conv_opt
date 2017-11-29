import pkg_resources

with open(pkg_resources.resource_filename('conv_opt', 'VERSION'), 'r') as file:
    __version__ = file.read().strip()
# :obj:`str`: version

from .core import (ExportFormat, ModelType, ObjectiveDirection, Presolve,
                   SolveOptions, Solver, StatusCode, VariableType, Verbosity,
                   Constraint, LinearTerm, Model, QuadraticTerm, Term, Variable, Result, ConvOptError,
                   ENABLED_SOLVERS, SolverModel)
