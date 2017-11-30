""" Module for solving linear and quadratic optimization problems using
multiple open-source and commercials solvers.

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2017-11-14
:Copyright: 2017, Karr Lab
:License: MIT
"""

import abc
import attrdict
import enum
import numpy
import os
import six


class VariableType(enum.Enum):
    """ Variable type """
    binary = 0
    continuous = 1
    integer = 2
    semi_integer = 3
    semi_continuous = 4
    partially_integer = 5


class ObjectiveDirection(enum.Enum):
    """ Direction to solve a mathematical model """
    max = 0
    maximize = 0
    min = 1
    minimize = 1


class ModelType(enum.Enum):
    """ Model type """
    fixed_milp = 0
    fixed_miqp = 1
    lp = 2
    milp = 3
    miqp = 4
    qp = 5


class Solver(enum.Enum):
    """ Solver """
    cbc = 0
    cplex = 1
    cvxopt = 2
    glpk = 3
    gurobi = 4
    mosek = 5
    quadprog = 6
    scipy = 7
    xpress = 8


class Presolve(enum.Enum):
    """ Presolve mode """
    auto = 0
    on = 1
    off = 2


class Verbosity(enum.Enum):
    """ Verbosity level """
    off = 0
    error = 1
    warning = 2
    status = 3


class StatusCode(enum.Enum):
    """ Status code for the result of solving a mathematical model """
    optimal = 0
    infeasible = 1
    other = 2


class ExportFormat(enum.Enum):
    """ Export format """
    alp = 0
    cbf = 1
    dpe = 2
    dua = 3
    jktask = 4
    lp = 5
    mps = 6
    opf = 7
    ppe = 8
    rew = 9
    rlp = 10
    sav = 11
    task = 12
    xml = 13


ENABLED_SOLVERS = [Solver.cvxopt, Solver.glpk, Solver.quadprog, Solver.scipy]
# :obj:`list` of :obj:`Solver`: list of enabled solvers

try:
    import cylp.cy
    try:
        cylp.cy.CyClpSimplex()
        ENABLED_SOLVERS.append(Solver.cbc)
    except:  # pragma: no cover
        pass  # pragma: no cover
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

try:
    import cplex
    try:
        cplex.Cplex()
        ENABLED_SOLVERS.append(Solver.cplex)
    except cplex.exceptions.CplexError:  # pragma: no cover
        pass  # pragma: no cover
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

try:
    import gurobipy
    try:
        gurobipy.Model()
        ENABLED_SOLVERS.append(Solver.gurobi)
    except gurobipy.GurobiError:  # pragma: no cover
        pass  # pragma: no cover
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

try:
    import mosek
    try:
        mosek.Env()
        ENABLED_SOLVERS.append(Solver.mosek)
    except mosek.Error:  # pragma: no cover
        pass  # pragma: no cover
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

try:
    import xpress
    try:
        xpress.problem()
        ENABLED_SOLVERS.append(Solver.xpress)
    except:  # pragma: no cover
        pass  # pragma: no cover
except (ImportError, RuntimeError):  # pragma: no cover
    pass  # pragma: no cover


class Variable(object):
    """ A variable

    Attributes:
        name (:obj:`str`): name
        type (:obj:`VariableType`): type
        lower_bound (:obj:`float`): lower bound
        upper_bound (:obj:`float`): upper bound
        primal (:obj:`float`): primal value
        reduced_cost (:obj:`float`): reduced cost
    """

    def __init__(self, name='', type=VariableType.continuous, lower_bound=None, upper_bound=None):
        """
        Args:
            name (:obj:`str`, optional): name
            type (:obj:`VariableType`, optional): type
            lower_bound (:obj:`float`, optional): lower bound
            upper_bound (:obj:`float`, optional): upper bound
        """
        self.name = name
        self.type = type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.primal = numpy.nan
        self.reduced_cost = numpy.nan


class Term(object):
    """ Term (of an objective or contraint)

    Attributes:
        coefficient (:obj:`float`): coefficient
    """

    def __init__(self, coefficient):
        """
        Args:
            coefficient (:obj:`float`): coefficient
        """
        self.coefficient = coefficient


class LinearTerm(Term):
    """ Linear term (of an objective or contraint)

    Attributes:
        variable (:obj:`Variable`): variable
    """

    def __init__(self, variable, coefficient):
        """
        Args:
            variable (:obj:`Variable`): variable
            coefficient (:obj:`float`): coefficient
        """
        super(LinearTerm, self).__init__(coefficient)
        self.variable = variable


class QuadraticTerm(Term):
    """ Quadtratic term (of an objective or contraint)

    Attributes:
        variable_1 (:obj:`Variable`): first variable
        variable_2 (:obj:`Variable`): second variable
    """

    def __init__(self, variable_1, variable_2, coefficient):
        """
        Args:
            variable_1 (:obj:`Variable`): first variable
            variable_2 (:obj:`Variable`): second variable
            coefficient (:obj:`float`): coefficient
        """
        super(QuadraticTerm, self).__init__(coefficient)
        self.variable_1 = variable_1
        self.variable_2 = variable_2


class Constraint(object):
    """ A constraint

    Attributes:
        name (:obj:`str`): name
        terms (:obj:`list` of :obj:`Term`): the variables and their coefficients
        lower_bound (:obj:`float`): lower bound
        upper_bound (:obj:`float`): upper bound
        dual (:obj:`float`): dual value
    """

    def __init__(self, terms=None, name='', lower_bound=None, upper_bound=None):
        """
        Args:
            terms (:obj:`list` of :obj:`Term`, optional): the variables and their coefficients
            name (:obj:`str`, optional): name
            lower_bound (:obj:`float`, optional): lower bound
            upper_bound (:obj:`float`, optional): upper bound
        """
        self.terms = terms or []
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dual = numpy.nan


class SolveOptions(object):
    """ Options for :obj:`Model.solve`

    Attributes:
        solver (:obj:`Solver`): solver
        tune (:obj:`bool`): tune (used by Gurobi)
        presolve (:obj:`Presolve`): presolve
        round_results_to_bounds (:obj:`bool`): if :obj:`True`, round the results to the variable bounds
        verbosity (:obj:`Verbosity`): determines how much status, warnings, and errors is printed out
        solver_options (:obj:`attrdict.AttrDict`): solver options
    """

    def __init__(self, solver=Solver.cplex, tune=False, presolve=Presolve.off,
                 round_results_to_bounds=True, verbosity=Verbosity.off, solver_options=None):
        """
        Args:
            solver (:obj:`Solver`, optional): solver
            tune (:obj:`bool`, optional): tune (used by Gurobi)
            presolve (:obj:`Presolve`, optional): presolve
            round_results_to_bounds (:obj:`bool`, optional): if :obj:`True`, round the results to the variable bounds
            verbosity (:obj:`Verbosity`, optional): determines how much status, warnings, and errors is printed out
            solver_options (:obj:`attrdict.AttrDict`, optional): options for specific solvers
        """
        self.solver = solver
        self.tune = tune
        self.presolve = presolve
        self.round_results_to_bounds = round_results_to_bounds
        self.verbosity = verbosity
        self.solver_options = solver_options or attrdict.AttrDict()


class Model(object):
    """ A mathematical model

    Attributes:
        name (:obj:`str`): name
        variables (:obj:`list` of :obj:`Variable`): the variables, :math:`x`
        objective_direction (:obj:`ObjectiveDirection`): objective direction
        objective_terms (:obj:`list` of :obj:`LinearTerm`): the elements of the objective, :math:`c` and :math:`Q`
        constraints (:obj:`list` of :obj:`LinearTerm`): the constraints, :math:`A` and :math:`b`
    """

    def __init__(self, name='', variables=None, objective_direction=ObjectiveDirection.minimize, objective_terms=None, constraints=None):
        """
        Args:
            name (:obj:`str`, optional): name
            variables (:obj:`list` of :obj:`Variable`, optional): the variables, :math:`x`
            objective_direction (:obj:`ObjectiveDirection`, optional): objective direction
            objective_terms (:obj:`list` of :obj:`LinearTerm`, optional): the elements of the objective, :math:`c` and :math:`Q`
            constraints (:obj:`list` of :obj:`LinearTerm`, optional): the constraints, :math:`A` and :math:`b`
        """
        self.name = name
        self.variables = variables or []
        self.objective_direction = objective_direction
        self.objective_terms = objective_terms or []
        self.constraints = constraints or []

    def get_type(self):
        """ Get the type of the model

        Returns:
            :obj:`ModelType`: model type
        """
        has_integer = False
        for variable in self.variables:
            if variable.type in [VariableType.binary, VariableType.integer, VariableType.semi_integer]:
                has_integer = True
                break
            elif variable.type not in [VariableType.continuous, VariableType.semi_continuous]:
                raise ConvOptError('Unsupported variable of type "{}"'.format(variable.type))

        is_linear = True
        for term in self.objective_terms:
            if isinstance(term, QuadraticTerm):
                is_linear = False
                break
            elif not isinstance(term, LinearTerm):
                raise ConvOptError('Unsupported objective term of type "{}"'.format(term.__class__.__name__))

        if is_linear:
            if has_integer:
                return ModelType.milp
            else:
                return ModelType.lp
        else:
            if has_integer:
                return ModelType.miqp
            else:
                return ModelType.qp

    def convert(self, options=None):
        """ Generate a data structure for the model for another package

        Args:
            options (:obj:`SolveOptions`, optional): options

        Returns:
            :obj:`object`: model in a data structure for another package

        Raises:
            :obj:`ConvOptError`: if the solver is not supported
        """

        options = options or SolveOptions()

        if options.solver == Solver.cbc:
            from .solver import cbc
            return cbc.CbcModel(self, options)
        elif options.solver == Solver.cplex:
            from .solver import cplex
            return cplex.CplexModel(self, options)
        elif options.solver == Solver.cvxopt:
            from .solver import cvxopt
            return cvxopt.CvxoptModel(self, options)
        elif options.solver == Solver.glpk:
            from .solver import glpk
            return glpk.GlpkModel(self, options)
        elif options.solver == Solver.gurobi:
            from .solver import gurobi
            return gurobi.GurobiModel(self, options)
        elif options.solver == Solver.mosek:
            from .solver import mosek
            return mosek.MosekModel(self, options)
        elif options.solver == Solver.quadprog:
            from .solver import quadprog
            return quadprog.QuadprogModel(self, options)
        elif options.solver == Solver.scipy:
            from .solver import scipy
            return scipy.ScipyModel(self, options)
        elif options.solver == Solver.xpress:
            from .solver import xpress
            return xpress.XpressModel(self, options)
        else:
            raise ConvOptError('Unsupported solver "{}"'.format(options.solver))

    def solve(self, options=None):
        """ Solve the model

        Args:
            options (:obj:`SolveOptions`, optional): options

        Returns:
            :obj:`Result`: result

        Raises:
            :obj:`ConvOptError`: if the solver is not supported
        """

        options = options or SolveOptions()

        # solve model
        solver_model = self.convert(options=options)
        result = solver_model.solve()

        # round results to variable bounds
        if options.round_results_to_bounds:
            for i_variable, variable in enumerate(self.variables):
                if variable.lower_bound is not None:
                    result.primals[i_variable] = max(result.primals[i_variable], variable.lower_bound)

                if variable.upper_bound is not None:
                    result.primals[i_variable] = min(result.primals[i_variable], variable.upper_bound)

        # assign primal and dual attributes of the variables and constraints
        self._unpack_result(result)

        # return result
        return result

    def _unpack_result(self, result):
        """ Assign primal and dual attributes of the variables and constraints

        Args:
            result (:obj:`Result`): result
        """
        for i_variable, (variable, primal, reduced_cost) in enumerate(zip(self.variables, result.primals, result.reduced_costs)):
            variable.primal = primal
            variable.reduced_cost = reduced_cost

        for constraint, dual in zip(self.constraints, result.duals):
            constraint.dual = dual

    def export(self, filename, format=None, solver=None):
        """ Export a model to a file in one of these support formats

        * **alp**: model with generic names in lp format, where the variable names are annotated to indicate the type and bounds of each variable
        * **cbf**
        * **dpe**: dual perturbed model
        * **dua**: dual
        * **jtask**: Jtask format
        * **lp**
        * **mps**
        * **opf**
        * **ppe**: perturbed model
        * **rew**: model with generic names in mps format
        * **rlp**: model with generic names in lp format
        * **sav**
        * **task**: Task format
        * **xml**: OSiL

        Args:
            filename (:obj:`str`): path to save model
            format (:obj:`str`, optional): export format; if the format is not provided, the
                format is inferred from the filename
            solver (:obj:`Solver`, optional): desired solver to do the exporting; if none,
                a supported solver will be found

        Raises:
            :obj:`ConvOptError`: if the format is not supported
        """
        if not format:
            format = os.path.splitext(filename)[1][1:]

        if solver is None:
            preferred_solvers = ENABLED_SOLVERS
        else:
            preferred_solvers = [solver]

        if Solver.cbc in preferred_solvers and format in ['lp']:
            from .solver import cbc
            simplex = cbc.CbcModel(self).get_model()
            simplex.writeLp(filename)
        elif Solver.cbc in preferred_solvers and format in ['mps']:
            from .solver import cbc
            simplex = cbc.CbcModel(self).get_model()
            simplex.writeMps(filename)
        elif Solver.cplex in preferred_solvers and format in ['alp', 'dpe', 'dua', 'lp', 'mps', 'ppe', 'rew', 'lp', 'rlp', 'sav']:
            from .solver import cplex
            with cplex.CplexModel(self).get_model() as model:
                model.write(filename, filetype=format)
        elif Solver.gurobi in preferred_solvers and format in ['lp', 'mps', 'rew', 'rlp']:
            from .solver import gurobi
            model = gurobi.GurobiModel(self).get_model()
            model.write(filename)
        elif Solver.mosek in preferred_solvers and format in ['cbf', 'jtask', 'lp', 'mps', 'opf', 'task', 'xml']:
            from .solver import mosek
            with mosek.MosekModel(self).get_model() as task:
                task.writedata(filename)
        elif Solver.xpress in preferred_solvers and format in ['lp', 'mps']:
            from .solver import xpress
            model = xpress.XpressModel(self).get_model()
            if format == 'lp':
                flags = 'l'
            else:
                flags = 'x'
            model.write(filename, flags)
        else:
            raise ConvOptError('Unsupported format "{}"'.format(format))


class SolverModel(six.with_metaclass(abc.ABCMeta, object)):
    """ A solver

    Attributes:
        _model (:obj:`Model`): model
        options (:obj:`SolveOptions`): options
    """

    def __init__(self, model, options=None):
        """
        Args:
            model (:obj:`Model`): model
            options (:obj:`SolveOptions`, optional): options
        """
        self._options = options or SolveOptions()
        self._model = self.load(model)

    def get_model(self):
        """ Get the model in the raw format of the solver. This is useful for acessing specific
        properties and methods of the solver.

        Returns:
            :obj:`object`: the model in the format of the solver
        """
        return self._model

    @abc.abstractmethod
    def load(self, conv_opt_model):
        """ Load a model to the data structure of the solver

        Args:
            conv_opt_model (:obj:`Model`): model

        Returns:
            :obj:`object`: the model in the data structure of the solver
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def solve(self):
        """ Solve the model

        Returns:
            :obj:`Result`: result
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def get_stats(self):
        """ Get diagnostic information about the model

        Returns:
            :obj:`object`: diagnostic information about the model
        """
        pass  # pragma: no cover


class Result(object):
    """ The result of solving a mathematical model

    Attributes:
        status_code (:obj:`StatusCode`): status code
        status_message (:obj:`str`): status message
        value (:obj:`float`): objective value
        primals (:obj:`numpy.ndarray`): primal values
        reduced_costs (:obj:`numpy.ndarray`): reduced costs
        duals (:obj:`numpy.ndarray`): dual values/shadow prices
    """

    def __init__(self, status_code, status_message, value, primals, reduced_costs, duals):
        """
        Args:
            status_code (:obj:`StatusCode`): status code
            status_message (:obj:`str`): status message
            value (:obj:`float`): objective value
            primals (:obj:`numpy.ndarray`): primal values
            reduced_costs (:obj:`numpy.ndarray`): reduced costs
            duals (:obj:`numpy.ndarray`): dual values/reduced costs/shadow prices
        """
        self.status_code = status_code
        self.status_message = status_message
        self.value = value
        self.primals = primals
        self.reduced_costs = reduced_costs
        self.duals = duals


class ConvOptError(Exception):
    """ conv_opt exception """
    pass
