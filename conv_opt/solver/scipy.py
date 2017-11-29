""" SciPy module

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2017-11-22
:Copyright: 2017, Karr Lab
:License: MIT
"""

from __future__ import absolute_import
from ..core import (ModelType, ObjectiveDirection, Presolve,
                    SolveOptions, Solver, StatusCode, VariableType, Verbosity,
                    Constraint, LinearTerm, Model, QuadraticTerm, Term, Variable, Result, ConvOptError,
                    SolverModel)
import copy
import numpy
import scipy


class ScipyModel(SolverModel):
    """ SciPy solver

    Attributes:
        _stats (:obj:`dict`): solver statistics
    """

    def __init__(self, model, options=None):
        """
        Args:
            model (:obj:`Model`): model
            options (:obj:`SolveOptions`, optional): options
        """
        super(ScipyModel, self).__init__(model, options=options)
        self._stats = {}

    def load(self, conv_opt_model):
        """ Load a model to SciPy's data structure

        Args:
            conv_opt_model (:obj:`Model`): model

        Returns:
            :obj:`dict`: the model in SciPy's data structure

        Raises:
            :obj:`ConvOptError`: if the model is not of a supported type
        """
        type = conv_opt_model.get_type()
        if type == ModelType.lp:
            return self._load_linprog(conv_opt_model)
        elif type == ModelType.qp:
            return self._load_minimize(conv_opt_model)
        else:
            raise ConvOptError('Unsupported model type "{}"'.format(type)) 

    def _load_linprog(self, conv_opt_model):
        """ Load a model to SciPy's data structure

        Args:
            conv_opt_model (:obj:`Model`): model

        Returns:
            :obj:`dict`: the model in SciPy's data structure

        Raises:
            :obj:`ConvOptError`: if the presolve mode is unsupported, a variable has an unsupported type,
                an objective has an unsupported term, a constraint
                has an unsupported term, a constraint is unbounded, or the model is not of a supported type
        """

        # options
        method = getattr(self._options.solver_options, 'method', 'simplex')

        # variables
        bounds = []
        for variable in conv_opt_model.variables:
            if variable.type != VariableType.continuous:
                raise ConvOptError('Unsupported variable of type "{}"'.format(variable.type))

            if variable.lower_bound is None:
                lb = -numpy.inf
            else:
                lb = variable.lower_bound

            if variable.upper_bound is None:
                ub = numpy.inf
            else:
                ub = variable.upper_bound

            bounds.append((lb, ub))

        # objective
        c = numpy.zeros((len(conv_opt_model.variables), ))
        for term in conv_opt_model.objective_terms:
            if not isinstance(term, LinearTerm):
                raise ConvOptError('Unsupported objective term of type "{}"'.format(term.__class__.__name__))
            i_variable = conv_opt_model.variables.index(term.variable)
            c[i_variable] += float(term.coefficient)

        if conv_opt_model.objective_direction in [ObjectiveDirection.max, ObjectiveDirection.maximize]:
            c *= -1.
        elif conv_opt_model.objective_direction not in [ObjectiveDirection.min, ObjectiveDirection.minimize]:
            raise ConvOptError('Unsupported objective direction "{}"'.format(conv_opt_model.objective_direction))

        # contraints
        A_ub = numpy.zeros((0, len(conv_opt_model.variables)))
        A_eq = numpy.zeros((0, len(conv_opt_model.variables)))
        b_ub = numpy.zeros((0,))
        b_eq = numpy.zeros((0,))

        for constraint in conv_opt_model.constraints:
            A_row = numpy.zeros((1, len(conv_opt_model.variables)))
            for term in constraint.terms:
                if not isinstance(term, LinearTerm):
                    raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))
                i_variable = conv_opt_model.variables.index(term.variable)
                A_row[0, i_variable] += float(term.coefficient)

            if constraint.lower_bound is None and constraint.upper_bound is None:
                raise ConvOptError('Constraints must have at least one bound')
            elif constraint.lower_bound is None:
                A_ub = numpy.concatenate((A_ub, A_row))
                b_ub = numpy.concatenate((b_ub, [float(constraint.upper_bound)]))
            elif constraint.upper_bound is None:
                A_ub = numpy.concatenate((A_ub, -A_row))
                b_ub = numpy.concatenate((b_ub, [-float(constraint.lower_bound)]))
            elif constraint.lower_bound == constraint.upper_bound:
                A_eq = numpy.concatenate((A_eq, A_row))
                b_eq = numpy.concatenate((b_eq, [float(constraint.lower_bound)]))
            else:
                A_ub = numpy.concatenate((A_ub, A_row))
                A_ub = numpy.concatenate((A_ub, -A_row))
                b_ub = numpy.concatenate((b_ub, [float(constraint.upper_bound)]))
                b_ub = numpy.concatenate((b_ub, [-float(constraint.lower_bound)]))

        return {
            '_method': 'linprog',
            'c': c,
            'A_ub': A_ub,
            'b_ub': b_ub,
            'A_eq': A_eq,
            'b_eq': b_eq,
            'bounds': bounds,
            'method': method,
            '_objective_direction': conv_opt_model.objective_direction,
            '_num_variables': len(conv_opt_model.variables),
            '_num_constraints': len(conv_opt_model.constraints),
        }

    def _load_minimize(self, conv_opt_model):
        """ Load a model to SciPy's data structure

        Args:
            conv_opt_model (:obj:`Model`): model

        Returns:
            :obj:`dict`: the model in SciPy's data structure

        Raises:
            :obj:`ConvOptError`: if the presolve mode is unsupported, a variable has an unsupported type,
                an objective has an unsupported term, a constraint
                has an unsupported term, a constraint is unbounded, or the model is not of a supported type
        """

        # options
        method = getattr(self._options.solver_options, 'method', 'COBYLA')
        tol = getattr(self._options.solver_options, 'tol', 1e-10)
        x0 = getattr(self._options.solver_options, 'x0', numpy.zeros((len(conv_opt_model.variables), )))

        # variables
        for variable in conv_opt_model.variables:
            if variable.type != VariableType.continuous:
                raise ConvOptError('Unsupported variable of type "{}"'.format(variable.type))

        # objective
        c = numpy.zeros((len(conv_opt_model.variables), ))
        Q = numpy.zeros((len(conv_opt_model.variables), len(conv_opt_model.variables)))
        for term in conv_opt_model.objective_terms:
            if isinstance(term, LinearTerm):
                i_variable = conv_opt_model.variables.index(term.variable)
                c[i_variable] += float(term.coefficient)
            elif isinstance(term, QuadraticTerm):
                i_variable_1 = conv_opt_model.variables.index(term.variable_1)
                i_variable_2 = conv_opt_model.variables.index(term.variable_2)
                Q[i_variable_1, i_variable_2] += 0.5 * float(term.coefficient)
                Q[i_variable_2, i_variable_1] += 0.5 * float(term.coefficient)
            else:
                raise ConvOptError('Unsupported objective term of type "{}"'.format(term.__class__.__name__))

        if conv_opt_model.objective_direction in [ObjectiveDirection.max, ObjectiveDirection.maximize]:
            c *= -1.
            Q *= -1.
        elif conv_opt_model.objective_direction not in [ObjectiveDirection.min, ObjectiveDirection.minimize]:
            raise ConvOptError('Unsupported objective direction "{}"'.format(conv_opt_model.objective_direction))

        def fun(x):
            return numpy.dot(x, Q).dot(x) + numpy.dot(c, x)

        # contraints
        constraints = []
        for constraint in conv_opt_model.constraints:
            A_row = numpy.zeros((1, len(conv_opt_model.variables)))
            for term in constraint.terms:
                if not isinstance(term, LinearTerm):
                    raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))
                i_variable = conv_opt_model.variables.index(term.variable)
                A_row[0, i_variable] += float(term.coefficient)

            if constraint.lower_bound is None and constraint.upper_bound is None:
                raise ConvOptError('Constraints must have at least one bound')

            if constraint.lower_bound is not None:
                b = float(constraint.lower_bound)
                constraints.append({'type': 'ineq', 'fun': lambda x, A_row=A_row, b=b: numpy.dot(A_row, x) - b})

            if constraint.upper_bound is not None:
                b = float(constraint.upper_bound)
                constraints.append({'type': 'ineq', 'fun': lambda x, A_row=A_row, b=b: -numpy.dot(A_row, x) + b})

        # variable bounds
        if method == 'COBYLA':
            bounds = None
            for i_variable, variable in enumerate(conv_opt_model.variables):
                if variable.lower_bound is not None:
                    b = float(variable.lower_bound)
                    constraints.append({'type': 'ineq', 'fun': lambda x, i_variable=i_variable, b=b: x[i_variable] - b})

                if variable.upper_bound is not None:
                    b = float(variable.upper_bound)
                    constraints.append({'type': 'ineq', 'fun': lambda x, i_variable=i_variable, b=b: -x[i_variable] + b})
        else:
            bounds = []
            for variable in conv_opt_model.variables:
                if variable.lower_bound is None:
                    lb = -numpy.inf
                else:
                    lb = variable.lower_bound

                if variable.upper_bound is None:
                    ub = numpy.inf
                else:
                    ub = variable.upper_bound

                bounds.append((lb, ub))

        return {
            '_method': 'minimize',
            'fun': fun,
            'x0': x0,
            'bounds': bounds,
            'constraints': constraints,
            'method': method,
            'tol': tol,
            '_objective_direction': conv_opt_model.objective_direction,
            '_num_variables': len(conv_opt_model.variables),
            '_num_constraints': len(conv_opt_model.constraints),
        }

    def solve(self):
        """ Solve the model

        Returns:
            :obj:`Result`: result
        """
        # presolve
        if self._options.presolve != Presolve.off:
            raise ConvOptError('Unsupported presolve mode {}'.format(self._options.presolve))

        # tune
        if self._options.tune:
            raise ConvOptError('Unsupported tuning mode {}'.format(self._options.tune))

        # solve
        model = self._model

        kwargs = copy.copy(model)
        del(kwargs['_method'])
        del(kwargs['_objective_direction'])
        del(kwargs['_num_variables'])
        del(kwargs['_num_constraints'])

        if self._model['_method'] == 'linprog':
            del(kwargs['c'])
            result = scipy.optimize.linprog(self._model['c'], **kwargs)
        else:
            del(kwargs['fun'])
            del(kwargs['x0'])
            result = scipy.optimize.minimize(self._model['fun'], self._model['x0'], **kwargs)

        if model['method'] == 'simplex':
            if result.status == 0:
                status_code = StatusCode.optimal
            elif result.status == 2:
                status_code = StatusCode.infeasible
            else:
                status_code = StatusCode.other
        elif model['method'] == 'COBYLA':
            if result.status == 1:
                status_code = StatusCode.optimal
            elif result.status == 4:
                status_code = StatusCode.infeasible
            else:
                status_code = StatusCode.other
        elif model['method'] == 'SLSQP':
            if result.status == 0:
                status_code = StatusCode.optimal
            elif result.status in [2, 4]:
                status_code = StatusCode.infeasible
            else:
                status_code = StatusCode.other
        else:
            raise ConvOptError('Unsupported solver method "{}"'.format(model['method']))

        status_message = result.message
        print(result.success, result.status, status_message)

        self._stats.clear()
        self._stats['slacks'] = None
        self._stats['num_iterations'] = numpy.nan

        if status_code == StatusCode.optimal:
            value = result.fun
            if model['_objective_direction'] in [ObjectiveDirection.max, ObjectiveDirection.maximize]:
                value *= -1

            primals = result.x
            if hasattr(result, 'slack'):
                self._stats['slacks'] = result.slack
            if hasattr(result, 'nit'):
                self._stats['num_iterations'] = result.nit
        else:
            value = numpy.nan
            primals = numpy.full((model['_num_variables'], ), numpy.nan)

        reduced_costs = numpy.full((model['_num_variables'], ), numpy.nan)

        duals = numpy.full((model['_num_constraints'], ), numpy.nan)

        return Result(status_code, status_message, value, primals, reduced_costs, duals)

    def get_stats(self):
        """ Get diagnostic information about the model

        Returns:
            :obj:`dict`: diagnostic information about the model
        """
        return self._stats
