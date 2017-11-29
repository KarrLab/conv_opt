""" CVXPY module

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
import cvxpy
import numpy


class CvxpyModel(SolverModel):
    """ CVXPY solver

    Attributes:
        SOLVER (:obj:`str`): desired solver
    """
    SOLVER = None

    def load(self, conv_opt_model):
        """ Load a model to CVXPY's data structure

        Args:
            conv_opt_model (:obj:`cvxpy.Problem`): model

        Returns:
            :obj:`dict`: the model in CVXPY's data structure

        Raises:
            :obj:`ConvOptError`: if the presolve mode is unsupported, a variable has an unsupported type,
                an objective has an unsupported term, a constraint
                has an unsupported term, a constraint is unbounded, or the model is not of a supported type
        """

        # variables
        if len(conv_opt_model.variables) > len(set([v.name for v in conv_opt_model.variables])):
            raise ConvOptError('Variables must have unique names')

        solver_vars = []
        for variable in conv_opt_model.variables:
            if variable.type == VariableType.binary:
                solver_var = cvxpy.Bool(name=variable.name)
            elif variable.type == VariableType.integer:
                solver_var = cvxpy.Int(name=variable.name)
            elif variable.type == VariableType.continuous:
                solver_var = cvxpy.Variable(name=variable.name)
            else:
                raise ConvOptError('Unsupported variable of type "{}"'.format(variable.type))
            solver_vars.append(solver_var)

        # objective
        expr = None
        for term in conv_opt_model.objective_terms:
            if isinstance(term, LinearTerm):
                i_variable = conv_opt_model.variables.index(term.variable)
                if expr is None:
                    expr = term.coefficient * solver_vars[i_variable]
                else:
                    expr += term.coefficient * solver_vars[i_variable]
            elif isinstance(term, QuadraticTerm):
                i_variable_1 = conv_opt_model.variables.index(term.variable_1)
                i_variable_2 = conv_opt_model.variables.index(term.variable_2)
                if expr is None:
                    expr = term.coefficient * solver_vars[i_variable_1] * solver_vars[i_variable_2]
                else:
                    expr += term.coefficient * solver_vars[i_variable_1] * solver_vars[i_variable_2]
            else:
                raise ConvOptError('Unsupported objective term of type "{}"'.format(term.__class__.__name__))

        if conv_opt_model.objective_direction in [ObjectiveDirection.max, ObjectiveDirection.maximize]:
            solver_obj = cvxpy.Maximize(expr)
        elif conv_opt_model.objective_direction in [ObjectiveDirection.min, ObjectiveDirection.minimize]:
            solver_obj = cvxpy.Minimize(expr)
        else:
            raise ConvOptError('Unsupported objective direction "{}"'.format(conv_opt_model.objective_direction))

        # constraints
        solver_cons = []
        for constraint in conv_opt_model.constraints:
            expr = None
            for term in constraint.terms:
                if isinstance(term, LinearTerm):
                    i_variable = conv_opt_model.variables.index(term.variable)
                    if expr is None:
                        expr = term.coefficient * solver_vars[i_variable]
                    else:
                        expr += term.coefficient * solver_vars[i_variable]
                else:
                    raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))

            if constraint.lower_bound is None and constraint.upper_bound is None:
                raise ConvOptError('Constraints must have at least one bound')
            elif constraint.lower_bound is None:
                solver_cons.append(expr <= constraint.upper_bound)
            elif constraint.upper_bound is None:
                solver_cons.append(expr >= constraint.lower_bound)
            elif constraint.lower_bound == constraint.upper_bound:
                solver_cons.append(expr == constraint.lower_bound)
            else:
                solver_cons.append(expr >= constraint.lower_bound)
                solver_cons.append(expr <= constraint.upper_bound)

        # variable bounds
        for variable, solver_var in zip(conv_opt_model.variables, solver_vars):
            if variable.lower_bound is None and variable.upper_bound is None:
                pass
            elif variable.lower_bound is None:
                solver_cons.append(solver_var <= variable.upper_bound)
            elif variable.upper_bound is None:
                solver_cons.append(solver_var >= variable.lower_bound)
            elif variable.lower_bound == variable.upper_bound:
                solver_cons.append(solver_var == variable.lower_bound)
            else:
                solver_cons.append(solver_var >= variable.lower_bound)
                solver_cons.append(solver_var <= variable.upper_bound)

        solver_model = cvxpy.Problem(solver_obj, solver_cons)
        solver_model._variable_names = [variable.name for variable in conv_opt_model.variables]
        solver_model._constraint_names = [constraint.name for constraint in conv_opt_model.constraints]
        return solver_model

    def solve(self):
        """ Solve the model

        Returns:
            :obj:`Result`: result
        """
        model = self._model

        # presolve
        if self._options.presolve != Presolve.off:
            raise ConvOptError('Unsupported presolve mode {}'.format(self._options.presolve))

        # tune
        if self._options.tune:
            raise ConvOptError('Unsupported tuning mode {}'.format(self._options.tune))

        # solve
        value = model.solve(solver=self.SOLVER, verbose=self._options.verbosity.value >= Verbosity.status.value)
        if model.status == cvxpy.OPTIMAL:
            status_code = StatusCode.optimal
        elif model.status == cvxpy.INFEASIBLE:
            status_code = StatusCode.infeasible
        else:
            status_code = StatusCode.other

        status_message = model.status

        if status_code == StatusCode.optimal:
            primals_dict = {v.name(): v.value for v in model.variables()}
            primals_list = []
            for name in model._variable_names:
                primals_list.append(primals_dict[name])
            primals = numpy.array(primals_list)
        else:
            value = numpy.nan
            primals = numpy.full((len(model.variables()),), numpy.nan)

        reduced_costs = numpy.full((len(model.variables()),), numpy.nan)
        duals = numpy.full((len(model._constraint_names),), numpy.nan)

        return Result(status_code, status_message, value, primals, reduced_costs, duals)

    def get_stats(self):
        """ Get diagnostic information about the model

        Returns:
            :obj:`dict`: diagnostic information about the model
        """
        model = self._model
        size_metrics = model.size_metrics
        solver_stats = model.solver_stats
        return {
            'num_scalar_variables': size_metrics.num_scalar_variables,
            'num_scalar_data': size_metrics.num_scalar_data,
            'num_scalar_eq_constr': size_metrics.num_scalar_eq_constr,
            'num_scalar_leq_constr': size_metrics.num_scalar_leq_constr,
            'max_data_dimension': size_metrics.max_data_dimension,
            'max_big_small_squared': size_metrics.max_big_small_squared,
            'is_dcp': model.is_dcp(),
            'is_qp': model.is_qp(),
            'solve_time': solver_stats.solve_time if solver_stats else numpy.nan,
            'setup_time': solver_stats.setup_time if solver_stats else numpy.nan,
            'num_iters': solver_stats.num_iters if solver_stats else numpy.nan,
        }
