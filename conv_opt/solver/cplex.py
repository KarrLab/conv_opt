""" IBM CPLEX module

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
try:
    import cplex
except ImportError:
    import warnings
    warnings.warn('IBM CPLEX is not installed', UserWarning)
import numpy
import sys


class CplexModel(SolverModel):
    """ IBM CPLEX solver """

    def load(self, conv_opt_model):
        """ Load a model to CPLEX's data structure

        Args:
            conv_opt_model (:obj:`Model`): model

        Returns:
            :obj:`cplex.Cplex`: the model in CPLEX's data structure

        Raises:
            :obj:`ConvOptError`: if the presolve mode is not supported a variable has an unsupported
                type, an objective has an unsupported term, a
                constraint has an unsupported term, a constraint is unbounded, or the model is not of a
                supported type
        """

        solver_model = cplex.Cplex()
        solver_model.set_problem_name(conv_opt_model.name)

        # create variables and set bounds
        solver_types = solver_model.variables.type

        names = []
        types = []
        lb = []
        ub = []

        for variable in conv_opt_model.variables:
            names.append(variable.name)

            if variable.type == VariableType.binary:
                types.append(solver_types.binary)
            elif variable.type == VariableType.integer:
                types.append(solver_types.integer)
            elif variable.type == VariableType.continuous:
                types.append(solver_types.continuous)
            elif variable.type == VariableType.semi_integer:
                types.append(solver_types.semi_integer)
            elif variable.type == VariableType.semi_continuous:
                types.append(solver_types.semi_continuous)
            else:
                raise ConvOptError('Unsupported variable of type "{}"'.format(variable.type))

            if variable.lower_bound is not None:
                lb.append(variable.lower_bound)
            else:
                lb.append(-1 * cplex.infinity)

            if variable.upper_bound is not None:
                ub.append(variable.upper_bound)
            else:
                ub.append(cplex.infinity)
        solver_model.variables.add(names=names, types=types, lb=lb, ub=ub)

        # set objective
        if conv_opt_model.objective_direction in [ObjectiveDirection.max, ObjectiveDirection.maximize]:
            solver_model.objective.set_sense(solver_model.objective.sense.maximize)
        elif conv_opt_model.objective_direction in [ObjectiveDirection.min, ObjectiveDirection.minimize]:
            solver_model.objective.set_sense(solver_model.objective.sense.minimize)
        else:
            raise ConvOptError('Unsupported objective direction "{}"'.format(conv_opt_model.objective_direction))

        for term in conv_opt_model.objective_terms:
            if isinstance(term, LinearTerm):
                i = conv_opt_model.variables.index(term.variable)
                solver_model.objective.set_linear(i, solver_model.objective.get_linear(i) + term.coefficient)
            elif isinstance(term, QuadraticTerm):
                i_1 = conv_opt_model.variables.index(term.variable_1)
                i_2 = conv_opt_model.variables.index(term.variable_2)
                if i_1 == i_2:
                    coefficient = 2. * term.coefficient
                else:
                    coefficient = 1. * term.coefficient
                solver_model.objective.set_quadratic_coefficients(
                    i_1, i_2, solver_model.objective.get_quadratic_coefficients(i_1, i_2) + coefficient)
            else:
                raise ConvOptError('Unsupported objective term of type "{}"'.format(term.__class__.__name__))

        # set constraints
        names = []
        lin_expr = []
        senses = []
        rhs = []
        range_values = []
        for constraint in conv_opt_model.constraints:
            if constraint.lower_bound is None and constraint.upper_bound is None:
                raise ConvOptError('Constraints must have at least one bound')
            if constraint.lower_bound is None:
                senses.append('L')
                rhs.append(constraint.upper_bound)
                range_values.append(0.)
            elif constraint.upper_bound is None:
                senses.append('G')
                rhs.append(constraint.lower_bound)
                range_values.append(0.)
            elif constraint.lower_bound == constraint.upper_bound:
                senses.append('E')
                rhs.append(constraint.lower_bound)
                range_values.append(0.)
            else:
                senses.append('R')
                rhs.append(constraint.lower_bound)
                range_values.append(constraint.upper_bound - constraint.lower_bound)

            names.append(constraint.name)

            ind = []
            val = []
            for term in constraint.terms:
                if isinstance(term, LinearTerm):
                    ind.append(conv_opt_model.variables.index(term.variable))
                    val.append(term.coefficient)
                # elif isinstance(term, QuadraticTerm):
                    # :todo: implement quadratic constraints
                    # raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))
                else:
                    raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))
            lin_expr.append(cplex.SparsePair(ind=ind, val=val))

        solver_model.linear_constraints.add(names=names, lin_expr=lin_expr, senses=senses, rhs=rhs, range_values=range_values)

        # set model type, :todo: support other problem types
        problem_type = conv_opt_model.get_type()
        if problem_type == ModelType.lp:
            solver_model.set_problem_type(solver_model.problem_type.LP)
        elif problem_type == ModelType.qp:
            solver_model.set_problem_type(solver_model.problem_type.QP)
        elif problem_type == ModelType.milp:
            solver_model.set_problem_type(solver_model.problem_type.MILP)
        elif problem_type == ModelType.miqp:
            solver_model.set_problem_type(solver_model.problem_type.MIQP)
        # else: # condition not needed because of the above error checking
        #    raise ConvOptError('Unsupported model type "{}"'.format(problem_type))

        return solver_model

    def solve(self):
        """ Solve the model

        Returns:
            :obj:`Result`: result
        """

        model = self._model

        # set verbosity
        if self._options.verbosity.value < Verbosity.status.value:
            model.set_results_stream(None)
        else:
            model.set_results_stream(sys.stdout)

        if self._options.verbosity.value < Verbosity.warning.value:
            model.set_warning_stream(None)
        else:
            model.set_results_stream(sys.stdout)

        if self._options.verbosity.value < Verbosity.error.value:
            model.set_error_stream(None)
        else:
            model.set_results_stream(sys.stdout)

        # set presolve mode
        if self._options.presolve == Presolve.on:
            model.parameters.preprocessing.presolve.set(model.parameters.preprocessing.presolve.values.on)
        elif self._options.presolve == Presolve.off:
            model.parameters.preprocessing.presolve.set(model.parameters.preprocessing.presolve.values.off)
        else:
            raise ConvOptError('Unsupported presolve mode "{}"'.format(self._options.presolve))

        # tune
        if self._options.tune:
            model.parameters.tune_problem()

        model.solve()
        sol = model.solution

        tmp = sol.get_status()
        if tmp in [1, 101]:
            status_code = StatusCode.optimal
        elif tmp in [3, 103]:
            status_code = StatusCode.infeasible
        else:
            status_code = StatusCode.other

        status_message = sol.get_status_string()

        if status_code == StatusCode.optimal:
            value = sol.get_objective_value()
            primals = numpy.array(sol.get_values())
            if model.get_problem_type() in [model.problem_type.LP, model.problem_type.QP]:
                reduced_costs = numpy.array(sol.get_reduced_costs())
                duals = numpy.array(sol.get_dual_values())
            else:
                reduced_costs = numpy.full((model.variables.get_num(),), numpy.nan)
                duals = numpy.full((model.linear_constraints.get_num() + model.quadratic_constraints.get_num(),), numpy.nan)
        else:
            value = numpy.nan
            primals = numpy.full((model.variables.get_num(),), numpy.nan)
            reduced_costs = numpy.full((model.variables.get_num(),), numpy.nan)
            duals = numpy.full((model.linear_constraints.get_num() + model.quadratic_constraints.get_num(),), numpy.nan)

        return Result(status_code, status_message, value, primals, reduced_costs, duals)

    def get_stats(self):
        """ Get diagnostic information about the model

        Returns:
            :obj:`str`: diagnostic information about the model
        """
        model = self._model
        return model.get_stats()
