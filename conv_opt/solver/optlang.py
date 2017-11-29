""" optlang module

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
import numpy
import optlang
import sympy


class OptlangModel(SolverModel):
    """ optlang solver

    Attributes:
        INTERFACE (:obj:`module`): optlang interface
    """
    INTERFACE = None

    def load(self, conv_opt_model):
        """ Load a model to GPLK's data structure

        Args:
            conv_opt_model (:obj:`Model`): model

        Returns:
            :obj:`optlang.Model`: the model in optlang's data structure

        Raises:
            :obj:`ConvOptError`: if the presolve mode is unsupported, a variable has an unsupported type,
                an objective has an unsupported term, a constraint
                has an unsupported term, a constraint is unbounded, or the model is not of a supported type
        """
        solver_model = self.INTERFACE.Model(name=conv_opt_model.name)

        # variables
        solver_variables = []
        for variable in conv_opt_model.variables:
            if variable.type == VariableType.binary:
                type = 'binary'
            elif variable.type == VariableType.integer:
                type = 'integer'
            elif variable.type == VariableType.continuous:
                type = 'continuous'
            else:
                raise ConvOptError('Unsupported variable of type "{}"'.format(variable.type))
            solver_variable = self.INTERFACE.Variable(
                name=variable.name, lb=variable.lower_bound, ub=variable.upper_bound, type=type)
            solver_model.add(solver_variable)
            solver_variables.append(solver_variable)

        # objective
        if conv_opt_model.objective_direction in [ObjectiveDirection.max, ObjectiveDirection.maximize]:
            direction = 'max'
        elif conv_opt_model.objective_direction in [ObjectiveDirection.min, ObjectiveDirection.minimize]:
            direction = 'min'
        else:
            raise ConvOptError('Unsupported objective direction "{}"'.format(conv_opt_model.objective_direction))

        expr = sympy.Integer(0)
        for term in conv_opt_model.objective_terms:
            if isinstance(term, LinearTerm):
                expr += sympy.Float(term.coefficient) * solver_variables[conv_opt_model.variables.index(term.variable)]
            elif isinstance(term, QuadraticTerm):
                expr += sympy.Float(term.coefficient) \
                    * solver_variables[conv_opt_model.variables.index(term.variable_1)] \
                    * solver_variables[conv_opt_model.variables.index(term.variable_2)]
            else:
                raise ConvOptError('Unsupported objective term of type "{}"'.format(term.__class__.__name__))
        solver_model.objective = self.INTERFACE.Objective(expr, direction=direction)

        # constraints
        for constraint in conv_opt_model.constraints:
            expr = sympy.Integer(0)
            for term in constraint.terms:
                if isinstance(term, LinearTerm):
                    expr += sympy.Float(term.coefficient) * solver_variables[conv_opt_model.variables.index(term.variable)]
                else:
                    raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))
            solver_model.add(self.INTERFACE.Constraint(expr, lb=constraint.lower_bound, ub=constraint.upper_bound, name=constraint.name))

        # return model
        return solver_model

    def solve(self):
        """ Solve the model

        Returns:
            :obj:`Result`: result
        """
        model = self._model

        # verbosity
        model.configuration.verbosity = self._options.verbosity.value

        # presolve
        if self._options.presolve == Presolve.auto:
            model.configuration.presolve = 'auto'
        elif self._options.presolve == Presolve.on:
            model.configuration.presolve = True
        elif self._options.presolve == Presolve.off:
            model.configuration.presolve = False
        else:
            raise ConvOptError('Unsupported presolve mode "{}"'.format(self._options.presolve))

        # optimize
        model.optimize()

        if model.status == 'optimal':
            status_code = StatusCode.optimal
        elif model.status == 'infeasible':
            status_code = StatusCode.infeasible
        else:
            status_code = StatusCode.other
        status_message = model.status

        has_int = False
        for var in model.variables:
            if var.type != 'continuous':
                has_int = True
                break

        if status_code == StatusCode.optimal:
            value = model.objective.value
            primals = numpy.array(list(model.primal_values.values()))
            if has_int:
                reduced_costs = numpy.full((len(model.variables), ), numpy.nan)
                duals = numpy.full((len(model.constraints), ), numpy.nan)
            else:
                reduced_costs = numpy.array(list(model.reduced_costs.values()))
                duals = numpy.array(list(model.shadow_prices.values()))
        else:
            value = numpy.nan
            primals = numpy.full((len(model.variables), ), numpy.nan)
            reduced_costs = numpy.full((len(model.variables), ), numpy.nan)
            duals = numpy.full((len(model.constraints), ), numpy.nan)

        return Result(status_code, status_message, value, primals, reduced_costs, duals)
