""" Gurobi module

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
import capturer
try:
    import gurobipy
except ImportError:
    import warnings
    warnings.warn('Gurobi is not installed', UserWarning)
import numpy


class GurobiModel(SolverModel):
    """ Gurobi solver """

    def load(self, conv_opt_model):
        """ Load a model to Gurobi's data structure

        Args:
            conv_opt_model (:obj:`gurobipy.Model`): model

        Returns:
            :obj:`object`: the model in Gurobi's data structure

        Raises:
            :obj:`ConvOptError`: if the presolve mode is unsupported, a variable has an unsupported type,
                an objective has an unsupported term, a constraint has
                an unsupported term, or a constraint is unbounded
        """
        with capturer.CaptureOutput(relay=False):
            solver_model = gurobipy.Model(conv_opt_model.name or '')

        # variables
        solver_variables = []
        for variable in conv_opt_model.variables:
            if variable.type == VariableType.binary:
                type = gurobipy.GRB.BINARY
            elif variable.type == VariableType.integer:
                type = gurobipy.GRB.INTEGER
            elif variable.type == VariableType.continuous:
                type = gurobipy.GRB.CONTINUOUS
            elif variable.type == VariableType.semi_integer:
                type = gurobipy.GRB.SEMIINT
            elif variable.type == VariableType.semi_continuous:
                type = gurobipy.GRB.SEMICONT
            else:
                raise ConvOptError('Unsupported variable of type "{}"'.format(variable.type))

            if variable.lower_bound is None:
                lb = -gurobipy.GRB.INFINITY
            else:
                lb = variable.lower_bound

            if variable.upper_bound is None:
                ub = gurobipy.GRB.INFINITY
            else:
                ub = variable.upper_bound

            var = solver_model.addVar(lb=lb, ub=ub, vtype=type, name=variable.name or '')
            solver_variables.append(var)

        # objective
        if conv_opt_model.objective_direction in [ObjectiveDirection.max, ObjectiveDirection.maximize]:
            sense = gurobipy.GRB.MAXIMIZE
        elif conv_opt_model.objective_direction in [ObjectiveDirection.min, ObjectiveDirection.minimize]:
            sense = gurobipy.GRB.MINIMIZE
        else:
            raise ConvOptError('Unsupported objective direction "{}"'.format(conv_opt_model.objective_direction))

        lin_expr = gurobipy.LinExpr()
        quad_expr = gurobipy.QuadExpr()
        is_quad = False
        for term in conv_opt_model.objective_terms:
            if isinstance(term, LinearTerm):
                lin_expr.addTerms(term.coefficient,
                                  solver_variables[conv_opt_model.variables.index(term.variable)])
            elif isinstance(term, QuadraticTerm):
                is_quad = True
                quad_expr.addTerms(term.coefficient,
                                   solver_variables[conv_opt_model.variables.index(term.variable_1)],
                                   solver_variables[conv_opt_model.variables.index(term.variable_2)])
            else:
                raise ConvOptError('Unsupported objective term of type "{}"'.format(term.__class__.__name__))
        if is_quad:
            quad_expr.add(lin_expr)
            expr = quad_expr
        else:
            expr = lin_expr

        solver_model.setObjective(expr, sense)

        # constraints
        for constraint in conv_opt_model.constraints:
            lhs = gurobipy.LinExpr()
            for term in constraint.terms:
                if isinstance(term, LinearTerm):
                    lhs.addTerms(term.coefficient, solver_variables[conv_opt_model.variables.index(term.variable)])
                # elif isinstance(term, QuadraticTerm):
                    # :todo: implement quadratic constraints
                    # raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))
                else:
                    raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))

            if constraint.lower_bound is None and constraint.upper_bound is None:
                raise ConvOptError('Constraints must have at least one bound')
            elif constraint.lower_bound is None:
                solver_model.addConstr(lhs, gurobipy.GRB.LESS_EQUAL, constraint.upper_bound, name=constraint.name or '')
            elif constraint.upper_bound is None:
                solver_model.addConstr(lhs, gurobipy.GRB.GREATER_EQUAL, constraint.lower_bound, name=constraint.name or '')
            elif constraint.lower_bound == constraint.upper_bound:
                solver_model.addConstr(lhs, gurobipy.GRB.EQUAL, constraint.lower_bound, name=constraint.name or '')
            else:
                solver_model.addConstr(lhs, gurobipy.GRB.GREATER_EQUAL, constraint.lower_bound, name=(constraint.name or '') + '__lower__')
                solver_model.addConstr(lhs, gurobipy.GRB.LESS_EQUAL, constraint.upper_bound, name=(constraint.name or '') + '__upper__')

        # synchronize updates
        solver_model.update()

        return solver_model

    def solve(self):
        """ Solve the model

        Returns:
            :obj:`Result`: result
        """

        model = self._model

        # set verbosity
        if self._options.verbosity.value >= Verbosity.status.value:
            model.setParam('LogToConsole', 1)
        else:
            model.setParam('LogToConsole', 0)

        # tune
        if self._options.tune:
            if self._options.verbosity.value >= Verbosity.status.value:
                model.tune()
            else:
                with capturer.CaptureOutput(relay=False):
                    model.tune()

        # set presolve
        if self._options.presolve in [Presolve.auto, Presolve.on]:
            model.setParam(gurobipy.GRB.Param.Presolve, 1)
        elif self._options.presolve == Presolve.off:
            model.setParam(gurobipy.GRB.Param.Presolve, 0)
        else:
            raise ConvOptError('Unsupported presolve mode "{}"'.format(self._options.presolve))

        model.update()
        model.optimize()

        if model.status == gurobipy.GRB.Status.OPTIMAL:
            status_code = StatusCode.optimal
        elif model.status == gurobipy.GRB.Status.INFEASIBLE:
            status_code = StatusCode.infeasible
        else:
            status_code = StatusCode.other
        status_message = ''

        if status_code == StatusCode.optimal:
            value = model.objVal

            primals = []
            for variable in model.getVars():
                primals.append(variable.getAttr('X'))
            primals = numpy.array(primals)

            if model.getAttr(gurobipy.GRB.Attr.IsMIP):
                reduced_costs = numpy.full((len(model.getVars()), ), numpy.nan)

                duals = []
                for constraint in model.getConstrs():
                    if not constraint.getAttr(gurobipy.GRB.Attr.ConstrName).endswith('__upper__'):
                        duals.append(numpy.nan)
                duals = numpy.array(duals)
            else:
                reduced_costs = []
                for variable in model.getVars():
                    reduced_costs.append(variable.getAttr('RC'))
                reduced_costs = numpy.array(reduced_costs)

                duals = []
                for constraint in model.getConstrs():
                    if not constraint.getAttr('ConstrName').endswith('__upper__'):
                        duals.append(constraint.getAttr('Pi'))
                duals = numpy.array(duals)
        else:
            value = numpy.nan
            primals = numpy.full((len(model.getVars()), ), numpy.nan)
            reduced_costs = numpy.full((len(model.getVars()), ), numpy.nan)

            duals = []
            for constraint in model.getConstrs():
                if not constraint.getAttr(gurobipy.GRB.Attr.ConstrName).endswith('__upper__'):
                    duals.append(numpy.nan)
            duals = numpy.array(duals)

        return Result(status_code, status_message, value, primals, reduced_costs, duals)

    def get_stats(self):
        """ Get diagnostic information about the model

        Returns:
            :obj:`dict`: diagnostic information about the model
        """
        model = self._model

        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.printStats()
            stats = captured.stdout.get_text()

        return {
            'stats': stats,
            'sos_constraints': model.getSOSs(),
            'bar_iter_limit': model.getParamInfo('BarIterLimit'),
            'objective_cutoff': model.getParamInfo('Cutoff'),
            'simplex_interation_limit':  model.getParamInfo('IterationLimit'),
            'mip_node_limit': model.getParamInfo('NodeLimit'),
            'mip_feasible_solution_limit': model.getParamInfo('SolutionLimit'),
            'time_limit': model.getParamInfo('TimeLimit'),
            'best_objective_stop': model.getParamInfo('BestObjStop'),
            'best_objective_bound_stop': model.getParamInfo('BestBdStop'),
        }
