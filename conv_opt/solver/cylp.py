""" CyLP module

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
    import cylp.cy
    import cylp.py.modeling
except ImportError:
    import warnings
    warnings.warn('CyLP is not installed', UserWarning)
import capturer
import numpy


class CylpModel(SolverModel):
    """ CyLP solver """

    def load(self, conv_opt_model):
        """ Load a model to CyLP's data structure

        Args:
            conv_opt_model (:obj:`Model`): model

        Returns:
            :obj:`cylp.cy.CyClpSimplex`: the model in CyLP's data structure

        Raises:
            :obj:`ConvOptError`: if the presolve mode is unsupported, a variable has an unsupported type,
                an objective has an unsupported term, a constraint
                has an unsupported term, a constraint is unbounded, or the model is not of a supported type
        """
        from cylp.py.modeling.CyLPModel import getCoinInfinity
        inf = getCoinInfinity()

        solver_model = cylp.cy.CyClpSimplex()

        # variables
        for i_variable, variable in enumerate(conv_opt_model.variables):
            if variable.type in [VariableType.binary, VariableType.integer]:
                is_int = True
            elif variable.type == VariableType.continuous:
                is_int = False
            else:
                raise ConvOptError('Unsupported variable of type "{}"'.format(variable.type))
            solver_variable = solver_model.addVariable(variable.name or '', 1, isInt=is_int)
            solver_model.setVariableName(i_variable, variable.name or '')

            lb = -inf
            ub = inf

            if variable.lower_bound is not None:
                lb = max(-inf, variable.lower_bound)

            if variable.upper_bound is not None:
                ub = min(inf, variable.upper_bound)

            if variable.type == VariableType.binary:
                lb = max(lb, 0)
                ub = min(ub, 1)

            solver_variable.lower = numpy.array([lb])
            solver_variable.upper = numpy.array([ub])

        # objective
        objective = numpy.zeros((len(conv_opt_model.variables), ))
        for term in conv_opt_model.objective_terms:
            if isinstance(term, LinearTerm):
                i_variable = conv_opt_model.variables.index(term.variable)
                objective[i_variable] += [term.coefficient]
            else:
                raise ConvOptError('Unsupported objective term of type "{}"'.format(term.__class__.__name__))
        solver_model.objective = objective

        if conv_opt_model.objective_direction in [ObjectiveDirection.max, ObjectiveDirection.maximize]:
            solver_model.optimizationDirection = 'max'
        elif conv_opt_model.objective_direction in [ObjectiveDirection.min, ObjectiveDirection.minimize]:
            solver_model.optimizationDirection = 'min'
        else:
            raise ConvOptError('Unsupported objective direction "{}"'.format(conv_opt_model.objective_direction))

        # constraints
        for i_constraint, constraint in enumerate(conv_opt_model.constraints):
            expr = None
            for term in constraint.terms:
                if isinstance(term, LinearTerm):
                    term_expr = numpy.matrix([term.coefficient]) * \
                        solver_model.variables[conv_opt_model.variables.index(term.variable)]
                    if expr is None:
                        expr = term_expr
                    else:
                        expr += term_expr
                else:
                    raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))

            if constraint.lower_bound is None and constraint.upper_bound is None:
                raise ConvOptError('Constraints must have at least one bound')
            elif constraint.lower_bound is None:
                ub = cylp.py.modeling.CyLPArray([constraint.upper_bound])
                solver_model += expr <= ub
            elif constraint.upper_bound is None:
                lb = cylp.py.modeling.CyLPArray([constraint.lower_bound])
                solver_model += expr >= lb
            elif constraint.lower_bound == constraint.upper_bound:
                lb = cylp.py.modeling.CyLPArray([constraint.lower_bound])
                solver_model += expr == lb
            else:
                lb = cylp.py.modeling.CyLPArray([constraint.lower_bound])
                ub = cylp.py.modeling.CyLPArray([constraint.upper_bound])
                solver_model += lb <= expr <= ub

            solver_model.setConstraintName(i_constraint, constraint.name or '')

        # return model
        return solver_model

    def solve(self):
        """ Solve the model

        Returns:
            :obj:`Result`: result
        """
        model = self._model

        # presolve
        if self._options.presolve == Presolve.on:
            presolve = 'on'
        elif self._options.presolve == Presolve.off:
            presolve = 'off'
        else:
            raise ConvOptError('Unsupported presolve mode "{}"'.format(self._options.presolve))

        # tune
        if self._options.tune:
            raise ConvOptError('Unsupported tuning mode {}'.format(self._options.tune))

        # solve
        if self._options.verbosity.value >= Verbosity.status.value:
            model.initialSolve(presolve=presolve)
        elif self._options.verbosity.value >= Verbosity.error.value:
            with capturer.CaptureOutput(merged=False, relay=False) as captured:
                model.initialSolve(presolve=presolve)
                stderr = captured.stderr.get_text()
            if stderr:
                sys.stderr.write(stderr)
        else:
            with capturer.CaptureOutput(relay=False):
                model.initialSolve(presolve=presolve)

        # get solution
        if model.getStatusCode() == 0:
            status_code = StatusCode.optimal
        elif model.getStatusCode() == 1:
            status_code = StatusCode.infeasible
        else:
            status_code = StatusCode.other
        status_message = model.getStatusString()

        if status_code == StatusCode.optimal:
            value = model.objectiveValue
            primals = numpy.array([model.primalVariableSolution[name][0] for name in model.variableNames])
            duals = numpy.array([model.dualConstraintSolution[c.name] for c in model.constraints])
        else:
            value = numpy.nan
            primals = numpy.full((len(model.variables), ), numpy.nan)
            duals = numpy.full((len(model.constraints), ), numpy.nan)

        reduced_costs = numpy.full((len(model.variables), ), numpy.nan)

        return Result(status_code, status_message, value, primals, reduced_costs, duals)

    def get_stats(self):
        """ Get diagnostic information about the model

        Returns:
            :obj:`dict`: diagnostic information about the model
        """
        model = self._model

        variable_statuses = []
        for variable in model.variables:
            variable_status = model.getVariableStatus(variable)
            variable_statuses.append(variable_status)

        constraint_statuses = []
        for constraint in model.constraints:
            constraint_status = model.getConstraintStatus(constraint.name)
            constraint_statuses.append(constraint_status)

        cbc_model = model.getCbcModel()

        return {
            'variable_statuses': variable_statuses,
            'constraint_statuses': constraint_statuses,
            'pivot_variable': model.getPivotVariable(),
            'pivot_constraint': model.pivotRow(),
            'complementarity_list': model.getComplementarityList(),
            'is_relaxation_abandoned': cbc_model.isRelaxationAbondoned(),
            'is_relaxation_dual_infeasible': cbc_model.isRelaxationDualInfeasible(),
            'is_relaxation_infeasible': cbc_model.isRelaxationInfeasible(),
            'is_relaxation_optimal': cbc_model.isRelaxationOptimal(),
        }
