""" FICO XPRESS module

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
import numpy
import sys
try:
    import xpress
except ImportError:
    import warnings
    warnings.warn('FICO XPRESS is not installed', UserWarning)


class XpressModel(SolverModel):
    """ FICO XPRESS solver """

    def load(self, conv_opt_model):
        """ Load a model to XPRESS' data structure

        Args:
            conv_opt_model (:obj:`Model`): model

        Returns:
            :obj:`object`: the model in XPRESS' data structure

        Raises:
            :obj:`ConvOptError`: if the presolve mode is unsupported, a variable has an unsupported type,
                an objective has an unsupported term, a constraint
                has an unsupported term, or a constraint is unbounded
        """

        solver_model = xpress.problem(name=conv_opt_model.name)

        # variables
        for variable in conv_opt_model.variables:
            if variable.type == VariableType.binary:
                vartype = xpress.binary
            elif variable.type == VariableType.integer:
                vartype = xpress.integer
            elif variable.type == VariableType.continuous:
                vartype = xpress.continuous
            elif variable.type == VariableType.semi_integer:
                vartype = xpress.semiinteger
            elif variable.type == VariableType.semi_continuous:
                vartype = xpress.semicontinuous
            elif variable.type == VariableType.partially_integer:
                vartype = xpress.partiallyinteger
            else:
                raise ConvOptError('Unsupported variable of type "{}"'.format(variable.type))

            if variable.lower_bound is None:
                lb = -float('inf')
            else:
                lb = variable.lower_bound

            if variable.upper_bound is None:
                ub = float('inf')
            else:
                ub = variable.upper_bound

            solver_model.addVariable(xpress.var(name=variable.name, lb=lb, ub=ub, vartype=vartype))

        # objective
        if conv_opt_model.objective_direction in [ObjectiveDirection.max, ObjectiveDirection.maximize]:
            sense = xpress.maximize
        elif conv_opt_model.objective_direction in [ObjectiveDirection.min, ObjectiveDirection.minimize]:
            sense = xpress.minimize
        else:
            raise ConvOptError('Unsupported objective direction "{}"'.format(conv_opt_model.objective_direction))

        terms = []
        for term in conv_opt_model.objective_terms:
            if isinstance(term, LinearTerm):
                i_variable = conv_opt_model.variables.index(term.variable)
                terms.append(term.coefficient * solver_model.getVariable(index=i_variable))
            elif isinstance(term, QuadraticTerm):
                i_variable_1 = conv_opt_model.variables.index(term.variable_1)
                i_variable_2 = conv_opt_model.variables.index(term.variable_2)
                terms.append(term.coefficient
                             * solver_model.getVariable(index=i_variable_1)
                             * solver_model.getVariable(index=i_variable_2))
            else:
                raise ConvOptError('Unsupported objective term of type "{}"'.format(term.__class__.__name__))

        solver_model.setObjective(xpress.Sum(terms), sense)

        # constraints
        for constraint in conv_opt_model.constraints:
            body = []
            for term in constraint.terms:
                if isinstance(term, LinearTerm):
                    i_variable = conv_opt_model.variables.index(term.variable)
                    body.append(term.coefficient * solver_model.getVariable(index=i_variable))
                # elif isinstance(term, QuadraticTerm):
                    # :todo: implement quadratic constraints
                    # raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))
                else:
                    raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))

            if constraint.lower_bound is None and constraint.upper_bound is None:
                raise ConvOptError('Constraints must have at least one bound')
            elif constraint.lower_bound is None:
                solver_model.addConstraint(xpress.constraint(body=xpress.Sum(
                    body), ub=constraint.upper_bound, sense=xpress.leq, name=constraint.name))
            elif constraint.upper_bound is None:
                solver_model.addConstraint(xpress.constraint(body=xpress.Sum(
                    body), lb=constraint.lower_bound, sense=xpress.geq, name=constraint.name))
            elif constraint.lower_bound == constraint.upper_bound:
                solver_model.addConstraint(xpress.constraint(body=xpress.Sum(
                    body), rhs=constraint.lower_bound, sense=xpress.eq, name=constraint.name))
            else:
                # :todo: figure out how to use xpress.range type
                solver_model.addConstraint(xpress.constraint(body=xpress.Sum(body), sense=xpress.geq,
                                                             rhs=constraint.lower_bound, name=constraint.name + '__lower__'))
                solver_model.addConstraint(xpress.constraint(body=xpress.Sum(body), sense=xpress.leq,
                                                             rhs=constraint.upper_bound, name=constraint.name + '__upper__'))

        return solver_model

    def solve(self):
        """ Solve the model

        Returns:
            :obj:`Result`: result
        """

        model = self._model

        # verbosity
        if self._options.verbosity == Verbosity.off:
            model.setlogfile('xpress.log')

        # presolve
        if self._options.presolve == Presolve.on:
            if self._options.verbosity == Verbosity.off:
                with capturer.CaptureOutput(merged=False, relay=False) as captured:
                    model.presolve()
            elif self._options.verbosity == Verbosity.error:
                with capturer.CaptureOutput(merged=False, relay=False) as captured:
                    model.presolve()
                    err = captured.stderr.get_text()
                    if err:
                        sys.stderr.write(err)
            else:
                model.presolve()
        elif self._options.presolve != Presolve.off:
            raise ConvOptError('Unsupported presolve mode "{}"'.format(self._options.presolve))

        # tune
        if self._options.tune:
            raise ConvOptError('Unsupported tuning mode "{}"'.format(self._options.tune))

        # solve
        model.solve()

        # get status
        if self.is_mixed_integer():
            if model.getProbStatus() == xpress.mip_optimal:
                status_code = StatusCode.optimal
            elif model.getProbStatus() == xpress.mip_infeas:
                status_code = StatusCode.infeasible
            else:
                status_code = StatusCode.other
        elif self.is_non_linear():
            if model.getProbStatus() in [xpress.nlp_locally_optimal]:
                status_code = StatusCode.optimal
            elif model.getProbStatus() in [xpress.lp_infeas, xpress.nlp_infeasible]:
                status_code = StatusCode.infeasible
            else:
                status_code = StatusCode.other
        else:
            if model.getProbStatus() == xpress.lp_optimal:
                status_code = StatusCode.optimal
            elif model.getProbStatus() == xpress.lp_infeas:
                status_code = StatusCode.infeasible
            else:
                status_code = StatusCode.other

        status_message = model.getProbStatusString()

        # get solution
        if status_code == StatusCode.optimal:
            value = model.getObjVal()
            primals = numpy.array([model.getSolution(var) for var in model.getVariable()])
            if self.is_mixed_integer():
                reduced_costs = numpy.full((len(model.getVariable()), ), numpy.nan)
                duals = numpy.array([numpy.nan
                                     for constraint in model.getConstraint() if not constraint.name.endswith('__upper__')])
            else:
                reduced_costs = numpy.array([model.getRCost(var) for var in model.getVariable()])
                duals = numpy.array([model.getDual(constraint)
                                     for constraint in model.getConstraint() if not constraint.name.endswith('__upper__')])
        else:
            value = numpy.nan
            primals = numpy.full((len(model.getVariable()), ), numpy.nan)
            reduced_costs = numpy.full((len(model.getVariable()), ), numpy.nan)
            duals = numpy.array([numpy.nan
                                 for constraint in model.getConstraint() if not constraint.name.endswith('__upper__')])

        return Result(status_code, status_message, value, primals, reduced_costs, duals)

    def get_stats(self):
        """ Get diagnostic information about the model

        Returns:
            :obj:`dict`: diagnostic information about the model
        """
        model = self._model

        sol = model.getSolution()

        stats = {}

        stats['slacks'] = [numpy.nan] * len(model.getConstraint())
        model.calcslacks(sol, stats['slacks'])

        if not self.is_non_linear():
            stats['row_status'] = [0] * len(model.getConstraint())
            stats['col_status'] = [0] * len(model.getVariable())
            model.getbasis(stats['row_status'], stats['col_status'])

        stats['primal_infeasible_variables'] = []
        stats['primal_infeasible_constraints'] = []
        stats['dual_infeasible_constraints'] = []
        stats['dual_infeasible_variables'] = []

        if not self.is_mixed_integer():
            model.getinfeas(
                stats['primal_infeasible_variables'], stats['primal_infeasible_constraints'],
                stats['dual_infeasible_constraints'], stats['dual_infeasible_variables'])

            duals = model.getDual()
            stats['abs_primal_infeas'] = model.calcsolinfo(sol, duals, xpress.solinfo_absprimalinfeas)
            stats['rel_primal_infeas'] = model.calcsolinfo(sol, duals, xpress.solinfo_relprimalinfeas)
            stats['abs_dual_infeas'] = model.calcsolinfo(sol, duals, xpress.solinfo_absdualinfeas)
            stats['rel_dual_infeas'] = model.calcsolinfo(sol, duals, xpress.solinfo_reldualinfeas)
            stats['max_mip_fractional'] = model.calcsolinfo(sol, duals, xpress.solinfo_maxmipfractional)

        stats['sos'] = model.getSOS()

        return stats

    def is_mixed_integer(self):
        """ Determine if the model has at least one binary or integer variable

        Returns:
            :obj:`bool`: :obj:`True` if the model has at least one binary or integer variable, and false otherwise
        """
        for variable in self._model.getVariable():
            if variable.vartype in [xpress.binary, xpress.integer, xpress.semiinteger, xpress.partiallyinteger]:
                return True
        return False

    def is_non_linear(self):
        """ Determine if the model is non-linear

        Returns:
            :obj:`bool`: :obj:`True` if the model is non-linear and false otherwise
        """
        n_var = len(self._model.getVariable())
        size = int((n_var + 1) * n_var / 2)
        mstart = [0.] * size
        mclind = [0.] * size
        dobjval = [0.] * size
        self._model.getmqobj(mstart, mclind, dobjval, size, 0, n_var - 1)
        return len(dobjval) != 0
