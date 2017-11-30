""" MOSEK Optimizer module

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
    import mosek
except ImportError:
    import warnings
    warnings.warn('MOSEK is not installed', UserWarning)
import numpy


class MosekModel(SolverModel):
    """ MOSEK Optimizer solver

    Attributes
        _environment (:obj:`mosek.Env`): MOSEK Optimizer environment
    """

    def __init__(self, model, options=None):
        """
        Args:
            model (:obj:`Model`): model
            options (:obj:`SolveOptions`, optional): options
        """
        self._environment = mosek.Env()
        super(MosekModel, self).__init__(model, options=options)

    def load(self, conv_opt_model):
        """ Load a model to MOSEK Optimizer's data structure

        Args:
            conv_opt_model (:obj:`Model`): model

        Returns:
            :obj:`mosek.Task`: the model in MOSEK Optimizer's data structure

        Raises:
            :obj:`ConvOptError`: if the presolve mode is unsupported, a variable has an unsupported type,
                an objective has an unsupported term, a constraint
                has an unsupported term, or the model is not of a supported type
        """

        task = self._environment.Task()

        # variables
        task.appendvars(len(conv_opt_model.variables))
        for i_variable, variable in enumerate(conv_opt_model.variables):
            if variable.lower_bound is None and variable.upper_bound is None:
                boundkey = mosek.boundkey.fr
                lb = -float('inf')
                ub = float('inf')
            elif variable.lower_bound is None:
                boundkey = mosek.boundkey.up
                lb = -float('inf')
                ub = variable.upper_bound
            elif variable.upper_bound is None:
                boundkey = mosek.boundkey.lo
                lb = variable.lower_bound
                ub = float('inf')
            elif variable.lower_bound == variable.upper_bound:
                boundkey = mosek.boundkey.fx
                lb = variable.lower_bound
                ub = variable.upper_bound
            else:
                boundkey = mosek.boundkey.ra
                lb = variable.lower_bound
                ub = variable.upper_bound

            if variable.type == VariableType.binary:
                task.putvartype(i_variable, mosek.variabletype.type_int)
                lb = max(0, lb)
                ub = min(1, ub)
            elif variable.type == VariableType.integer:
                task.putvartype(i_variable, mosek.variabletype.type_int)
            elif variable.type == VariableType.continuous:
                task.putvartype(i_variable, mosek.variabletype.type_cont)
            else:
                raise ConvOptError('Unsupported variable of type "{}"'.format(variable.type))

            task.putvarbound(i_variable, boundkey, lb, ub)

        # objective
        if conv_opt_model.objective_direction in [ObjectiveDirection.max, ObjectiveDirection.maximize]:
            task.putobjsense(mosek.objsense.maximize)
        elif conv_opt_model.objective_direction in [ObjectiveDirection.min, ObjectiveDirection.minimize]:
            task.putobjsense(mosek.objsense.minimize)
        else:
            raise ConvOptError('Unsupported objective direction "{}"'.format(conv_opt_model.objective_direction))

        for term in conv_opt_model.objective_terms:
            if isinstance(term, LinearTerm):
                i_variable = conv_opt_model.variables.index(term.variable)
                task.putcj(i_variable, task.getcj(i_variable) + term.coefficient)
            elif isinstance(term, QuadraticTerm):
                i_variable_1 = conv_opt_model.variables.index(term.variable_1)
                i_variable_2 = conv_opt_model.variables.index(term.variable_2)
                if i_variable_1 == i_variable_2:
                    mult = 2.
                    i_min_variable = i_variable_1
                    i_max_variable = i_variable_1
                else:
                    mult = 1.
                    i_min_variable = max(i_variable_1, i_variable_2)
                    i_max_variable = min(i_variable_1, i_variable_2)

                task.putqobjij(i_min_variable, i_max_variable,
                               task.getqobjij(i_min_variable, i_max_variable) + mult * term.coefficient)
            else:
                raise ConvOptError('Unsupported objective term of type "{}"'.format(term.__class__.__name__))

        # constraints
        task.appendcons(len(conv_opt_model.constraints))
        for i_constraint, constraint in enumerate(conv_opt_model.constraints):
            for term in constraint.terms:
                if isinstance(term, LinearTerm):
                    i_variable = conv_opt_model.variables.index(term.variable)
                    task.putaij(i_constraint, i_variable, task.getaij(i_constraint, i_variable) + term.coefficient)
                # elif isinstance(term, QuadraticTerm):
                    # :todo: implement quadratic constraints
                    # raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))
                else:
                    raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))

            if constraint.lower_bound is None and constraint.upper_bound is None:
                boundkey = mosek.boundkey.fr
                lb = -float('inf')
                ub = float('inf')
            elif constraint.lower_bound is None:
                boundkey = mosek.boundkey.up
                lb = -float('inf')
                ub = constraint.upper_bound
            elif constraint.upper_bound is None:
                boundkey = mosek.boundkey.lo
                lb = constraint.lower_bound
                ub = float('inf')
            elif constraint.lower_bound == constraint.upper_bound:
                boundkey = mosek.boundkey.fx
                lb = constraint.lower_bound
                ub = constraint.upper_bound
            else:
                boundkey = mosek.boundkey.ra
                lb = constraint.lower_bound
                ub = constraint.upper_bound

            task.putconbound(i_constraint, boundkey, lb, ub)

        return task

    def solve(self):
        """ Solve the model

        Returns:
            :obj:`Result`: result
        """

        model = self._model

        if self._options.presolve != Presolve.off:
            raise ConvOptError('Unsupported presolve mode "{}"'.format(self._options.presolve))

        model.optimize()

        is_quadratic = model.getprobtype() == mosek.problemtype.qo

        has_int = False
        for i_variable in range(model.getnumvar()):
            if model.getvartype(i_variable) == mosek.variabletype.type_int:
                has_int = True

        if has_int:
            if model.getsolsta(mosek.soltype.itg) in [mosek.solsta.integer_optimal, mosek.solsta.near_integer_optimal]:
                status_code = StatusCode.optimal
            # I don't think this is reachable because Mosek seems to reurn solsta 'unknown'
            # elif model.getsolsta(mosek.soltype.itg) in [mosek.solsta.prim_infeas_cer, mosek.solsta.near_prim_infeas_cer]:
            #    status_code = StatusCode.infeasible
            else:
                status_code = StatusCode.other
        elif is_quadratic:
            if model.getsolsta(mosek.soltype.itr) in [mosek.solsta.optimal, mosek.solsta.near_optimal]:
                status_code = StatusCode.optimal
            elif model.getsolsta(mosek.soltype.itr) in [mosek.solsta.prim_infeas_cer, mosek.solsta.near_prim_infeas_cer]:
                status_code = StatusCode.infeasible
            else:
                status_code = StatusCode.other
        else:
            if model.getsolsta(mosek.soltype.bas) in [mosek.solsta.optimal, mosek.solsta.near_optimal]:
                status_code = StatusCode.optimal
            elif model.getsolsta(mosek.soltype.bas) in [mosek.solsta.prim_infeas_cer, mosek.solsta.near_prim_infeas_cer]:
                status_code = StatusCode.infeasible
            else:
                status_code = StatusCode.other

        status_message = ''

        if status_code == StatusCode.optimal:
            if has_int:
                value = model.getprimalobj(mosek.soltype.itg)

                primals = [0.] * model.getnumvar()
                model.getxx(mosek.soltype.itg, primals)

                reduced_costs = numpy.full((model.getnumvar(),), numpy.nan)
                duals = numpy.full((model.getnumcon(),), numpy.nan)
            elif is_quadratic:
                value = model.getprimalobj(mosek.soltype.itr)

                primals = [0.] * model.getnumvar()
                model.getxx(mosek.soltype.itr, primals)

                reduced_costs = [0.] * model.getnumvar()
                model.getreducedcosts(mosek.soltype.itr, 0, model.getnumvar(), reduced_costs)

                duals = [0.] * model.getnumcon()
                model.gety(mosek.soltype.itr, duals)

                reduced_costs = numpy.array(reduced_costs)
                duals = numpy.array(duals)
            else:
                value = model.getprimalobj(mosek.soltype.bas)

                primals = [0.] * model.getnumvar()
                model.getxx(mosek.soltype.bas, primals)

                reduced_costs = [0.] * model.getnumvar()
                model.getreducedcosts(mosek.soltype.bas, 0, model.getnumvar(), reduced_costs)

                duals = [0.] * model.getnumcon()
                model.gety(mosek.soltype.bas, duals)

                reduced_costs = numpy.array(reduced_costs)
                duals = numpy.array(duals)

            primals = numpy.array(primals)
        else:
            value = numpy.nan
            primals = numpy.full((model.getnumvar(),), numpy.nan)
            reduced_costs = numpy.full((model.getnumvar(),), numpy.nan)
            duals = numpy.full((model.getnumcon(),), numpy.nan)

        return Result(status_code, status_message, value, primals, reduced_costs, duals)

    def get_stats(self):
        """ Get diagnostic information about the model

        Returns:
            :obj:`dict`: diagnostic information about the model
        """
        model = self._model
        stats = {
            'problem': '',
            'sensitivity': '',
            'optimizer': '',
        }

        def func(text): stats['problem'] += text
        model.set_Stream(mosek.streamtype.msg, func)
        model.analyzeproblem(mosek.streamtype.msg)

        def func(text): stats['optimizer'] += text
        model.set_Stream(mosek.streamtype.msg, func)
        model.optimizersummary(mosek.streamtype.msg)

        def func(text): stats['sensitivity'] += text
        model.set_Stream(mosek.streamtype.msg, func)
        model.sensitivityreport(mosek.streamtype.msg)

        model.optimize()

        return stats
