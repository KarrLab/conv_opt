""" quadprog module

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
import quadprog


class QuadprogModel(SolverModel):
    """ quadprog solver

    Attributes:
        _stats (:obj:`dict`): diagnostic information about the solution
    """

    def __init__(self, model, options=None):
        """
        Args:
            model (:obj:`Model`): model
            options (:obj:`SolveOptions`, optional): options
        """
        super(QuadprogModel, self).__init__(model, options=options)
        self._stats = {}

    def load(self, conv_opt_model):
        """ Load a model to quadprog's data structure

        Args:
            conv_opt_model (:obj:`Model`): model

        Returns:
            :obj:`dict`: the model in quadprog's data structure

        Raises:
            :obj:`ConvOptError`: if the presolve mode is unsupported, a variable has an unsupported type,
                an objective has an unsupported term, a constraint has
                an unsupported term, or a constraint is unbounded
        """

        # variables
        for variable in conv_opt_model.variables:
            if variable.type != VariableType.continuous:
                raise ConvOptError('Unsupported variable of type "{}"'.format(variable.type))

        # objective
        a = numpy.zeros((len(conv_opt_model.variables), ))
        G = numpy.zeros((len(conv_opt_model.variables), len(conv_opt_model.variables)))
        for term in conv_opt_model.objective_terms:
            if isinstance(term, LinearTerm):
                i_variable = conv_opt_model.variables.index(term.variable)
                a[i_variable] -= term.coefficient
            elif isinstance(term, QuadraticTerm):
                i_variable_1 = conv_opt_model.variables.index(term.variable_1)
                i_variable_2 = conv_opt_model.variables.index(term.variable_2)
                G[i_variable_1, i_variable_2] += term.coefficient
                G[i_variable_2, i_variable_1] += term.coefficient
            else:
                raise ConvOptError('Unsupported objective term of type "{}"'.format(term.__class__.__name__))

        if conv_opt_model.objective_direction in [ObjectiveDirection.max, ObjectiveDirection.maximize]:
            a = -a
            G = -G
        elif conv_opt_model.objective_direction in [ObjectiveDirection.min, ObjectiveDirection.minimize]:
            pass
        else:
            raise ConvOptError('Unsupported objective direction "{}"'.format(conv_opt_model.objective_direction))

        # constraints
        b_eq = numpy.zeros((0, ))
        C_eq = numpy.zeros((0, len(conv_opt_model.variables)))
        C_eq_label = []
        b_ineq = numpy.zeros((0, ))
        C_ineq = numpy.zeros((0, len(conv_opt_model.variables)))
        C_ineq_label = []
        for i_constraint, constraint in enumerate(conv_opt_model.constraints):
            C_i = numpy.zeros((1, len(conv_opt_model.variables)))
            for term in constraint.terms:
                if isinstance(term, LinearTerm):
                    i_variable = conv_opt_model.variables.index(term.variable)
                    C_i[0, i_variable] += term.coefficient
                else:
                    raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))

            if constraint.lower_bound is None and constraint.upper_bound is None:
                raise ConvOptError('Constraints must have at least one bound')
            elif constraint.lower_bound is None:
                C_ineq = numpy.concatenate((C_ineq, -C_i))
                b_ineq = numpy.concatenate((b_ineq, [-constraint.upper_bound]))
                C_ineq_label.append(constraint.name)
            elif constraint.upper_bound is None:
                C_ineq = numpy.concatenate((C_ineq, C_i))
                b_ineq = numpy.concatenate((b_ineq, [constraint.lower_bound]))
                C_ineq_label.append(constraint.name)
            elif constraint.lower_bound == constraint.upper_bound:
                C_eq = numpy.concatenate((C_eq, C_i))
                b_eq = numpy.concatenate((b_eq, [constraint.lower_bound]))
                C_eq_label.append(constraint.name)
            else:
                C_ineq = numpy.concatenate((C_ineq, -C_i))
                C_ineq = numpy.concatenate((C_ineq, C_i))
                b_ineq = numpy.concatenate((b_ineq, [-constraint.upper_bound]))
                b_ineq = numpy.concatenate((b_ineq, [constraint.lower_bound]))
                C_ineq_label.append(constraint.name + '_upper')
                C_ineq_label.append(constraint.name + '_lower')

        # variable bounds
        for i_variable, variable in enumerate(conv_opt_model.variables):
            C_i = numpy.zeros((1, len(conv_opt_model.variables)))
            C_i[0, i_variable] = 1.

            if variable.lower_bound is None and variable.upper_bound is None:
                continue
            elif variable.lower_bound is None:
                C_ineq = numpy.concatenate((C_ineq, -C_i))
                b_ineq = numpy.concatenate((b_ineq, [-variable.upper_bound]))
                C_ineq_label.append(variable.name)
            elif variable.upper_bound is None:
                C_ineq = numpy.concatenate((C_ineq, C_i))
                b_ineq = numpy.concatenate((b_ineq, [variable.lower_bound]))
                C_ineq_label.append(variable.name)
            elif variable.lower_bound == variable.upper_bound:
                C_eq = numpy.concatenate((C_eq, C_i))
                b_eq = numpy.concatenate((b_eq, [variable.lower_bound]))
                C_eq_label.append(variable.name)
            else:
                C_ineq = numpy.concatenate((C_ineq, -C_i))
                C_ineq = numpy.concatenate((C_ineq, C_i))
                b_ineq = numpy.concatenate((b_ineq, [-variable.upper_bound]))
                b_ineq = numpy.concatenate((b_ineq, [variable.lower_bound]))
                C_eq_label.append(variable.name + '_upper')
                C_eq_label.append(variable.name + '_lower')

        # prepare input arguments
        C = numpy.concatenate((C_eq, C_ineq))
        b = numpy.concatenate((b_eq, b_ineq))
        meq = b_eq.size

        return {
            'G': G,
            'a': a,
            'C': C.transpose(),
            'b': b,
            'meq': meq,
            'objective_direction': conv_opt_model.objective_direction,
            'constraint_names': C_eq_label + C_ineq_label,
        }

    def solve(self):
        """ Solve the model

        Returns:
            :obj:`Result`: result
        """

        model = self._model

        kwargs = copy.copy(model)
        del(kwargs['objective_direction'])
        del(kwargs['constraint_names'])

        # set presolve
        if self._options.presolve == Presolve.on:
            kwargs['factorized'] = True
        elif self._options.presolve == Presolve.off:
            kwargs['factorized'] = False
        else:
            raise ConvOptError('Unsupported presolve mode "{}"'.format(self._options.presolve))

        # tune
        if self._options.tune:
            raise ConvOptError('Unsupported tuning mode {}'.format(self._options.tune))

        # solve
        self._stats.clear()

        try:
            primals, value, xu, iters, reduced_costs, iact = quadprog.solve_qp(**kwargs)
            status_code = StatusCode.optimal
            status_message = ''

            if model['objective_direction'] in [ObjectiveDirection.max, ObjectiveDirection.maximize]:
                value = -value

            self._stats['xu'] = xu
            self._stats['iterations'] = iters
            self._stats['iact'] = iact

        except ValueError as err:
            status_code = StatusCode.other
            status_message = str(err)
            value = numpy.nan
            primals = [numpy.nan] * model['a'].size
            reduced_costs = [numpy.nan] * model['a'].size

            self._stats['xu'] = [numpy.nan] * model['a'].size
            self._stats['iterations'] = (None, None)
            self._stats['iact'] = []

        duals = [numpy.nan] * len(model['constraint_names'])

        return Result(status_code, status_message, value, numpy.array(primals), numpy.array(reduced_costs), numpy.array(duals))

    def get_stats(self):
        """ Get diagnostic information about the model

        Returns:
            :obj:`dict`: diagnostic information about the model
        """
        return self._stats
