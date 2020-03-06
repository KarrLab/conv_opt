""" SoPlex module

References:

* `SoPlex <http://soplex.zib.de>`_
* `soplex_cython <https://github.com/SBRG/soplex_cython>`_

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2018-08-17
:Copyright: 2018, Karr Lab
:License: MIT
"""

from ..core import (ModelType, ObjectiveDirection, Presolve,
                    SolveOptions, Solver, StatusCode, VariableType, Verbosity,
                    Constraint, LinearTerm, Model, QuadraticTerm, Term, Variable, Result, ConvOptError,
                    SolverModel)
import copy
import mock
import numpy
try:
    import soplex
except ImportError:  # pragma: no cover
    import warnings  # pragma: no cover
    warnings.warn('SoPlex is not installed', UserWarning)  # pragma: no cover
import sys


class SoplexModel(SolverModel):
    """ `SoPlex <http://soplex.zib.de>`_ solver """

    INF = 1e256

    DEFAULT_OPTIONS = {
        'ITERLIMIT':  2 ** 31 - 1,
        'FEASTOL': 1e-20,
        'OPTTOL': 1e-20,
    }

    def load(self, conv_opt_model):
        """ Load a model to SoPlex's data structure

        Args:
            conv_opt_model (:obj:`Model`): model

        Returns:
            :obj:`soplex.Soplex`: the model in SoPlex's data structure

        Raises:
            :obj:`ConvOptError`: if the presolve mode is not supported a variable has an unsupported
                type, an objective has an unsupported term, a
                constraint has an unsupported term, a constraint is unbounded, or the model is not of a
                supported type
        """

        cobra_model = mock.Mock()

        # create variables and set bounds
        cobra_model.reactions = []
        for i_variable, variable in enumerate(conv_opt_model.variables):
            rxn = mock.Mock()
            cobra_model.reactions.append(rxn)
            rxn.id = variable.name or ('variable_' + str(i_variable))

            if variable.type != VariableType.continuous:
                raise ConvOptError('Unsupported variable of type "{}"'.format(variable.type))

            if variable.lower_bound is not None:
                rxn.lower_bound = max(variable.lower_bound, -self.INF)
            else:
                rxn.lower_bound = -self.INF

            if variable.upper_bound is not None:
                rxn.upper_bound = min(variable.upper_bound, self.INF)
            else:
                rxn.upper_bound = self.INF

            rxn._metabolites = {}
            rxn.objective_coefficient = 0.

        # set objective
        if conv_opt_model.objective_direction in [ObjectiveDirection.max, ObjectiveDirection.maximize]:
            objective_sense = 'maximize'
        elif conv_opt_model.objective_direction in [ObjectiveDirection.min, ObjectiveDirection.minimize]:
            objective_sense = 'minimize'
        else:
            raise ConvOptError('Unsupported objective direction "{}"'.format(conv_opt_model.objective_direction))

        for term in conv_opt_model.objective_terms:
            if isinstance(term, LinearTerm):
                i_variable = conv_opt_model.variables.index(term.variable)
                cobra_model.reactions[i_variable].objective_coefficient += term.coefficient
            else:
                raise ConvOptError('Unsupported objective term of type "{}"'.format(term.__class__.__name__))

        # set constraints
        cobra_model.metabolites = MetabolitesList()
        range_metabolites = []
        for i_constraint, constraint in enumerate(conv_opt_model.constraints):
            met = mock.Mock()
            met2 = mock.Mock()

            met.id = constraint.name or ('constraint_' + str(i_constraint))
            met2.id = constraint.name + ' _lower_bound_of_range'

            range_constraint = False
            if constraint.lower_bound is None and constraint.upper_bound is None:
                raise ConvOptError('Constraints must have at least one bound')
            if constraint.lower_bound is None:
                met._constraint_sense = 'L'
                met._bound = min(constraint.upper_bound, self.INF)
            elif constraint.upper_bound is None:
                met._constraint_sense = 'G'
                met._bound = max(constraint.lower_bound, -self.INF)
            elif constraint.lower_bound == constraint.upper_bound:
                met._constraint_sense = 'E'
                met._bound = max(constraint.lower_bound, -self.INF)
            else:
                range_constraint = True

                met._constraint_sense = 'L'
                met._bound = min(constraint.upper_bound, self.INF)

                met2._constraint_sense = 'G'
                met2._bound = max(constraint.lower_bound, -self.INF)

            for term in constraint.terms:
                if isinstance(term, LinearTerm):
                    i_reaction = conv_opt_model.variables.index(term.variable)
                    if met in cobra_model.reactions[i_reaction]._metabolites:
                        cobra_model.reactions[i_reaction]._metabolites[met] += term.coefficient
                        if range_constraint:
                            cobra_model.reactions[i_reaction]._metabolites[met2] += term.coefficient
                    else:
                        cobra_model.reactions[i_reaction]._metabolites[met] = term.coefficient
                        if range_constraint:
                            cobra_model.reactions[i_reaction]._metabolites[met2] = term.coefficient
                else:
                    raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))

            cobra_model.metabolites.append(met)
            if range_constraint:
                range_metabolites.append(met2)

        cobra_model.metabolites += range_metabolites

        # convert model to SoPlex problem
        soplex_model = soplex.Soplex(cobra_model)
        soplex_model.set_objective_sense(objective_sense)

        self._cobra_model = cobra_model
        self._num_variables = len(conv_opt_model.variables)
        self._num_constraints = len(conv_opt_model.constraints)

        return soplex_model

    def solve(self):
        """ Solve the model

        Returns:
            :obj:`Result`: result
        """

        model = self._model

        # set solver options
        options = copy.copy(self.DEFAULT_OPTIONS)
        for key, val in self._options.solver_options.get('soplex', {}).items():
            options[key] = val

        # set verbosity
        options['VERBOSITY'] = self._options.verbosity.value

        # set presolve mode
        if self._options.presolve == Presolve.on:
            raise ConvOptError('Unsupported presolve mode "{}"'.format(self._options.presolve))

        # tune
        if self._options.tune:
            raise ConvOptError('Unsupported tune mode "{}"'.format(self._options.tune))

        # solve problem
        status_message = model.solve_problem(**options)
        if status_message == 'optimal':
            status_code = StatusCode.optimal
            sol = model.format_solution(self._cobra_model)
            value = model.get_objective_value()
            primals = numpy.array(sol.x)
            reduced_costs = numpy.full((model.numCols,), numpy.nan)
            duals = numpy.array(sol.y[0:self._num_constraints])
        else:
            status_code = StatusCode.other
            value = numpy.nan
            primals = numpy.full((model.numCols,), numpy.nan)
            reduced_costs = numpy.full((model.numCols,), numpy.nan)
            duals = numpy.full((self._num_constraints,), numpy.nan)

        return Result(status_code, status_message, value, primals, reduced_costs, duals)

    def set_solver_options(self):
        """ Set solver options """
        pass  # pragma: no cover

    def get_stats(self):
        """ Get diagnostic information about the model

        Returns:
            :obj:`str`: diagnostic information about the model
        """
        pass


class MetabolitesList(list):
    """ List of metabolites of a COBRA model """

    def index(self, id):
        """ Get the index of the metabolite with identifier :obj:`id`

        Args:
            id (:obj:`str`): identifier

        Returns:
            :obj:`int`: index of metabolite with identifier :obj:`id` in list

        Raises:
            :obj:`ValueError`: if no metabolite with identifier :obj:`id` is in list
        """
        for i_metabolite, metabolite in enumerate(self):
            if metabolite.id == id:
                return i_metabolite
        raise ValueError("'{}' is not in list".format(id))
