""" MINOS module

References:

* `MINOS <https://web.stanford.edu/group/SOL/minos.htm>`_
* `solveME <https://github.com/SBRG/solvemepy>`_

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2018-08-16
:Copyright: 2018, Karr Lab
:License: MIT
"""

from ..core import (ModelType, ObjectiveDirection,
                    StatusCode, VariableType, Verbosity, Presolve,
                    Constraint, LinearTerm, Model, Term, Variable, Result, ConvOptError,
                    SolverModel)
import attrdict
try:
    import capturer
except ModuleNotFoundError:  # pragma: no cover
    capturer = None  # pragma: no cover
import copy
try:
    from qminospy import qwarmLP
    from qminospy import warmLP
except ImportError:  # pragma: no cover
    import warnings  # pragma: no cover
    warnings.warn('MINOS is not installed', UserWarning)  # pragma: no cover
import numpy
import scipy
import sys


class MinosModel(SolverModel):
    """ `MINOS <https://web.stanford.edu/group/SOL/minos.htm>`_ solver """

    DEFAULT_OPTIONS = {
        'minos': {
            'sense': 'Maximize',
            'solution': 'Solution No',
            'New basis file': 11,
            'Save frequency': 500000,
            'Print level': 0,
            'Print frequency': 100000,
            'Scale option': 2,
            'Iteration limit': 2000000,
            'Expand frequency': 100000,
            'Penalty parameter': 100.0,
            'LU factor tol': 1.9,
            'LU update tol': 1.9,
            'LU singularity tol': 1e-12,
            'Feasibility tol': 1e-7,
            'Optimality tol': 1e-7,
            'Unbounded step size': 1e18,
        },
        'qminos': {
            'sense': 'Maximize',
            'solution': 'Solution No',
            'New basis file': 11,
            'Save frequency': 500000,
            'Print level': 0,
            'Print frequency': 100000,
            'Scale option': 2,
            'Iteration limit': 2000000,
            'Expand frequency': 100000,
            'Penalty parameter': 100.0,
            'LU factor tol': 10.0,
            'LU update tol': 10.0,
            'LU singularity tol': 1e-30,
            'Feasibility tol': 1e-20,
            'Optimality tol': 1e-20,
            'Unbounded step size': 1e+30,
        },
    }

    def load(self, conv_opt_model):
        """ Load a model to MINOS' data structure

        Args:
            conv_opt_model (:obj:`Model`): model

        Returns:
            :obj:`attrdict.AttrDict`: the model in MINOS' data structure

        Raises:
            :obj:`ConvOptError`: if the presolve mode is not supported a variable has an unsupported
                type, an objective has an unsupported term, a
                constraint has an unsupported term, a constraint is unbounded, or the model is not of a
                supported type
        """

        solver_model = attrdict.AttrDict()
        solver_model['probname'] = conv_opt_model.name
        if self._options.precision <= 64:
            solver_model['options'] = copy.deepcopy(self.DEFAULT_OPTIONS['minos'])
        else:
            solver_model['options'] = copy.deepcopy(self.DEFAULT_OPTIONS['qminos'])

        # create variables and set bounds
        lb = numpy.full((len(conv_opt_model.variables), 1), numpy.nan)
        ub = numpy.full((len(conv_opt_model.variables), 1), numpy.nan)
        for i_variable, variable in enumerate(conv_opt_model.variables):
            if variable.type != VariableType.continuous:
                raise ConvOptError('Unsupported variable of type "{}"'.format(variable.type))

            if variable.lower_bound is not None:
                lb[i_variable][0] = variable.lower_bound
            else:
                lb[i_variable][0] = -numpy.inf

            if variable.upper_bound is not None:
                ub[i_variable][0] = variable.upper_bound
            else:
                ub[i_variable][0] = numpy.inf

        # set objective
        if conv_opt_model.objective_direction in [ObjectiveDirection.max, ObjectiveDirection.maximize]:
            solver_model['options']['sense'] = 'Maximize'
        elif conv_opt_model.objective_direction in [ObjectiveDirection.min, ObjectiveDirection.minimize]:
            solver_model['options']['sense'] = 'Minimize'
        else:
            raise ConvOptError('Unsupported objective direction "{}"'.format(conv_opt_model.objective_direction))

        objective = [0] * len(conv_opt_model.variables)
        for term in conv_opt_model.objective_terms:
            if isinstance(term, LinearTerm):
                i = conv_opt_model.variables.index(term.variable)
                objective[i] += term.coefficient
            else:
                raise ConvOptError('Unsupported objective term of type "{}"'.format(term.__class__.__name__))

        # set constraints
        constraint_matrix = numpy.zeros((0, len(conv_opt_model.variables)))
        constraint_matrix_range = numpy.zeros((0, len(conv_opt_model.variables)))
        constraint_senses = []
        constraint_senses_range = []
        rhs = []
        rhs_range = []
        for constraint in conv_opt_model.constraints:
            constraint_row = numpy.zeros((1, len(conv_opt_model.variables)))
            for term in constraint.terms:
                if isinstance(term, LinearTerm):
                    constraint_row[0][conv_opt_model.variables.index(term.variable)] += term.coefficient
                else:
                    raise ConvOptError('Unsupported constraint term of type "{}"'.format(term.__class__.__name__))

            if constraint.lower_bound is None and constraint.upper_bound is None:
                raise ConvOptError('Constraints must have at least one bound')

            if constraint.lower_bound is None:
                constraint_matrix = numpy.concatenate((constraint_matrix, constraint_row))
                constraint_senses.append('L')
                rhs.append(constraint.upper_bound)

            elif constraint.upper_bound is None:
                constraint_matrix = numpy.concatenate((constraint_matrix, constraint_row))
                constraint_senses.append('G')
                rhs.append(constraint.lower_bound)

            elif constraint.lower_bound == constraint.upper_bound:
                constraint_matrix = numpy.concatenate((constraint_matrix, constraint_row))
                constraint_senses.append('E')
                rhs.append(constraint.lower_bound)

            else:
                constraint_matrix = numpy.concatenate((constraint_matrix, constraint_row))
                constraint_senses.append('L')
                rhs.append(constraint.upper_bound)

                constraint_matrix_range = numpy.concatenate((constraint_matrix_range, constraint_row))
                constraint_senses_range.append('G')
                rhs_range.append(constraint.lower_bound)

        constraint_matrix = numpy.concatenate((constraint_matrix, constraint_matrix_range))
        constraint_senses = constraint_senses + constraint_senses_range
        rhs = rhs + rhs_range

        # prepare problem for MINOS
        constraint_matrix_csc = scipy.sparse.coo_matrix(constraint_matrix).tocsc()
        J, ne, P, I, V, bl, bu = makeME_LP(constraint_matrix_csc, rhs, objective, lb, ub, constraint_senses)
        solver_model['m'], solver_model['n'] = m, n = J.shape
        solver_model['ha'] = I
        solver_model['ka'] = P
        solver_model['ad'] = V
        solver_model['bld'] = [bi for bi in bl.flat]
        solver_model['bud'] = [bi for bi in bu.flat]
        solver_model['nb'] = nb = m + n
        solver_model['hs'] = numpy.zeros(nb, numpy.dtype('i4'))

        solver_model['_num_variables'] = len(conv_opt_model.variables)
        solver_model['_num_constraints'] = len(conv_opt_model.constraints)
        solver_model['_objective'] = objective

        return solver_model

    def solve(self):
        """ Solve the model

        Returns:
            :obj:`Result`: result
        """

        model = self._model

        # get solver
        if self._options.precision <= 64:
            solve_func = warmLP.warmlp
        else:
            solve_func = qwarmLP.qwarmlp

        # format options
        self.set_solver_options()

        options = model['options']

        if self._options.presolve != Presolve.off:
            raise ConvOptError('Unsupported presolve mode "{}"'.format(self._options.presolve))

        if self._options.tune:
            raise ConvOptError('Unsupported tune mode "{}"'.format(self._options.tune))

        if self._options.verbosity.value <= Verbosity.status.off.value:
            options['Print level'] = 0
        else:
            options['Print level'] = 1

        str_opts = {}
        int_opts = {}
        float_opts = {}
        for key, val in options.items():
            if isinstance(val, str):
                str_opts[key] = val
            elif isinstance(val, int):
                int_opts[key] = val
            elif isinstance(val, float):
                float_opts[key] = val

        num_str_opts = len(str_opts)
        str_opt_vals = numpy.array(
            numpy.array([c for c in [s.ljust(72) for s in str_opts.values()]],
                        dtype='c').T)

        num_int_opts = len(int_opts)
        keys = int_opts.keys()
        int_opt_names = numpy.array(
            numpy.array([c for c in [s.ljust(55) for s in keys]],
                        dtype='c').T)
        int_opt_vals = numpy.array([int_opts[key] for key in keys], dtype='i4')

        num_float_opts = len(float_opts)
        keys = float_opts.keys()
        float_opt_names = numpy.array(
            numpy.array([c for c in [s.ljust(55) for s in keys]],
                        dtype='c').T)
        float_opt_vals = numpy.array([float_opts[key] for key in keys], dtype='d')

        # allocate memory for results
        model.inform = numpy.array(0)

        # solve model
        if capturer and self._options.verbosity == Verbosity.status.off:
            capture_output = capturer.CaptureOutput(merged=False, relay=False)
            capture_output.start_capture()

        warm = False
        x, duals, reduced_costs = solve_func(
            model.inform, model.probname,
            model.m, model.ha, model.ka, model.ad,
            model.bld, model.bud, model.hs, warm,
            str_opt_vals, int_opt_names, float_opt_names,
            int_opt_vals, float_opt_vals,
            nstropts=num_str_opts, nintopts=num_int_opts, nrealopts=num_float_opts)

        if capturer and self._options.verbosity == Verbosity.status.off:
            capture_output.finish_capture()

        tmp = model.inform
        if tmp == 0:
            status_code = StatusCode.optimal
            status_message = 'optimal'
        else:
            status_code = StatusCode.other
            status_message = 'other'

        if status_code == StatusCode.optimal:
            value = numpy.dot(numpy.array(model['_objective']), numpy.array(x[0:model['_num_variables']]))
            primals = numpy.array(x[0:model['_num_variables']])
            reduced_costs = numpy.array(reduced_costs)
            duals = numpy.array(duals[0:model['_num_constraints']])
        else:
            value = numpy.nan
            primals = numpy.full((model['_num_variables'],), numpy.nan)
            reduced_costs = numpy.full((model['_num_variables'],), numpy.nan)
            duals = numpy.full((model['_num_constraints'],), numpy.nan)

        return Result(status_code, status_message, value, primals, reduced_costs, duals)

    def set_solver_options(self):
        """ Set solver options """
        for key, val in self._options.solver_options.get('minos', {}).items():
            self._model['options'][key] = val

    def get_stats(self):
        """ Get diagnostic information about the model

        Returns:
            :obj:`str`: diagnostic information about the model
        """
        pass

# --- from https://github.com/SBRG/solvemepy --- #


def makeME_LP(S, b, c, xl, xu, csense):
    """
    Create simple LP for qMINOS and MINOS
    Inputs:
    nlp_compat  Make matrices compatible with NLP so that basis can
                be used to warm start NLP by setting 
    12 Aug 2015: first version
    """
    import numpy as np
    import scipy as sp
    import scipy.sparse as sps
    import time

    # c is added as a free (unbounded slacks) row,
    # so that MINOS treats problem as an LP - Ding Ma
    J = sps.vstack((
        S,
        c)
    ).tocsc()
    J.sort_indices()
    if hasattr(b, 'tolist'):
        b = b.tolist()
    b2 = b + [0.0]
    m, n = J.shape
    ne = J.nnz
    # Finally, make the P, I, J, V, as well
    # Row indices: recall fortran is 1-based indexing
    I = [i+1 for i in J.indices]
    V = J.data
    # Pointers to start of each column
    # Just change to 1-based indexing for Fortran
    P = [pi+1 for pi in J.indptr]

    # Make primal and slack bounds
    bigbnd = 1e+40
    # For csense==E rows (equality)
    sl = np.array([bi for bi in b2], ndmin=2).transpose()
    su = np.array([bi for bi in b2], ndmin=2).transpose()
    for row, csen in enumerate(csense):
        if csen == 'L':
            sl[row] = -bigbnd
        elif csen == 'G':
            su[row] = bigbnd
    # Objective row has free bounds
    sl[m-1] = -bigbnd
    su[m-1] = bigbnd

    bl = sp.vstack([xl, sl])
    bu = sp.vstack([xu, su])

    return J, ne, P, I, V, bl, bu
