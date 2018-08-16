""" Tests for the MINOS solver

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2018-08-16
:Copyright: 2018, Karr Lab
:License: MIT
"""

from ..test_core import SolverTestCase
import capturer
import conv_opt
import numpy
import time
import unittest


if conv_opt.Solver.minos in conv_opt.ENABLED_SOLVERS:
    import qminospy


@unittest.skipUnless(conv_opt.Solver.minos in conv_opt.ENABLED_SOLVERS, 'MINOS is not installed')
class MinosTestCase(SolverTestCase):
    def test_solve_lp_minos(self):
        model = self.create_lp()
        self.assert_lp(model, 'minos')

    def test_solve_lp_qminos(self):
        model = self.create_lp()
        self.assert_lp(model, 'minos', precision=128)

    def test_unsupported_variable_types(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='binary_var', type=None)
        model.variables.append(var)
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported variable of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.minos))

    def test_unsupported_objective_direction(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_direction = None
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective direction'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.minos))

    def test_unsupported_objective_term(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_terms = [None]
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective term of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.minos))

    def test_unsupported_constraint_type(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            None,
        ], upper_bound=0, lower_bound=0))
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported constraint term of type '):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.minos))

    def test_unconstrainted_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], upper_bound=None, lower_bound=None))
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Constraints must have at least one bound'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.minos))

    def test_unsupported_presolve_mode(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], upper_bound=1, lower_bound=-1))
        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.minos, presolve=conv_opt.Presolve.on))
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported presolve mode '):
            result = solver_model.solve()

    def test_unsupported_tune_mode(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], upper_bound=1, lower_bound=-1))
        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.minos, tune=True))
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported tune mode '):
            result = solver_model.solve()

    def test_verbose(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.minos,
                                        verbosity=conv_opt.Verbosity.status)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)
            self.assertNotEqual(captured.stdout.get_text(), '')
            self.assertEqual(captured.stderr.get_text(), '')

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.minos,
                                        verbosity=conv_opt.Verbosity.off)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)

            self.assertEqual(captured.stdout.get_text(), '')
            self.assertEqual(captured.stderr.get_text(), '')

    def test_set_solver_options(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)
        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]
        options = conv_opt.SolveOptions(
            solver=conv_opt.Solver.minos,
            solver_options={
                'minos': {
                    'Print frequency': 10000,
                },
            })
        minos_model = model.convert(options=options)
        self.assertEqual(minos_model._model['options']['Print frequency'], 100000)

        minos_model.solve()

        self.assertEqual(minos_model._model['options']['Print frequency'], 10000)

    def test_infeasible(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)

        cons = conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], lower_bound=3., upper_bound=5., name='constraint')
        model.constraints.append(cons)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.minos))
        self.assertEqual(result.status_code, conv_opt.StatusCode.other)
        numpy.testing.assert_equal(result.value, numpy.nan)
        numpy.testing.assert_array_equal(result.primals, numpy.array([numpy.nan]))
        numpy.testing.assert_array_equal(result.reduced_costs, numpy.array([numpy.nan]))
        numpy.testing.assert_array_equal(result.duals, numpy.array([numpy.nan]))
