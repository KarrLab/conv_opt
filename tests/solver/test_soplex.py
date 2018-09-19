""" Tests for the SoPlex solver

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2018-08-17
:Copyright: 2018, Karr Lab
:License: MIT
"""

from ..test_core import SolverTestCase
import capturer
import conv_opt
import mock
import numpy
import time
import unittest


if conv_opt.Solver.soplex in conv_opt.ENABLED_SOLVERS:
    import soplex


@unittest.skipUnless(conv_opt.Solver.soplex in conv_opt.ENABLED_SOLVERS, 'SoPlex is not installed')
class SoplexTestCase(SolverTestCase):
    def test_solve_lp_soplex(self):
        model = self.create_lp()
        self.assert_lp(model, 'soplex', check_reduced_costs=False)

    def test_unsupported_variable_types(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='binary_var', type=None)
        model.variables.append(var)
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported variable of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.soplex))

    def test_unsupported_objective_direction(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_direction = None
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported objective direction'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.soplex))

    def test_unsupported_objective_term(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_terms = [None]
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported objective term of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.soplex))

    def test_unsupported_constraint_type(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            None,
        ], upper_bound=0, lower_bound=0))
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported constraint term of type '):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.soplex))

    def test_unconstrainted_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], upper_bound=None, lower_bound=None))
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Constraints must have at least one bound'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.soplex))

    def test_unsupported_presolve_mode(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], upper_bound=1, lower_bound=-1))
        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.soplex, presolve=conv_opt.Presolve.on))
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported presolve mode '):
            result = solver_model.solve()

    def test_unsupported_tune_mode(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], upper_bound=1, lower_bound=-1))
        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.soplex, tune=True))
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported tune mode '):
            result = solver_model.solve()

    def test_verbose(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.soplex,
                                        verbosity=conv_opt.Verbosity.status)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)
            self.assertNotEqual(captured.stdout.get_text(), '')
            self.assertEqual(captured.stderr.get_text(), '')

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.soplex,
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
            solver=conv_opt.Solver.soplex,
            solver_options={
                'soplex': {
                    'ITERLIMIT': 10000,
                },
            })
        soplex_model = model.convert(options=options)
        soplex_model.solve()

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

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.soplex))
        self.assertEqual(result.status_code, conv_opt.StatusCode.other)
        numpy.testing.assert_equal(result.value, numpy.nan)
        numpy.testing.assert_array_equal(result.primals, numpy.array([numpy.nan]))
        numpy.testing.assert_array_equal(result.reduced_costs, numpy.array([numpy.nan]))
        numpy.testing.assert_array_equal(result.duals, numpy.array([numpy.nan]))

    def test_repeated_constraint(self):
        model = conv_opt.Model()

        ex_a = conv_opt.Variable(name='ex_a', lower_bound=0., upper_bound=10.)
        model.variables.append(ex_a)

        ex_b = conv_opt.Variable(name='ex_b', lower_bound=0., upper_bound=10.)
        model.variables.append(ex_b)

        rxn = conv_opt.Variable(name='rxn', lower_bound=0., upper_bound=100.)
        model.variables.append(rxn)

        cons = conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=ex_a, coefficient=1.),
            conv_opt.LinearTerm(variable=rxn, coefficient=-1.),
            conv_opt.LinearTerm(variable=rxn, coefficient=-1.),
        ], lower_bound=-1., upper_bound=1., name='meta_')
        model.constraints.append(cons)

        cons = conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=rxn, coefficient=1.),
            conv_opt.LinearTerm(variable=ex_b, coefficient=-1.),
        ], lower_bound=0., upper_bound=0., name='met_b')
        model.constraints.append(cons)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=rxn, coefficient=1.)]

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.soplex))
        numpy.testing.assert_equal(result.value, 5.5)

    def test_export(self):
        self.assert_export('lp', conv_opt.Solver.soplex)


class MetabolitesListTestCase(unittest.TestCase):
    def test(self):
        mets = conv_opt.solver.soplex.MetabolitesList()

        met_0 = mock.Mock(id='met_0')
        met_1 = mock.Mock(id='met_1')
        met_2 = mock.Mock(id='met_2')

        mets.append(met_2)
        mets.append(met_0)
        mets.append(met_1)

        self.assertEqual(mets.index('met_0'), 1)
        self.assertEqual(mets.index('met_1'), 2)
        self.assertEqual(mets.index('met_2'), 0)

        with self.assertRaisesRegex(ValueError, 'not in list'):
            mets.index('met_3')
