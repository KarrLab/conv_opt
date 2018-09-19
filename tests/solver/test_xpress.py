""" Tests for the FICO XPRESS solver

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2017-11-28
:Copyright: 2017, Karr Lab
:License: MIT
"""

from ..test_core import SolverTestCase
import capturer
import conv_opt
import numpy
import unittest

if conv_opt.Solver.xpress in conv_opt.ENABLED_SOLVERS:
    import xpress


@unittest.skipUnless(conv_opt.Solver.xpress in conv_opt.ENABLED_SOLVERS, 'FICO XPRESS is not installed')
class XpressTestCase(SolverTestCase):

    def test_convert(self):
        model = self.create_lp()
        xpress_model = model.convert(conv_opt.SolveOptions(solver=conv_opt.Solver.xpress)).get_model()

        inf = float('inf')
        xpress_vars = xpress_model.getVariable()
        self.assertEqual([var.name for var in xpress_vars], ['ex_a', 'r1', 'r2', 'r3', 'r4', 'biomass_production', 'ex_biomass'])
        self.assertEqual([var.lb for var in xpress_vars], [-inf, 0., 0., 0., 0., 0., 0.])
        self.assertEqual([var.ub for var in xpress_vars], [1., inf, inf, inf, inf, inf, inf])
        self.assertEqual([var.vartype for var in xpress_vars],
                         [xpress.continuous, xpress.continuous, xpress.continuous,
                          xpress.continuous, xpress.continuous, xpress.continuous, xpress.continuous])

        # self.assertIsInstance(xpress_model.objective.sense, xpress.maximize)
        obj = []
        xpress_model.getobj(obj, first=0, last=len(xpress_vars) - 1)
        self.assertEqual(obj, [0., 0., 0., 0., 0., 1.0, 0., ])

        inf = 1e20
        self.assertEqual([c.name for c in xpress_model.getConstraint()],
                         ['a', 'b', 'c', 'd', 'e', 'biomass', 'upper_bound', 'lower_bound',
                          'range_bound__lower__', 'range_bound__upper__'])
        self.assertEqual([c.lb for c in xpress_model.getConstraint()],
                         [0., 0., 0., 0., 0., 0., -inf, -1., -10., -inf])
        self.assertEqual([c.ub for c in xpress_model.getConstraint()],
                         [0., 0., 0., 0., 0., 0., 1., inf, inf, 10.])

        self.assertIn(str(xpress_model.getConstraint(0).body), [
            str(-1. * xpress_vars[0] + 1. * xpress_vars[1]),
            str(1. * xpress_vars[0] + -1. * xpress_vars[1]),
        ])
        self.assertIn(str(xpress_model.getConstraint(1).body), [
            str(1. * xpress_vars[1] + -1. * xpress_vars[2] + -1. * xpress_vars[3]),
            str(-1. * xpress_vars[1] + 1. * xpress_vars[2] + 1. * xpress_vars[3]),
        ])
        self.assertIn(str(xpress_model.getConstraint(2).body), [
            str(1. * xpress_vars[2]),
            str(-1. * xpress_vars[2]),
        ])
        self.assertIn(str(xpress_model.getConstraint(3).body), [
            str(1. * xpress_vars[2] + -1. * xpress_vars[4]),
            str(-1. * xpress_vars[2] + 1. * xpress_vars[4]),
        ])
        self.assertIn(str(xpress_model.getConstraint(4).body), [
            str(2. * xpress_vars[3] + 1. * xpress_vars[4] + -1 * xpress_vars[5]),
            str(-2. * xpress_vars[3] + -1. * xpress_vars[4] + 1. * xpress_vars[5]),
        ])
        self.assertIn(str(xpress_model.getConstraint(5).body), [
            str(1. * xpress_vars[5] + -1. * xpress_vars[6]),
            str(-1. * xpress_vars[5] + 1. * xpress_vars[6]),
        ])
        self.assertIn(str(xpress_model.getConstraint(6).body), [
            str(-1. * xpress_vars[0]),
            str(1. * xpress_vars[0]),
        ])
        self.assertIn(str(xpress_model.getConstraint(7).body), [
            str(1. * xpress_vars[5]),
            str(-1. * xpress_vars[5]),
        ])
        self.assertIn(str(xpress_model.getConstraint(8).body), [
            str(-1. * xpress_vars[1]),
            str(1. * xpress_vars[1]),
        ])
        self.assertIn(str(xpress_model.getConstraint(9).body), [
            str(-1. * xpress_vars[1]),
            str(1. * xpress_vars[1]),
        ])

    def test_solve_lp(self):
        model = self.create_lp()
        self.assert_lp(model, 'xpress', status_message='lp_optimal')

    def test_solve_milp(self):
        model = self.create_milp()
        self.assert_milp(model, 'xpress', status_message='mip_optimal')

    def test_solve_qp(self):
        model = self.create_qp()
        self.assert_qp(model, 'xpress', status_message='lp_optimal')

    def test_solve_qp_2(self):
        model = self.create_qp_2()
        self.assert_qp_2(model, 'xpress')

    def test_solve_qp_3(self):
        model = self.create_qp_3()
        self.assert_qp_3(model, 'xpress')

    def test_solve_qp_4(self):
        model = self.create_qp_4()
        self.assert_qp_4(model, 'xpress')

    @unittest.expectedFailure
    def test_solve_miqp(self):
        # failure expected; XPRESS doesn't seem to find as optimal a solution as other solvers
        model = self.create_miqp()
        self.assert_miqp(model, 'xpress', status_message='mip_optimal')

    def test_supported_variable_types(self):
        model = conv_opt.Model()

        binary_var = conv_opt.Variable(name='binary_var', type=conv_opt.VariableType.binary)
        model.variables.append(binary_var)

        integer_var = conv_opt.Variable(name='integer_var', type=conv_opt.VariableType.integer)
        model.variables.append(integer_var)

        continuous_var = conv_opt.Variable(name='continuous_var', type=conv_opt.VariableType.continuous)
        model.variables.append(continuous_var)

        semi_integer_var = conv_opt.Variable(name='semi_integer_var', type=conv_opt.VariableType.semi_integer)
        model.variables.append(semi_integer_var)

        semi_continuous_var = conv_opt.Variable(name='semi_continuous_var', type=conv_opt.VariableType.semi_continuous)
        model.variables.append(semi_continuous_var)

        part_integer_var = conv_opt.Variable(name='part_integer_var', type=conv_opt.VariableType.partially_integer)
        model.variables.append(part_integer_var)

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress))
        xpress_model = solver_model.get_model()
        xpress_vars = xpress_model.getVariable()
        self.assertEqual(xpress_vars[0].vartype, xpress.binary)
        self.assertEqual(xpress_vars[1].vartype, xpress.integer)
        self.assertEqual(xpress_vars[2].vartype, xpress.continuous)
        self.assertEqual(xpress_vars[3].vartype, xpress.semiinteger)
        self.assertEqual(xpress_vars[4].vartype, xpress.semicontinuous)
        self.assertEqual(xpress_vars[5].vartype, xpress.partiallyinteger)

    def test_unsupported_variable_types(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='binary_var', type=None)
        model.variables.append(var)
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported variable of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress))

    def test_unsupported_objective_direction(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_direction = None
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective direction'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress))

    def test_unsupported_objective_term(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_terms = [None]
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective term of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress))

    def test_unconstrained_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint())
        with self.assertRaisesRegexp(conv_opt.ConvOptError, 'Constraints must have at least one bound'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress))

    def test_unsupported_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            None,
        ], upper_bound=0, lower_bound=0))
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported constraint term of type '):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress))

    def test_verbose_presolve(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.xpress,
                                        presolve=conv_opt.Presolve.on,
                                        verbosity=conv_opt.Verbosity.status)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)

            self.assertRegex(captured.stdout.get_text(), '^Problem is nonlinear presolved')
            self.assertEqual(captured.stderr.get_text(), '')

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.xpress,
                                        presolve=conv_opt.Presolve.on,
                                        verbosity=conv_opt.Verbosity.error)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)

            self.assertRegex(captured.stdout.get_text(), 'Maximizing LP ')
            self.assertEqual(captured.stderr.get_text(), '')

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.xpress,
                                        presolve=conv_opt.Presolve.on,
                                        verbosity=conv_opt.Verbosity.off)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)

            self.assertEqual(captured.stdout.get_text(), '')
            self.assertEqual(captured.stderr.get_text(), '')

    def test_verbose(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.xpress,
                                        verbosity=conv_opt.Verbosity.status)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)

            self.assertRegex(captured.stdout.get_text(), '^Maximizing LP ')
            self.assertEqual(captured.stderr.get_text(), '')

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.xpress,
                                        verbosity=conv_opt.Verbosity.off)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)

            self.assertEqual(captured.stdout.get_text(), '')
            self.assertEqual(captured.stderr.get_text(), '')

    def test_unsupported_presolve(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported presolve mode '):
            model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress, presolve=None))

    def test_tune(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)
        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported tuning mode '):
            result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress, tune=True))

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

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress))
        self.assertEqual(result.status_code, conv_opt.StatusCode.infeasible)
        numpy.testing.assert_equal(result.value, numpy.nan)
        numpy.testing.assert_array_equal(result.primals, numpy.array([numpy.nan]))
        numpy.testing.assert_array_equal(result.reduced_costs, numpy.array([numpy.nan]))
        numpy.testing.assert_array_equal(result.duals, numpy.array([numpy.nan]))

    def test_unbounded(self):
        model = conv_opt.Model()

        var_1 = conv_opt.Variable(name='var_1')
        model.variables.append(var_1)

        var_2 = conv_opt.Variable(name='var_2')
        model.variables.append(var_2)

        cons = conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=var_2, coefficient=1.)
        ], lower_bound=3., upper_bound=5., name='constraint')
        model.constraints.append(cons)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var_1, coefficient=1.)]

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress))
        self.assertEqual(result.status_code, conv_opt.StatusCode.other)
        numpy.testing.assert_equal(result.value, numpy.nan)
        numpy.testing.assert_array_equal(result.primals, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.reduced_costs, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.duals, numpy.array([numpy.nan]))

    def test_milp_optimal(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='variable',
                                lower_bound=-3, upper_bound=2.,
                                type=conv_opt.VariableType.integer)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress))
        self.assertEqual(result.status_code, conv_opt.StatusCode.optimal)
        numpy.testing.assert_equal(result.value, 2.)

    def test_milp_infeasible(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='variable',
                                lower_bound=-3, upper_bound=2.,
                                type=conv_opt.VariableType.integer)
        model.variables.append(var)

        cons = conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], lower_bound=3., upper_bound=5., name='constraint')
        model.constraints.append(cons)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress))
        self.assertEqual(result.status_code, conv_opt.StatusCode.infeasible)
        numpy.testing.assert_equal(result.value, numpy.nan)

    @unittest.expectedFailure
    def test_milp_unbounded(self):
        # XPRESS returns optimal even though the problem isn't bounded
        # XPRESS seems to add default bounds to integer variables

        model = conv_opt.Model()

        var = conv_opt.Variable(name='var',
                                type=conv_opt.VariableType.integer)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress))

        self.assertEqual(result.status_code, conv_opt.StatusCode.other)

    def test_infeasible_qp(self):
        model = self.create_qp()
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=model.variables[0], coefficient=1.)
        ], upper_bound=-1.))
        results = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress))
        self.assertEqual(results.status_code, conv_opt.StatusCode.infeasible)

    def test_unbounded_qp(self):
        model = self.create_qp()
        model.variables[0].upper_bound = None
        results = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.xpress))
        self.assertEqual(results.status_code, conv_opt.StatusCode.other)

    def test_export(self):
        self.assert_export('lp', conv_opt.Solver.xpress)
        self.assert_export('mps', conv_opt.Solver.xpress)
