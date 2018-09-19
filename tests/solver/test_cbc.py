""" Tests for the CBC solver

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2017-11-28
:Copyright: 2017, Karr Lab
:License: MIT
"""

from ..test_core import SolverTestCase
import capturer
import conv_opt
import mock
import numpy
import unittest

if conv_opt.Solver.cbc in conv_opt.ENABLED_SOLVERS:
    import cylp.py.modeling
    from cylp.py.modeling.CyLPModel import getCoinInfinity


@unittest.skipUnless(conv_opt.Solver.cbc in conv_opt.ENABLED_SOLVERS, 'Cbc is not installed')
class CbcTestCase(SolverTestCase):

    def test_convert(self):
        inf = getCoinInfinity()

        model = self.create_lp()
        cylp_model = model.convert(conv_opt.SolveOptions(solver=conv_opt.Solver.cbc)).get_model()

        self.assertEqual([v.name for v in cylp_model.variables], ['ex_a', 'r1', 'r2', 'r3', 'r4', 'biomass_production', 'ex_biomass'])
        self.assertEqual([v.lower[0] for v in cylp_model.variables], [-inf, 0., 0., 0., 0., 0., 0.])
        self.assertEqual([v.upper[0] for v in cylp_model.variables], [1., inf, inf, inf, inf, inf, inf])
        self.assertEqual(set([v.isInt for v in cylp_model.variables]), set([False]))

        self.assertEqual(cylp_model.optimizationDirection, 'max')
        numpy.testing.assert_array_almost_equal(cylp_model.objective, numpy.array([0., 0., 0., 0., 0., 1., 0.]))

        self.assertEqual([c.lower[0] for c in cylp_model.constraints], [0., 0., 0., 0., 0., 0., -inf, -1., -10.])
        self.assertEqual([c.upper[0] for c in cylp_model.constraints], [0., 0., 0., 0., 0., 0., 1., inf, 10.])

        self.assertEqual(cylp_model.constraints[0].variables, cylp_model.variables[0:2])
        self.assertEqual(cylp_model.constraints[0].varCoefs[cylp_model.variables[0]], 1)
        self.assertEqual(cylp_model.constraints[0].varCoefs[cylp_model.variables[1]], -1)

        self.assertEqual(cylp_model.constraints[1].variables, cylp_model.variables[1:4])
        self.assertEqual(cylp_model.constraints[1].varCoefs[cylp_model.variables[1]], 1)
        self.assertEqual(cylp_model.constraints[1].varCoefs[cylp_model.variables[2]], -1)
        self.assertEqual(cylp_model.constraints[1].varCoefs[cylp_model.variables[3]], -1)

        self.assertEqual(cylp_model.constraints[2].variables, cylp_model.variables[2:3])
        self.assertEqual(cylp_model.constraints[2].varCoefs[cylp_model.variables[2]], 1)

        self.assertEqual(cylp_model.constraints[3].variables, [cylp_model.variables[2], cylp_model.variables[4]])
        self.assertEqual(cylp_model.constraints[3].varCoefs[cylp_model.variables[2]], 1)
        self.assertEqual(cylp_model.constraints[3].varCoefs[cylp_model.variables[4]], -1)

        self.assertEqual(cylp_model.constraints[4].variables, cylp_model.variables[3:6])
        self.assertEqual(cylp_model.constraints[4].varCoefs[cylp_model.variables[3]], 2)
        self.assertEqual(cylp_model.constraints[4].varCoefs[cylp_model.variables[4]], 1)
        self.assertEqual(cylp_model.constraints[4].varCoefs[cylp_model.variables[5]], -1)

        self.assertEqual(cylp_model.constraints[5].variables, cylp_model.variables[5:7])
        self.assertEqual(cylp_model.constraints[5].varCoefs[cylp_model.variables[5]], 1)
        self.assertEqual(cylp_model.constraints[5].varCoefs[cylp_model.variables[6]], -1)

    def test_solve_lp(self):
        model = self.create_lp()
        self.assert_lp(model, 'cbc', check_reduced_costs=False)

    def test_solve_milp(self):
        model = self.create_milp()
        self.assert_milp(model, 'cbc', status_message='optimal', check_duals=True)

    def test_supported_variable_types(self):
        inf = getCoinInfinity()

        model = conv_opt.Model()

        binary_var = conv_opt.Variable(name='binary_var', type=conv_opt.VariableType.binary)
        model.variables.append(binary_var)

        integer_var = conv_opt.Variable(name='integer_var', type=conv_opt.VariableType.integer)
        model.variables.append(integer_var)

        continuous_var = conv_opt.Variable(name='continuous_var', type=conv_opt.VariableType.continuous)
        model.variables.append(continuous_var)

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cbc))
        cbc_model = solver_model.get_model()
        self.assertEqual(cbc_model.variables[0].isInt, True)
        self.assertEqual(cbc_model.variables[1].isInt, True)
        self.assertEqual(cbc_model.variables[2].isInt, False)

        self.assertEqual(cbc_model.variables[0].lower, 0)
        self.assertEqual(cbc_model.variables[1].lower, -inf)
        self.assertEqual(cbc_model.variables[2].lower, -inf)
        self.assertEqual(cbc_model.variables[0].upper, 1)
        self.assertEqual(cbc_model.variables[1].upper, inf)
        self.assertEqual(cbc_model.variables[2].upper, inf)

    def test_unsupported_variable_types(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='binary_var', type=None)
        model.variables.append(var)
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported variable of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cbc))

    def test_unsupported_objective_direction(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='var')
        model.variables.append(var)
        model.objective_direction = None
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported objective direction'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cbc))

    def test_unsupported_objective_term(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='var')
        model.variables.append(var)
        model.objective_terms = [None]
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported objective term of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cbc))

    def test_miqp_model(self):
        model = conv_opt.Model()

        # model
        integer_var = conv_opt.Variable(name='integer_var', type=conv_opt.VariableType.integer, upper_bound=2., lower_bound=-3.)
        model.variables.append(integer_var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.QuadraticTerm(variable_1=integer_var, variable_2=integer_var, coefficient=1.)]

        # check problem type
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported objective term of type '):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cbc))

    def test_unconstrained_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='var')
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint())
        with self.assertRaisesRegex(conv_opt.ConvOptError, 'Constraints must have at least one bound'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cbc))

    def test_unsupported_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='var')
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            None,
        ], upper_bound=0, lower_bound=0))
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported constraint term of type '):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cbc))

    @unittest.skip('Sometimes fails in Docker due a bug with `capturer`')
    def test_verbose(self):
        model = self.create_lp()

        # status
        options = conv_opt.SolveOptions(solver=conv_opt.Solver.cbc,
                                        verbosity=conv_opt.Verbosity.status)
        with capturer.CaptureOutput(merged=False, relay=False, termination_delay=1.) as captured:
            model.solve(options=options)

            self.assertRegex(captured.stdout.get_text(), ' 0  Obj -0 Primal ')
            self.assertEqual(captured.stderr.get_text(), '')

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.cbc,
                                        verbosity=conv_opt.Verbosity.error)

        # off
        options = conv_opt.SolveOptions(solver=conv_opt.Solver.cbc,
                                        verbosity=conv_opt.Verbosity.off)
        with capturer.CaptureOutput(merged=False, relay=False, termination_delay=1.) as captured:
            model.solve(options=options)

            self.assertEqual(captured.stdout.get_text(), '')
            self.assertEqual(captured.stderr.get_text(), '')

    def test_presolve_on(self):
        model = self.create_lp()
        self.assert_lp(model, 'cbc', presolve=conv_opt.Presolve.on, check_reduced_costs=False)

    def test_presolve_off(self):
        model = self.create_lp()
        self.assert_lp(model, 'cbc', presolve=conv_opt.Presolve.off, check_reduced_costs=False)

    def test_unsupported_presolve(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='var')
        model.variables.append(var)
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported presolve mode '):
            model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cbc, presolve=None))

    def test_tune(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='var', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported tuning mode '):
            model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cbc, tune=True))

    def test_infeasible(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='var', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)

        cons = conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], lower_bound=3., upper_bound=5., name='constraint')
        model.constraints.append(cons)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cbc))
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

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cbc))
        self.assertEqual(result.status_code, conv_opt.StatusCode.other)
        numpy.testing.assert_equal(result.value, numpy.nan)
        numpy.testing.assert_array_equal(result.primals, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.reduced_costs, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.duals, numpy.array([numpy.nan]))

    def test_export(self):
        self.assert_export('lp', conv_opt.Solver.cbc)
        self.assert_export('mps', conv_opt.Solver.cbc)
