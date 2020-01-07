""" Tests for the CVXOPT solver

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

if conv_opt.Solver.cvxopt in conv_opt.ENABLED_SOLVERS:
    import cvxpy


@unittest.skipUnless(conv_opt.Solver.cvxopt in conv_opt.ENABLED_SOLVERS, 'CVXOPT is not installed')
class CvxoptTestCase(SolverTestCase):

    def test_convert(self):
        model = self.create_lp()
        cvxpy_model = model.convert(conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt)).get_model()

        cvxpy_vars = {var.name(): var for var in cvxpy_model.variables()}

        self.assertEqual(set(cvxpy_vars.keys()), set(['ex_a', 'r1', 'r2', 'r3', 'r4', 'biomass_production', 'ex_biomass']))
        #self.assertEqual(set([v.lb for v in glpk.variables]), set([0.]))
        #self.assertEqual([v.ub for v in glpk.variables], [1., None, None, None, None, None, None])
        #self.assertEqual(set([v.type for v in glpk.variables]), set(['continuous']))

        self.assertIsInstance(cvxpy_model.objective, cvxpy.Maximize)
        self.assertEqual(cvxpy_model.objective.variables(), [cvxpy_vars['biomass_production']])
        self.assertEqual([c._value for c in cvxpy_model.objective.constants()], [1])

        self.assertEqual(set(cvxpy_model.constraints[0].variables()), set([cvxpy_vars['ex_a'], cvxpy_vars['r1']]))
        self.assertEqual(set(cvxpy_model.constraints[1].variables()), set([cvxpy_vars['r1'], cvxpy_vars['r2'], cvxpy_vars['r3']]))
        self.assertEqual(set(cvxpy_model.constraints[2].variables()), set([cvxpy_vars['r2']]))
        self.assertEqual(set(cvxpy_model.constraints[3].variables()), set([cvxpy_vars['r2'], cvxpy_vars['r4']]))
        self.assertEqual(set(cvxpy_model.constraints[4].variables()), set(
            [cvxpy_vars['r3'], cvxpy_vars['r4'], cvxpy_vars['biomass_production']]))
        self.assertEqual(set(cvxpy_model.constraints[5].variables()), set([cvxpy_vars['biomass_production'], cvxpy_vars['ex_biomass']]))

        self.assertEqual(sorted([c._value for c in cvxpy_model.constraints[0].constants()]), [-1, 0, 1])
        self.assertEqual(sorted([c._value for c in cvxpy_model.constraints[1].constants()]), [-1, -1, 0, 1])
        self.assertEqual(sorted([c._value for c in cvxpy_model.constraints[2].constants()]), [0, 1])
        self.assertEqual(sorted([c._value for c in cvxpy_model.constraints[3].constants()]), [-1, 0, 1])
        self.assertEqual(sorted([c._value for c in cvxpy_model.constraints[4].constants()]), [-1, 0, 1, 2])
        self.assertEqual(sorted([c._value for c in cvxpy_model.constraints[5].constants()]), [-1, 0, 1])

        self.assertEqual(len(cvxpy_model.constraints), 6 + 7 + 4)

    def test_solve_lp(self):
        model = self.create_lp()

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))
        cvxopt_model = solver_model.get_model()
        self.assertTrue(cvxopt_model.objective.args[0].is_affine())

        self.assert_lp(model, 'cvxopt', check_reduced_costs=False, check_duals=False)

    def test_solve_qp(self):
        # example fails because CVXPY doesn't identify the model as :math:`\min{(ex_a - r_1)^2 - biomass}`
        model = self.create_qp()

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))
        cvxopt_model = solver_model.get_model()
        self.assertFalse(cvxopt_model.objective.args[0].is_affine())

        with self.assertRaises(cvxpy.error.DCPError):
            self.assert_qp(model, 'cvxopt', presolve=conv_opt.Presolve.off, check_reduced_costs=False, check_duals=False)

    def test_solve_qp_2(self):
        # example fails because CVXPY doesn't identify the model as :math:`\min{v_1 - v_2)^2}`
        model = self.create_qp_2()
        with self.assertRaises(cvxpy.error.DCPError):
            self.assert_qp_2(model, 'cvxopt')

    def test_solve_qp_3(self):
        # example fails because CVXPY doesn't identify the model as :math:`\min{v_1 - v_2)^2}`
        model = self.create_qp_3()
        with self.assertRaises(cvxpy.error.DCPError):
            self.assert_qp_3(model, 'cvxopt')

    def test_solve_qp_4(self):
        # example fails because CVXPY doesn't identify the model as :math:`\min{v_1 - v_2)^2}`
        model = self.create_qp_4()
        with self.assertRaises(cvxpy.error.DCPError):
            self.assert_qp_4(model, 'cvxopt')

    def test_duplicate_variable_names(self):
        model = conv_opt.Model()

        var_1 = conv_opt.Variable(name='var')
        model.variables.append(var_1)

        var_2 = conv_opt.Variable(name='var')
        model.variables.append(var_2)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [
            conv_opt.LinearTerm(variable=var_1, coefficient=1.),
            conv_opt.LinearTerm(variable=var_2, coefficient=1.),
        ]

        with self.assertRaisesRegex(conv_opt.ConvOptError, 'Variables must have unique names'):
            solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))

    def test_supported_variable_types(self):
        model = conv_opt.Model()

        continuous_var = conv_opt.Variable(name='continuous_var', type=conv_opt.VariableType.continuous)
        model.variables.append(continuous_var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [
            conv_opt.LinearTerm(variable=continuous_var, coefficient=1.),
        ]

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))
        cvxopt_model = solver_model.get_model()
        cvxopt_vars = cvxopt_model.variables()
        self.assertIsInstance(cvxopt_vars[0], cvxpy.Variable)

    def test_unsupported_variable_types(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='var', type=conv_opt.VariableType.binary)
        model.variables.append(var)
        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported variable of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))

        model = conv_opt.Model()
        var = conv_opt.Variable(name='var', type=conv_opt.VariableType.integer)
        model.variables.append(var)
        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported variable of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))

        model = conv_opt.Model()
        var = conv_opt.Variable(name='var', type=None)
        model.variables.append(var)
        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported variable of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))

    def test_fixed_variable(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='var', lower_bound=1., upper_bound=1.)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.minimize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=-1.)]

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))
        cvxpy_model = solver_model.get_model()

        self.assertEqual(cvxpy_model.constraints[0].variables(), cvxpy_model.variables())
        self.assertEqual(len(cvxpy_model.constraints[0].constants()), 1)
        self.assertEqual(cvxpy_model.constraints[0].constants()[0]._value, 1.)

    def test_minimize(self):
        model = conv_opt.Model()

        var = conv_opt.Variable()
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.minimize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=-1.)]

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))
        cvxpy_model = solver_model.get_model()
        self.assertIsInstance(cvxpy_model.objective, cvxpy.Minimize)

    def test_unsupported_objective_direction(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_direction = None
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported objective direction'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))

    def test_unsupported_objective_term(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_terms = [None]
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported objective term of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))

    def test_quadratic_objective_with_multiple_terms(self):
        model = conv_opt.Model()

        var_1 = conv_opt.Variable(name='var_1')
        model.variables.append(var_1)

        var_2 = conv_opt.Variable(name='var_2')
        model.variables.append(var_2)

        model.objective_direction = conv_opt.ObjectiveDirection.minimize
        model.objective_terms = [
            conv_opt.QuadraticTerm(variable_1=var_1, variable_2=var_1, coefficient=1.),
            conv_opt.QuadraticTerm(variable_1=var_2, variable_2=var_2, coefficient=1.),
        ]

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))
        cvxpy_model = solver_model.get_model()
        with self.assertRaises(AssertionError):
            # CVX doesn't correctly represent quadratic objective as function of two variables
            self.assertEqual(len(cvxpy_model.objective.variables()), 2)

    def test_unconstrained_constraint(self):
        model = conv_opt.Model()

        var = conv_opt.Variable()
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=-1.)]

        model.constraints.append(conv_opt.Constraint())

        with self.assertRaisesRegex(conv_opt.ConvOptError, 'Constraints must have at least one bound'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))

    def test_unsupported_constraint(self):
        model = conv_opt.Model()

        var = conv_opt.Variable()
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=-1.)]

        model.constraints.append(conv_opt.Constraint([
            None,
        ], upper_bound=0, lower_bound=0))
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported constraint term of type '):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))

    def test_verbose(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt,
                                        verbosity=conv_opt.Verbosity.status)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)

            self.assertRegex(captured.stdout.get_text(), '^     pcost       dcost')
            self.assertEqual(captured.stderr.get_text(), '')

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt,
                                        verbosity=conv_opt.Verbosity.off)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)

            self.assertEqual(captured.stdout.get_text(), '')
            self.assertEqual(captured.stderr.get_text(), '')

    def test_unsupported_presolve(self):
        model = conv_opt.Model()

        var = conv_opt.Variable()
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=-1.)]

        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported presolve mode '):
            model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt, presolve=None))

    def test_tune(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported tuning mode '):
            model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt, tune=True))

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

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))
        self.assertEqual(result.status_code, conv_opt.StatusCode.infeasible)
        numpy.testing.assert_equal(result.value, numpy.nan)
        numpy.testing.assert_array_equal(result.primals, numpy.array([numpy.nan]))
        numpy.testing.assert_array_equal(result.reduced_costs, numpy.array([numpy.nan]))
        numpy.testing.assert_array_equal(result.duals, numpy.array([numpy.nan]))

    def test_unbounded(self):
        model = conv_opt.Model()

        var_1 = conv_opt.Variable(name='var_1', lower_bound=2.)
        model.variables.append(var_1)

        var_2 = conv_opt.Variable(name='var_2')
        model.variables.append(var_2)

        cons = conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=var_2, coefficient=1.)
        ], lower_bound=3., upper_bound=5., name='constraint')
        model.constraints.append(cons)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [
            conv_opt.LinearTerm(variable=var_1, coefficient=1.),
            conv_opt.LinearTerm(variable=var_2, coefficient=1.),
        ]

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))
        self.assertEqual(result.status_code, conv_opt.StatusCode.other)
        numpy.testing.assert_equal(result.value, numpy.nan)
        numpy.testing.assert_array_equal(result.primals, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.reduced_costs, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.duals, numpy.array([numpy.nan]))

    def test_solver_error(self):
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
        model.objective_terms = [
            conv_opt.LinearTerm(variable=var_1, coefficient=1.),
            conv_opt.LinearTerm(variable=var_2, coefficient=1.),
        ]

        with self.assertRaisesRegex(cvxpy.SolverError, ''):
            model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cvxopt))
