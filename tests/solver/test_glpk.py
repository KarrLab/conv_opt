""" Tests for the GLPK solver

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2017-11-28
:Copyright: 2017, Karr Lab
:License: MIT
"""

from ..test_core import SolverTestCase
import capturer
import conv_opt
import numpy
import sympy
import unittest


class GlpkTestCase(SolverTestCase):

    def test_convert(self):
        model = self.create_lp()
        glpk_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.glpk)).get_model()

        self.assertEqual(glpk_model.name, 'test-lp')

        self.assertEqual(glpk_model.variables.keys(), ['ex_a', 'r1', 'r2', 'r3', 'r4', 'biomass_production', 'ex_biomass'])
        self.assertEqual([v.lb for v in glpk_model.variables], [None, 0., 0., 0., 0., 0., 0.])
        self.assertEqual([v.ub for v in glpk_model.variables], [1., None, None, None, None, None, None])
        self.assertEqual(set([v.type for v in glpk_model.variables]), set(['continuous']))

        self.assertEqual(glpk_model.objective.direction, 'max')
        self.assertEqual(glpk_model.objective.expression, 1. * glpk_model.variables[-2])

        self.assertEqual([c.name for c in glpk_model.constraints], ['a', 'b', 'c', 'd',
                                                                    'e', 'biomass', 'upper_bound', 'lower_bound', 'range_bound'])
        self.assertEqual([c.lb for c in glpk_model.constraints], [0., 0., 0., 0., 0., 0., None, -1., -10.])
        self.assertEqual([c.ub for c in glpk_model.constraints], [0., 0., 0., 0., 0., 0., 1., None, 10.])
        self.assertEqual(sympy.simplify(glpk_model.constraints[0].expression -
                                        (1. * glpk_model.variables[0] - 1. * glpk_model.variables[1])), 0)
        self.assertEqual(sympy.simplify(glpk_model.constraints[1].expression -
                                        (1. * glpk_model.variables[1] - 1. * glpk_model.variables[2] - 1. * glpk_model.variables[3])), 0)
        self.assertEqual(sympy.simplify(glpk_model.constraints[2].expression - (1. * glpk_model.variables[2])), 0)
        self.assertEqual(sympy.simplify(glpk_model.constraints[3].expression -
                                        (1. * glpk_model.variables[2] - 1. * glpk_model.variables[4])), 0)
        self.assertEqual(sympy.simplify(glpk_model.constraints[4].expression -
                                        (2. * glpk_model.variables[3] + 1. * glpk_model.variables[4] - 1. * glpk_model.variables[5])), 0)
        self.assertEqual(sympy.simplify(glpk_model.constraints[5].expression -
                                        (1. * glpk_model.variables[5] - 1. * glpk_model.variables[6])), 0)

    def test_solve_lp(self):
        model = self.create_lp()
        self.assert_lp(model, 'glpk')

    def test_solve_milp(self):
        model = self.create_milp()
        self.assert_milp(model, 'glpk')

    def test_supported_variable_types(self):
        model = conv_opt.Model()

        binary_var = conv_opt.Variable(name='binary_var', type=conv_opt.VariableType.binary)
        model.variables.append(binary_var)

        integer_var = conv_opt.Variable(name='integer_var', type=conv_opt.VariableType.integer)
        model.variables.append(integer_var)

        continuous_var = conv_opt.Variable(name='continuous_var', type=conv_opt.VariableType.continuous)
        model.variables.append(continuous_var)

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.glpk))
        glpk_model = solver_model.get_model()
        self.assertEqual(glpk_model.variables[0].type, 'binary')
        self.assertEqual(glpk_model.variables[1].type, 'integer')
        self.assertEqual(glpk_model.variables[2].type, 'continuous')

    def test_unsupported_variable_types(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='binary_var', type=None)
        model.variables.append(var)
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported variable of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.glpk))

    def test_unsupported_objective_direction(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='var')
        model.variables.append(var)
        model.objective_direction = None
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported objective direction'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.glpk))

    def test_unsupported_objective_term(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='var')
        model.variables.append(var)
        model.objective_terms = [None]
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported objective term of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.glpk))

    def test_unsupported_quadratic_objective(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='var')
        model.variables.append(var)
        model.objective_terms = [conv_opt.QuadraticTerm(variable_1=var, variable_2=var, coefficient=1.)]
        with self.assertRaisesRegex(ValueError, '^GLPK only supports linear objectives.'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.glpk))

    def test_unconstrained_constraint(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='var',)
        model.variables.append(var)

        constraint = conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], name='constraint')
        model.constraints.append(constraint)

        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.glpk))
        glpk_model = solver_model.get_model()

        self.assertEqual(glpk_model.constraints[0].lb, None)
        self.assertEqual(glpk_model.constraints[0].ub, None)

    def test_unsupported_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='var')
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            None,
        ], upper_bound=0, lower_bound=0))
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported constraint term of type '):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.glpk))

    def test_verbose(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.glpk,
                                        verbosity=conv_opt.Verbosity.status)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)

            self.assertRegex(captured.stdout.get_text(), '^GLPK Simplex Optimizer,')
            self.assertEqual(captured.stderr.get_text(), '')

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.glpk,
                                        verbosity=conv_opt.Verbosity.off)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)

            self.assertEqual(captured.stdout.get_text(), '')
            self.assertEqual(captured.stderr.get_text(), '')

    def test_presolve_auto(self):
        model = self.create_lp()
        self.assert_lp(model, 'glpk', presolve=conv_opt.Presolve.auto)

    @unittest.skip('Skip. This is aborting. This may require a different example problem')
    def test_presolve_on(self):
        model = self.create_lp()
        self.assert_lp(model, 'glpk', presolve=conv_opt.Presolve.on)

    def test_unsupported_presolve(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='var')
        model.variables.append(var)
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported presolve mode '):
            model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.glpk, presolve=None))

    def test_tune(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)
        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]
        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.glpk, tune=True))
        self.assertEqual(result.status_code, conv_opt.StatusCode.optimal)
        self.assertEqual(result.value, 2.)

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

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.glpk))
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

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.glpk))
        self.assertEqual(result.status_code, conv_opt.StatusCode.other)
        numpy.testing.assert_equal(result.value, numpy.nan)
        numpy.testing.assert_array_equal(result.primals, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.reduced_costs, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.duals, numpy.array([numpy.nan]))
