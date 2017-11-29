""" Tests for the quadprog solver

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2017-11-28
:Copyright: 2017, Karr Lab
:License: MIT
"""

from ..test_core import SolverTestCase
import conv_opt
import numpy
import unittest


class QuadprogTestCase(SolverTestCase):

    @unittest.expectedFailure
    def test_solve_qp(self):
        # quadprog is not able to solve this problem and doesn't correctly report an error
        model = self.create_qp()
        self.assert_qp(model, 'quadprog', status_message='', check_reduced_costs=True, check_duals=False)

    def test_solve_qp_2(self):
        model = self.create_qp_2()
        self.assert_qp_2(model, 'quadprog')

    def test_solve_qp_3(self):
        model = self.create_qp_3()
        self.assert_qp_3(model, 'quadprog')

    def test_solve_qp_4(self):
        model = self.create_qp_4()
        self.assert_qp_4(model, 'quadprog')

    def test_supported_variable_types(self):
        model = conv_opt.Model()

        continuous_var = conv_opt.Variable(name='continuous_var', type=conv_opt.VariableType.continuous)
        model.variables.append(continuous_var)

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.quadprog))
        quadprog_model = solver_model.get_model()
        self.assertEqual(quadprog_model['a'].size, 1)

    def test_unsupported_variable_types(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='binary_var', type=conv_opt.VariableType.integer)
        model.variables.append(var)
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported variable of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.quadprog))

    def test_unsupported_objective_direction(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_direction = None
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective direction'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.quadprog))

    def test_unsupported_objective_term(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_terms = [None]
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective term of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.quadprog))

    def test_variable_bounds(self):
        model = conv_opt.Model()

        var_1 = conv_opt.Variable(name='var_1', lower_bound=-1.)
        model.variables.append(var_1)

        var_1 = conv_opt.Variable(name='var_2', upper_bound=1.)
        model.variables.append(var_1)

        var_1 = conv_opt.Variable(name='var_3', lower_bound=-0.5, upper_bound=0.5)
        model.variables.append(var_1)

        var_1 = conv_opt.Variable(name='var_4', lower_bound=0.25, upper_bound=0.25)
        model.variables.append(var_1)

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.quadprog))
        quadprog_model = solver_model.get_model()

        numpy.testing.assert_array_equal(quadprog_model['C'].transpose(), numpy.array([
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [0., 0., 1., 0.],
        ]))
        numpy.testing.assert_array_equal(quadprog_model['b'], numpy.array([0.25, -1., -1., -0.5, -0.5]))
        numpy.testing.assert_array_equal(quadprog_model['meq'], 1)

    def test_unconstrained_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint())
        with self.assertRaisesRegexp(conv_opt.ConvOptError, 'Constraints must have at least one bound'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.quadprog))

    def test_multiple_types_of_constraints(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='var')
        model.variables.append(var)

        model.constraints.append(conv_opt.Constraint([
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], lower_bound=-1.))
        model.constraints.append(conv_opt.Constraint([
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], upper_bound=1.))
        model.constraints.append(conv_opt.Constraint([
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], lower_bound=-0.5, upper_bound=0.5))
        model.constraints.append(conv_opt.Constraint([
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], lower_bound=0.25, upper_bound=0.25))

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.quadprog))
        quadprog_model = solver_model.get_model()

        numpy.testing.assert_array_equal(quadprog_model['C'].transpose(), numpy.array([1., 1., -1., -1., 1.]).reshape((5, 1)))
        numpy.testing.assert_array_equal(quadprog_model['b'], numpy.array([0.25, -1., -1., -0.5, -0.5]))
        numpy.testing.assert_array_equal(quadprog_model['meq'], 1)

    def test_unsupported_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            None,
        ], upper_bound=0, lower_bound=0))
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported constraint term of type '):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.quadprog))

    def test_unsupported_presolve(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported presolve mode '):
            model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.quadprog, presolve=None))

    def test_tune(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)
        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported tuning mode '):
            result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.quadprog, tune=True))

    def test_infeasible_qp(self):
        model = self.create_qp()
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=model.variables[0], coefficient=1.)
        ], upper_bound=-1.))
        results = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.quadprog))
        self.assertEqual(results.status_code, conv_opt.StatusCode.other)
