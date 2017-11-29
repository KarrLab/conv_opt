""" Tests for the SciPy solver

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2017-11-28
:Copyright: 2017, Karr Lab
:License: MIT
"""

from ..test_core import SolverTestCase
import attrdict
import conv_opt
import numpy
import mock
import unittest


class ScipyLinprogTestCase(SolverTestCase):

    def test_solve_lp(self):
        model = self.create_lp()
        self.assert_lp(model, 'scipy', status_message='Optimization terminated successfully.',
                       check_reduced_costs=False, check_duals=False)

    def test_unsupported_model_type(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='var')
        model.variables.append(var)

        with mock.patch.object(conv_opt.Model, 'get_type', return_value=None):
            with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported model type '):
                model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy))

    def test_unsupported_variable_types(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='binary_var', type=None)
        model.variables.append(var)

        with mock.patch.object(conv_opt.Model, 'get_type', return_value=conv_opt.ModelType.lp):
            with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported variable of type'):
                model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy))

    def test_unsupported_objective_direction(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_direction = None
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective direction'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy))

    def test_unsupported_objective_term(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_terms = [None]
        with mock.patch.object(conv_opt.Model, 'get_type', return_value=conv_opt.ModelType.lp):
            with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective term of type'):
                model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy))

    def test_unconstrained_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint())
        with self.assertRaisesRegexp(conv_opt.ConvOptError, 'Constraints must have at least one bound'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy))

    def test_unsupported_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            None,
        ], upper_bound=0, lower_bound=0))
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported constraint term of type '):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy))

    def test_unsupported_presolve(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='var')
        model.variables.append(var)
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported presolve mode '):
            model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy, presolve=None))

    def test_tune(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='var', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported tuning mode '):
            model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy, tune=True))

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

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy))
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

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy))
        self.assertEqual(result.status_code, conv_opt.StatusCode.other)
        numpy.testing.assert_equal(result.value, numpy.nan)
        numpy.testing.assert_array_equal(result.primals, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.reduced_costs, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.duals, numpy.array([numpy.nan]))


class ScipyMinimizeTestCase(SolverTestCase):

    def test_solve_qp(self):
        model = self.create_qp()
        self.assert_qp(model, 'scipy', presolve=conv_opt.Presolve.off,
                       status_message='Optimization terminated successfully.')
        self.assert_qp(model, 'scipy', presolve=conv_opt.Presolve.off,
                       solver_options=attrdict.AttrDict(method='COBYLA'),
                       status_message='Optimization terminated successfully.')
        self.assert_qp(model, 'scipy', presolve=conv_opt.Presolve.off,
                       solver_options=attrdict.AttrDict(method='SLSQP'),
                       status_message='Optimization terminated successfully.')

    def test_solve_qp_2(self):
        model = self.create_qp_2()
        self.assert_qp_2(model, 'scipy',
                         solver_options=attrdict.AttrDict(method='COBYLA'))
        self.assert_qp_2(model, 'scipy',
                         solver_options=attrdict.AttrDict(method='SLSQP'))

    def test_solve_qp_3(self):
        model = self.create_qp_3()
        self.assert_qp_3(model, 'scipy',
                         solver_options=attrdict.AttrDict(method='COBYLA'))
        self.assert_qp_3(model, 'scipy',
                         solver_options=attrdict.AttrDict(method='SLSQP'))

    def test_solve_qp_4(self):
        model = self.create_qp_4()
        self.assert_qp_4(model, 'scipy',
                         solver_options=attrdict.AttrDict(method='COBYLA'))
        self.assert_qp_4(model, 'scipy',
                         solver_options=attrdict.AttrDict(method='SLSQP'))

    def test_unsupported_variable_types(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='var', type=None)
        model.variables.append(var)

        model.objective_terms = [conv_opt.QuadraticTerm(variable_1=var, variable_2=var, coefficient=1.)]

        with mock.patch.object(conv_opt.Model, 'get_type', return_value=conv_opt.ModelType.qp):
            with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported variable of type'):
                model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy))

    def test_unsupported_objective_direction(self):
        model = conv_opt.Model()

        var = conv_opt.Variable()
        model.variables.append(var)

        model.objective_direction = None
        model.objective_terms = [conv_opt.QuadraticTerm(variable_1=var, variable_2=var, coefficient=1.)]

        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective direction'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy))

    def test_unsupported_objective_term(self):
        model = conv_opt.Model()

        var = conv_opt.Variable()
        model.variables.append(var)

        model.objective_terms = [None, conv_opt.QuadraticTerm(variable_1=var, variable_2=var, coefficient=1.)]

        with mock.patch.object(conv_opt.Model, 'get_type', return_value=conv_opt.ModelType.qp):
            with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective term of type'):
                model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy))

    def test_unconstrained_constraint(self):
        model = conv_opt.Model()

        var = conv_opt.Variable()
        model.variables.append(var)

        model.objective_terms = [conv_opt.QuadraticTerm(variable_1=var, variable_2=var, coefficient=1.)]

        model.constraints.append(conv_opt.Constraint())

        with self.assertRaisesRegexp(conv_opt.ConvOptError, 'Constraints must have at least one bound'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy))

    def test_unsupported_constraint(self):
        model = conv_opt.Model()

        var = conv_opt.Variable()
        model.variables.append(var)

        model.objective_terms = [conv_opt.QuadraticTerm(variable_1=var, variable_2=var, coefficient=1.)]

        model.constraints.append(conv_opt.Constraint([
            None,
        ], upper_bound=0, lower_bound=0))

        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported constraint term of type '):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy))

    def test_infeasible_qp_2_cobyla(self):
        model = self.create_qp_2()
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=model.variables[0], coefficient=1.)
        ], upper_bound=2.))

        results = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy,
                                                            solver_options=attrdict.AttrDict(method='COBYLA')))
        self.assertEqual(results.status_code, conv_opt.StatusCode.infeasible)

    @unittest.expectedFailure
    def test_infeasible_qp_2_sqlsqp(self):
        model = self.create_qp_2()
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=model.variables[0], coefficient=1.)
        ], upper_bound=2.))
        results = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy,
                                                            solver_options=attrdict.AttrDict(method='SLSQP')))
        self.assertEqual(results.status_code, conv_opt.StatusCode.infeasible)

    @unittest.expectedFailure
    def test_unbounded_qp_cobyla(self):
        model = self.create_qp()
        model.variables[0].upper_bound = None
        results = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy,
                                                            solver_options=attrdict.AttrDict(method='COBYLA')))
        self.assertEqual(results.status_code, conv_opt.StatusCode.other)

    @unittest.expectedFailure
    def test_unbounded_qp_sqlsqp(self):
        model = self.create_qp()
        model.variables[0].upper_bound = None
        results = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.scipy,
                                                            solver_options=attrdict.AttrDict(method='SLSQP')))
        self.assertEqual(results.status_code, conv_opt.StatusCode.other)

    def test_unsupported_method(self):
        model = self.create_qp()
        with mock.patch('scipy.optimize.minimize', return_value=None):
            with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported solver method '):
                self.assert_qp(model, 'scipy', presolve=conv_opt.Presolve.off,
                               solver_options=attrdict.AttrDict(method='__none__'))
