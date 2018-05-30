""" Tests for the Mosek solver

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

if conv_opt.Solver.mosek in conv_opt.ENABLED_SOLVERS:
    import mosek


@unittest.skipUnless(conv_opt.Solver.mosek in conv_opt.ENABLED_SOLVERS, 'MOSEK Optimizer is not installed')
class MosekTestCase(SolverTestCase):

    def test_solve_lp(self):
        model = self.create_lp()
        self.assert_lp(model, 'mosek', status_message='')

    def test_solve_milp(self):
        model = self.create_milp()
        self.assert_milp(model, 'mosek', status_message='')

    def test_solve_qp(self):
        model = self.create_qp()
        self.assert_qp(model, 'mosek', presolve=conv_opt.Presolve.off, status_message='')

    def test_solve_qp_2(self):
        model = self.create_qp_2()
        self.assert_qp_2(model, 'mosek')

    def test_solve_qp_3(self):
        model = self.create_qp_3()
        self.assert_qp_3(model, 'mosek')

    def test_solve_qp_4(self):
        model = self.create_qp_4()
        self.assert_qp_4(model, 'mosek')

    def test_solve_miqp(self):
        model = self.create_miqp()
        self.assert_miqp(model, 'mosek', presolve=conv_opt.Presolve.off, status_message='')

    def test_supported_variable_types(self):
        model = conv_opt.Model()

        binary_var = conv_opt.Variable(name='binary_var', type=conv_opt.VariableType.binary)
        model.variables.append(binary_var)

        integer_var = conv_opt.Variable(name='integer_var', type=conv_opt.VariableType.integer)
        model.variables.append(integer_var)

        continuous_var = conv_opt.Variable(name='continuous_var', type=conv_opt.VariableType.continuous)
        model.variables.append(continuous_var)

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek))
        mosek_model = solver_model.get_model()
        self.assertEqual(mosek_model.getvartype(0), mosek.variabletype.type_int)
        self.assertEqual(mosek_model.getvartype(1), mosek.variabletype.type_int)
        self.assertEqual(mosek_model.getvartype(2), mosek.variabletype.type_cont)

    def test_unsupported_variable_types(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='binary_var', type=None)
        model.variables.append(var)
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported variable of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek))

    def test_fixed_variable(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='var', lower_bound=0., upper_bound=0.0)
        model.variables.append(var)
        mosek_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek)).get_model()
        self.assertEqual(mosek_model.getvarbound(0)[0], mosek.boundkey.fx)

    def test_unsupported_objective_direction(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_direction = None
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective direction'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek))

    def test_unsupported_objective_term(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_terms = [None]
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective term of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek))

    def test_miqp_model(self):
        model = conv_opt.Model()

        # model
        integer_var = conv_opt.Variable(name='integer_var', type=conv_opt.VariableType.integer, upper_bound=2., lower_bound=-3.)
        model.variables.append(integer_var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.QuadraticTerm(variable_1=integer_var, variable_2=integer_var, coefficient=1.)]

        # check problem type
        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek))
        mosek_model = solver_model.get_model()
        self.assertEqual(mosek_model.getprobtype(), mosek.problemtype.qo)

    def test_unconstrained_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint())

        mosek_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek)).get_model()
        self.assertEqual(mosek_model.getconbound(0)[0], mosek.boundkey.fr)

    def test_unsupported_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            None,
        ], upper_bound=0, lower_bound=0))
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported constraint term of type '):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek))

    def test_verbose(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.mosek,
                                        verbosity=conv_opt.Verbosity.status)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)

            self.assertEqual(captured.stdout.get_text(), '')
            self.assertEqual(captured.stderr.get_text(), '')

    def test_unsupported_presolve(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported presolve mode '):
            model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek, presolve=None))

    def test_tune(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)
        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]
        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek, tune=True))
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

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek))
        self.assertEqual(result.status_code, conv_opt.StatusCode.infeasible)
        numpy.testing.assert_equal(result.value, numpy.nan)
        numpy.testing.assert_array_equal(result.primals, numpy.array([numpy.nan]))
        numpy.testing.assert_array_equal(result.reduced_costs, numpy.array([numpy.nan]))
        numpy.testing.assert_array_equal(result.duals, numpy.array([numpy.nan]))

    def test_infeasible_milp(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2., type=conv_opt.VariableType.integer)
        model.variables.append(var)

        cons = conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=var, coefficient=1.)
        ], lower_bound=3., upper_bound=5., name='constraint')
        model.constraints.append(cons)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek))
        self.assertEqual(result.status_code, conv_opt.StatusCode.other)
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

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek))
        self.assertEqual(result.status_code, conv_opt.StatusCode.other)
        numpy.testing.assert_equal(result.value, numpy.nan)
        numpy.testing.assert_array_equal(result.primals, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.reduced_costs, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.duals, numpy.array([numpy.nan]))

    def test_unbounded_milp(self):
        model = conv_opt.Model()

        var_1 = conv_opt.Variable(name='var_1', type=conv_opt.VariableType.integer)
        model.variables.append(var_1)

        var_2 = conv_opt.Variable(name='var_2')
        model.variables.append(var_2)

        cons = conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=var_2, coefficient=1.)
        ], lower_bound=3., upper_bound=5., name='constraint')
        model.constraints.append(cons)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var_1, coefficient=1.)]

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek))
        self.assertEqual(result.status_code, conv_opt.StatusCode.other)
        numpy.testing.assert_equal(result.value, numpy.nan)
        numpy.testing.assert_array_equal(result.primals, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.reduced_costs, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.duals, numpy.array([numpy.nan]))

    def test_infeasible_qp(self):
        model = self.create_qp()
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=model.variables[0], coefficient=1.)
        ], upper_bound=-1.))
        results = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek))
        self.assertEqual(results.status_code, conv_opt.StatusCode.infeasible)

    def test_infeasible_qp_2(self):
        model = self.create_qp_2()
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=model.variables[0], coefficient=1.)
        ], upper_bound=2.))
        results = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek))
        self.assertEqual(results.status_code, conv_opt.StatusCode.infeasible)

    def test_unbounded_qp(self):
        model = self.create_qp()
        model.variables[0].upper_bound = None
        results = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.mosek))
        self.assertEqual(results.status_code, conv_opt.StatusCode.other)

    def test_export(self):
        self.assert_export('cbf', conv_opt.Solver.mosek)
        self.assert_export('jtask', conv_opt.Solver.mosek)
        self.assert_export('lp', conv_opt.Solver.mosek)
        self.assert_export('mps', conv_opt.Solver.mosek)
        self.assert_export('opf', conv_opt.Solver.mosek)
        self.assert_export('task', conv_opt.Solver.mosek)
        self.assert_export('xml', conv_opt.Solver.mosek)
