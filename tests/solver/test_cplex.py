""" Tests for IBM CPLEX solver

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2017-11-27
:Copyright: 2017, Karr Lab
:License: MIT
"""

from ..test_core import SolverTestCase
import capturer
import conv_opt
import numpy
import unittest

if conv_opt.Solver.cplex in conv_opt.ENABLED_SOLVERS:
    import cplex


@unittest.skipUnless(conv_opt.Solver.cplex in conv_opt.ENABLED_SOLVERS, 'IBM CPLEX is not installed')
class TestCplex(SolverTestCase):

    def test_convert(self):
        model = self.create_lp()
        cplex_model = model.convert(conv_opt.SolveOptions(solver=conv_opt.Solver.cplex)).get_model()

        self.assertEqual(cplex_model.get_problem_name(), 'test-lp')

        self.assertEqual(cplex_model.get_problem_type(), cplex_model.problem_type.LP)

        self.assertEqual(cplex_model.variables.get_names(), ['ex_a', 'r1', 'r2', 'r3', 'r4', 'biomass_production', 'ex_biomass'])
        self.assertEqual(cplex_model.variables.get_lower_bounds(), [-cplex.infinity, 0., 0., 0., 0., 0., 0.])
        self.assertEqual(cplex_model.variables.get_upper_bounds(),
                         [1., cplex.infinity, cplex.infinity, cplex.infinity, cplex.infinity, cplex.infinity, cplex.infinity])
        self.assertEqual(cplex_model.variables.get_num(), 7)
        self.assertEqual(cplex_model.variables.get_num_binary(), 0)
        self.assertEqual(cplex_model.variables.get_num_integer(), 0)
        self.assertEqual(cplex_model.variables.get_num_semiinteger(), 0)
        self.assertEqual(cplex_model.variables.get_num_semicontinuous(), 0)

        self.assertEqual(cplex_model.objective.get_sense(), cplex_model.objective.sense.maximize)
        self.assertEqual(cplex_model.objective.get_linear(), [0., 0., 0., 0., 0., 1., 0.])
        self.assertEqual(cplex_model.objective.get_quadratic(), [])

        self.assertEqual(cplex_model.linear_constraints.get_names(), [
                         'a', 'b', 'c', 'd', 'e', 'biomass', 'upper_bound', 'lower_bound', 'range_bound'])
        self.assertEqual(cplex_model.linear_constraints.get_senses(), ['E', 'E', 'E', 'E', 'E', 'E', 'L', 'G', 'R'])
        self.assertEqual(cplex_model.linear_constraints.get_rhs(), [0., 0., 0., 0., 0., 0., 1., -1., -10.])
        self.assertEqual(cplex_model.linear_constraints.get_range_values(), [0., 0., 0., 0., 0., 0., 0., 0., 20.])
        self.assertEqual(cplex_model.linear_constraints.get_num_nonzeros(), 16)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(0, 0), 1)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(0, 1), -1)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(1, 1), 1)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(1, 2), -1)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(1, 3), -1)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(2, 2), 1)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(3, 2), 1)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(3, 4), -1)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(4, 3), 2)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(4, 4), 1)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(4, 5), -1)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(5, 5), 1)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(5, 6), -1)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(6, 0), -1)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(7, 5), 1)
        self.assertEqual(cplex_model.linear_constraints.get_coefficients(8, 1), 1)

        self.assertEqual(cplex_model.quadratic_constraints.get_num(), 0)

    def test_solve_lp(self):
        model = self.create_lp()
        self.assert_lp(model, 'cplex')

    def test_solve_milp(self):
        model = self.create_milp()
        self.assert_milp(model, 'cplex', status_message='integer optimal solution')

    def test_solve_qp(self):
        model = self.create_qp()
        self.assert_qp(model, 'cplex')

    def test_solve_miqp(self):
        model = self.create_miqp()
        self.assert_miqp(model, 'cplex', status_message='integer optimal solution')

    def test_solve_qp_2(self):
        model = self.create_qp_2()
        self.assert_qp_2(model, 'cplex')

    def test_solve_qp_3(self):
        model = self.create_qp_3()
        self.assert_qp_3(model, 'cplex')

    def test_solve_qp_4(self):
        model = self.create_qp_4()
        self.assert_qp_4(model, 'cplex')

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

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cplex))
        cplex_model = solver_model.get_model()
        self.assertEqual(cplex_model.variables.get_types(0), cplex_model.variables.type.binary)
        self.assertEqual(cplex_model.variables.get_types(1), cplex_model.variables.type.integer)
        self.assertEqual(cplex_model.variables.get_types(2), cplex_model.variables.type.continuous)
        self.assertEqual(cplex_model.variables.get_types(3), cplex_model.variables.type.semi_integer)
        self.assertEqual(cplex_model.variables.get_types(4), cplex_model.variables.type.semi_continuous)

    def test_unsupported_variable_types(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='binary_var', type=None)
        model.variables.append(var)
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported variable of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cplex))

    def test_unsupported_objective_direction(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_direction = None
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective direction'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cplex))

    def test_unsupported_objective_term(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_terms = [None]
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective term of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cplex))

    def test_miqp_model(self):
        model = conv_opt.Model()

        # model
        integer_var = conv_opt.Variable(name='integer_var', type=conv_opt.VariableType.integer, upper_bound=2., lower_bound=-3.)
        model.variables.append(integer_var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.QuadraticTerm(variable_1=integer_var, variable_2=integer_var, coefficient=1.)]

        # check problem type
        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cplex))
        cplex_model = solver_model.get_model()
        self.assertEqual(cplex_model.get_problem_type(), cplex_model.problem_type.MIQP)

    def test_unconstrained_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint())
        with self.assertRaisesRegexp(conv_opt.ConvOptError, 'Constraints must have at least one bound'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cplex))

    def test_unsupported_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            None,
        ], upper_bound=0, lower_bound=0))
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported constraint term of type '):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cplex))

    def test_verbose(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.cplex,
                                        verbosity=conv_opt.Verbosity.status)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)

            self.assertRegex(captured.stdout.get_text(), '^CPXPARAM_Preprocessing_Presolve')
            self.assertEqual(captured.stderr.get_text(), '')

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.cplex,
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
            model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cplex, presolve=None))

    def test_tune(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)
        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]
        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cplex, tune=True))
        self.assertEqual(result.status_code, conv_opt.StatusCode.optimal)
        self.assertEqual(result.value, 2.)

    def test_set_solver_options(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)
        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]
        options = conv_opt.SolveOptions(
            solver=conv_opt.Solver.cplex,
            solver_options={
                'cplex': {
                    'parameters': {
                        'emphasis': {
                            'numerical': 1,
                        },
                        'read': {
                            'scale': 0,
                        },
                    },
                },
            })
        cplex_model = model.convert(options=options)

        cplex_model.solve()
        self.assertEqual(cplex_model._model.parameters.emphasis.numerical.get(), 1)
        self.assertEqual(cplex_model._model.parameters.read.scale.get(), 0)

        options.solver_options['cplex']['parameters']['emphasis']['numerical'] = 0
        options.solver_options['cplex']['parameters']['read']['scale'] = 1
        cplex_model.set_solver_options()
        self.assertEqual(cplex_model._model.parameters.emphasis.numerical.get(), 0)
        self.assertEqual(cplex_model._model.parameters.read.scale.get(), 1)

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

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cplex))
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

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.cplex))
        self.assertEqual(result.status_code, conv_opt.StatusCode.other)
        numpy.testing.assert_equal(result.value, numpy.nan)
        numpy.testing.assert_array_equal(result.primals, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.reduced_costs, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.duals, numpy.array([numpy.nan]))

    def test_export(self):
        self.assert_export('alp', conv_opt.Solver.cplex)
        self.assert_export('dpe', conv_opt.Solver.cplex)
        self.assert_export('dua', conv_opt.Solver.cplex)
        self.assert_export('lp', conv_opt.Solver.cplex)
        self.assert_export('mps', conv_opt.Solver.cplex)
        self.assert_export('ppe', conv_opt.Solver.cplex)
        self.assert_export('rew', conv_opt.Solver.cplex)
        self.assert_export('rlp', conv_opt.Solver.cplex)
        self.assert_export('sav', conv_opt.Solver.cplex)
