""" Tests for the Gurobi solver

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

if conv_opt.Solver.gurobi in conv_opt.ENABLED_SOLVERS:
    import gurobipy


@unittest.skipUnless(conv_opt.Solver.gurobi in conv_opt.ENABLED_SOLVERS, 'Gurobi is not installed')
class GurobiTestCase(SolverTestCase):

    def test_convert(self):
        model = self.create_lp()
        gurobi_model = model.convert(conv_opt.SolveOptions(solver=conv_opt.Solver.gurobi)).get_model()

        self.assertEqual(gurobi_model.getAttr('ModelName'), 'test-lp')

        vars = gurobi_model.getVars()
        self.assertEqual([var.VarName for var in vars], ['ex_a', 'r1', 'r2', 'r3', 'r4', 'biomass_production', 'ex_biomass'])
        self.assertEqual([var.LB for var in vars], [-gurobipy.GRB.INFINITY, 0., 0., 0., 0., 0., 0.])
        self.assertEqual([var.UB for var in vars],
                         [1., gurobipy.GRB.INFINITY, gurobipy.GRB.INFINITY, gurobipy.GRB.INFINITY,
                          gurobipy.GRB.INFINITY, gurobipy.GRB.INFINITY, gurobipy.GRB.INFINITY])
        self.assertEqual(set([var.VType for var in vars]), set([gurobipy.GRB.CONTINUOUS]))

        self.assertEqual(gurobi_model.ModelSense, gurobipy.GRB.MAXIMIZE)
        self.assertEqual(gurobi_model.getObjective(), gurobipy.LinExpr(1., vars[-2]))

        constraints = gurobi_model.getConstrs()
        self.assertEqual([c.ConstrName for c in constraints],
                         ['a', 'b', 'c', 'd', 'e', 'biomass', 'upper_bound', 'lower_bound',
                          'range_bound__lower__', 'range_bound__upper__'])
        self.assertEqual([c.Sense for c in constraints], [
            gurobipy.GRB.EQUAL, gurobipy.GRB.EQUAL, gurobipy.GRB.EQUAL,
            gurobipy.GRB.EQUAL, gurobipy.GRB.EQUAL, gurobipy.GRB.EQUAL,
            gurobipy.GRB.LESS_EQUAL, gurobipy.GRB.GREATER_EQUAL,
            gurobipy.GRB.GREATER_EQUAL, gurobipy.GRB.LESS_EQUAL,
        ])
        self.assertEqual([c.RHS for c in constraints], [0., 0., 0., 0., 0., 0., 1., -1., -10., 10.])

    def test_solve_lp(self):
        model = self.create_lp()
        self.assert_lp(model, 'gurobi', status_message='')

    def test_solve_milp(self):
        model = self.create_milp()
        self.assert_milp(model, 'gurobi', status_message='')

    def test_solve_qp(self):
        model = self.create_qp()
        self.assert_qp(model, 'gurobi', status_message='')

    def test_solve_qp_2(self):
        model = self.create_qp_2()
        self.assert_qp_2(model, 'gurobi')

    def test_solve_qp_3(self):
        model = self.create_qp_3()
        self.assert_qp_3(model, 'gurobi')

    def test_solve_qp_4(self):
        model = self.create_qp_4()
        self.assert_qp_4(model, 'gurobi')

    def test_solve_miqp(self):
        model = self.create_miqp()
        self.assert_miqp(model, 'gurobi', status_message='')

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

        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.gurobi))
        gurobi_model = solver_model.get_model()
        self.assertEqual(gurobi_model.getVars()[0].getAttr(gurobipy.GRB.Attr.VType), gurobipy.GRB.BINARY)
        self.assertEqual(gurobi_model.getVars()[1].getAttr(gurobipy.GRB.Attr.VType), gurobipy.GRB.INTEGER)
        self.assertEqual(gurobi_model.getVars()[2].getAttr(gurobipy.GRB.Attr.VType), gurobipy.GRB.CONTINUOUS)
        self.assertEqual(gurobi_model.getVars()[3].getAttr(gurobipy.GRB.Attr.VType), gurobipy.GRB.SEMIINT)
        self.assertEqual(gurobi_model.getVars()[4].getAttr(gurobipy.GRB.Attr.VType), gurobipy.GRB.SEMICONT)

    def test_unsupported_variable_types(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='binary_var', type=None)
        model.variables.append(var)
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported variable of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.gurobi))

    def test_unsupported_objective_direction(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_direction = None
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective direction'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.gurobi))

    def test_unsupported_objective_term(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.objective_terms = [None]
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported objective term of type'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.gurobi))

    def test_miqp_model(self):
        model = conv_opt.Model()

        # model
        integer_var = conv_opt.Variable(name='integer_var', type=conv_opt.VariableType.integer,
                                        upper_bound=2., lower_bound=-3.)
        model.variables.append(integer_var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.QuadraticTerm(
            variable_1=integer_var,
            variable_2=integer_var,
            coefficient=1.)]

        # check problem type
        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.gurobi))
        gurobi_model = solver_model.get_model()
        self.assertTrue(gurobi_model.getAttr(gurobipy.GRB.Attr.IsQP))
        self.assertTrue(gurobi_model.getAttr(gurobipy.GRB.Attr.IsMIP))

    def test_unconstrained_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint())
        with self.assertRaisesRegexp(conv_opt.ConvOptError, 'Constraints must have at least one bound'):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.gurobi))

    def test_unsupported_constraint(self):
        model = conv_opt.Model()
        var = conv_opt.Variable()
        model.variables.append(var)
        model.constraints.append(conv_opt.Constraint([
            None,
        ], upper_bound=0, lower_bound=0))
        with self.assertRaisesRegexp(conv_opt.ConvOptError, '^Unsupported constraint term of type '):
            model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver.gurobi))

    def test_verbose(self):
        model = conv_opt.Model()

        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.gurobi,
                                        tune=True,
                                        verbosity=conv_opt.Verbosity.status)
        with capturer.CaptureOutput(merged=False, relay=False) as captured:
            model.solve(options=options)

            self.assertRegex(captured.stdout.get_text(), '^Parameter LogToConsole unchanged')
            self.assertEqual(captured.stderr.get_text(), '')

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.gurobi,
                                        tune=True,
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
            model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.gurobi, presolve=None))

    def test_tune(self):
        model = conv_opt.Model()
        var = conv_opt.Variable(name='variable', lower_bound=-3, upper_bound=2.)
        model.variables.append(var)
        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=var, coefficient=1.)]
        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.gurobi, tune=True))
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

        result = model.solve(options=conv_opt.SolveOptions(solver=conv_opt.Solver.gurobi))
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

        result = model.solve(options=conv_opt.SolveOptions(
            solver=conv_opt.Solver.gurobi,
            presolve=conv_opt.Presolve.off))
        self.assertEqual(result.status_code, conv_opt.StatusCode.other)
        numpy.testing.assert_equal(result.value, numpy.nan)
        numpy.testing.assert_array_equal(result.primals, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.reduced_costs, numpy.array([numpy.nan] * 2))
        numpy.testing.assert_array_equal(result.duals, numpy.array([numpy.nan]))

    def test_export(self):
        self.assert_export('lp', conv_opt.Solver.gurobi)
        self.assert_export('mps', conv_opt.Solver.gurobi)
        self.assert_export('rew', conv_opt.Solver.gurobi)
        self.assert_export('rlp', conv_opt.Solver.gurobi)
