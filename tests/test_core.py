""" Tests for the conv_opt module

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2017-11-14
:Copyright: 2017, Karr Lab
:License: MIT
"""

import conv_opt
import numpy
import os
import tempfile
import unittest


class SolverTestCase(unittest.TestCase):

    def create_lp(self):
        model = conv_opt.Model(name='test-lp')

        # variables
        ex_a = conv_opt.Variable(name='ex_a', lower_bound=None, upper_bound=1.)
        model.variables.append(ex_a)

        r1 = conv_opt.Variable(name='r1', lower_bound=0.)
        model.variables.append(r1)

        r2 = conv_opt.Variable(name='r2', lower_bound=0.)
        model.variables.append(r2)

        r3 = conv_opt.Variable(name='r3', lower_bound=0.)
        model.variables.append(r3)

        r4 = conv_opt.Variable(name='r4', lower_bound=0.)
        model.variables.append(r4)

        biomass_production = conv_opt.Variable(name='biomass_production', lower_bound=0.)
        model.variables.append(biomass_production)

        ex_biomass = conv_opt.Variable(name='ex_biomass', lower_bound=0.)
        model.variables.append(ex_biomass)

        # constraints
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=ex_a, coefficient=1.),
            conv_opt.LinearTerm(variable=r1, coefficient=-1.),
        ], name='a', upper_bound=0, lower_bound=0))
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=r1, coefficient=1.),
            conv_opt.LinearTerm(variable=r2, coefficient=-1.),
            conv_opt.LinearTerm(variable=r3, coefficient=-1.),
        ], name='b', upper_bound=0, lower_bound=0))
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=r2, coefficient=1.),
        ], name='c', upper_bound=0, lower_bound=0))
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=r2, coefficient=1.),
            conv_opt.LinearTerm(variable=r4, coefficient=-1.),
        ], name='d', upper_bound=0, lower_bound=0))
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=r3, coefficient=2.),
            conv_opt.LinearTerm(variable=r4, coefficient=1.),
            conv_opt.LinearTerm(variable=biomass_production, coefficient=-1.),
        ], name='e', upper_bound=0, lower_bound=0))
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=biomass_production, coefficient=1.),
            conv_opt.LinearTerm(variable=ex_biomass, coefficient=-1.),
        ], name='biomass', upper_bound=0, lower_bound=0.))
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=ex_a, coefficient=-1.),
        ], name='upper_bound', upper_bound=1, lower_bound=None))
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=biomass_production, coefficient=1.),
        ], name='lower_bound', upper_bound=None, lower_bound=-1))
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=r1, coefficient=1.),
        ], name='range_bound', upper_bound=10., lower_bound=-10.))

        # objective
        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [conv_opt.LinearTerm(variable=biomass_production, coefficient=1.)]

        # return model
        return model

    def assert_lp(self, model, solver, precision=64, presolve=conv_opt.Presolve.off,
                  status_message='optimal', check_reduced_costs=True, check_duals=True,
                  places=10):
        options = conv_opt.SolveOptions(
            solver=conv_opt.Solver[solver],
            precision=precision,
            presolve=presolve,
        )
        solver_model = model.convert(options=options)
        result = solver_model.solve()
        self.assertEqual(result.status_code, conv_opt.StatusCode.optimal)
        self.assertEqual(result.status_message, status_message)
        self.assertAlmostEqual(result.value, 2., places=7)

        numpy.testing.assert_array_almost_equal(result.primals, numpy.array([1., 1., 0., 1., 0., 2., 2.]))

        if check_reduced_costs:
            self.assertAlmostEqual(result.reduced_costs[0], 2, places=places)
            self.assertAlmostEqual(result.reduced_costs[1], 0, places=places)
            #self.assertAlmostEqual(result.reduced_costs[2], -1, places=places)
            self.assertAlmostEqual(result.reduced_costs[3], 0, places=places)
            self.assertAlmostEqual(result.reduced_costs[4], 0, places=places)
            self.assertAlmostEqual(result.reduced_costs[5], 0, places=places)
            self.assertAlmostEqual(result.reduced_costs[6], 0, places=places)

        if check_duals:
            self.assertAlmostEqual(result.duals[0], -2, places=places)
            self.assertAlmostEqual(result.duals[1], -2, places=places)
            #self.assertAlmostEqual(result.duals[2], 0, places=places)
            self.assertAlmostEqual(result.duals[3], -1, places=places)
            self.assertAlmostEqual(result.duals[4], -1, places=places)
            self.assertAlmostEqual(result.duals[5], 0, places=places)

        solver_model.get_stats()

    def create_milp(self):
        model = self.create_lp()
        model.variables[0].type = conv_opt.VariableType.binary
        model.variables[1].type = conv_opt.VariableType.integer
        return model

    def assert_milp(self, model, solver, presolve=conv_opt.Presolve.off,
                    status_message='optimal', check_reduced_costs=False, check_duals=False):
        self.assert_lp(model, solver, presolve=presolve,
                       status_message=status_message, check_reduced_costs=check_reduced_costs, check_duals=check_duals)

    def create_qp(self):
        model = conv_opt.Model(name='test-lp')

        # variables
        ex_a = conv_opt.Variable(name='ex_a', lower_bound=0, upper_bound=1)
        model.variables.append(ex_a)

        r1 = conv_opt.Variable(name='r1', lower_bound=0)
        model.variables.append(r1)

        r2 = conv_opt.Variable(name='r2', lower_bound=0)
        model.variables.append(r2)

        r3 = conv_opt.Variable(name='r3', lower_bound=0)
        model.variables.append(r3)

        r4 = conv_opt.Variable(name='r4', lower_bound=0)
        model.variables.append(r4)

        biomass_production = conv_opt.Variable(name='biomass_production', lower_bound=0)
        model.variables.append(biomass_production)

        ex_biomass = conv_opt.Variable(name='ex_biomass', lower_bound=0)
        model.variables.append(ex_biomass)

        # constraints
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=ex_a, coefficient=1),
            conv_opt.LinearTerm(variable=r1, coefficient=-1),
        ], name='a', upper_bound=0, lower_bound=0))
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=r1, coefficient=1),
            conv_opt.LinearTerm(variable=r2, coefficient=-1),
            conv_opt.LinearTerm(variable=r3, coefficient=-1),
        ], name='b', upper_bound=0, lower_bound=0))
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=r2, coefficient=1),
        ], name='c', upper_bound=0, lower_bound=0))
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=r2, coefficient=1),
            conv_opt.LinearTerm(variable=r4, coefficient=-1),
        ], name='d', upper_bound=0, lower_bound=0))
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=r3, coefficient=2),
            conv_opt.LinearTerm(variable=r4, coefficient=1),
            conv_opt.LinearTerm(variable=biomass_production, coefficient=-1),
        ], name='e', upper_bound=0, lower_bound=0))
        model.constraints.append(conv_opt.Constraint(terms=[
            conv_opt.LinearTerm(variable=biomass_production, coefficient=1),
            conv_opt.LinearTerm(variable=ex_biomass, coefficient=-1),
        ], name='biomass', upper_bound=0, lower_bound=0))

        # objective
        model.objective_direction = conv_opt.ObjectiveDirection.minimize
        model.objective_terms = [
            conv_opt.QuadraticTerm(coefficient=1., variable_1=ex_a, variable_2=ex_a),
            conv_opt.QuadraticTerm(coefficient=1., variable_1=r1, variable_2=r1),
            conv_opt.QuadraticTerm(coefficient=-2., variable_1=r1, variable_2=ex_a),
            conv_opt.LinearTerm(coefficient=-1., variable=ex_biomass),
        ]

        # return model
        return model

    def assert_qp(self, model, solver, presolve=conv_opt.Presolve.on, solver_options=None,
                  status_message='optimal', check_reduced_costs=False, check_duals=False):
        options = conv_opt.SolveOptions(
            solver=conv_opt.Solver[solver],
            presolve=presolve,
            solver_options=solver_options,
        )
        solver_model = model.convert(options=options)
        result = solver_model.solve()
        self.assertEqual(result.status_code, conv_opt.StatusCode.optimal)
        self.assertEqual(result.status_message, status_message)
        self.assertAlmostEqual(result.value, -2.)

        numpy.testing.assert_array_almost_equal(result.primals, numpy.array([1., 1., 0., 1., 0., 2., 2.]))

        if check_reduced_costs:
            self.assertEqual(result.reduced_costs[0], 6)
            self.assertEqual(result.reduced_costs[1], 0)
            self.assertEqual(result.reduced_costs[2], 0)
            self.assertEqual(result.reduced_costs[3], 0)
            self.assertEqual(result.reduced_costs[4], 0)
            self.assertEqual(result.reduced_costs[5], 0)
            self.assertEqual(result.reduced_costs[6], 0)

        if check_duals:
            self.assertEqual(result.duals[0], 0)
            self.assertEqual(result.duals[1], 0)
            self.assertEqual(result.duals[2], 0)
            self.assertEqual(result.duals[3], 0)
            self.assertEqual(result.duals[4], 0)
            self.assertEqual(result.duals[5], 0)

        solver_model.get_stats()

    def create_qp_2(self):
        model = conv_opt.Model()

        var_1 = conv_opt.Variable(name='var_1', lower_bound=3.)
        model.variables.append(var_1)

        var_2 = conv_opt.Variable(name='var_2', upper_bound=1.)
        model.variables.append(var_2)

        model.objective_direction = conv_opt.ObjectiveDirection.maximize
        model.objective_terms = [
            conv_opt.QuadraticTerm(variable_1=var_1, variable_2=var_1, coefficient=-1.),
            conv_opt.QuadraticTerm(variable_1=var_2, variable_2=var_2, coefficient=-1.),
            conv_opt.QuadraticTerm(variable_1=var_1, variable_2=var_2, coefficient=2.),
        ]
        return model

    def assert_qp_2(self, model, solver, solver_options=None):
        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver[solver], solver_options=solver_options))
        result = solver_model.solve()
        self.assertAlmostEqual(result.value, -4., places=4)

        solver_model.get_stats()

    def create_qp_3(self):
        model = conv_opt.Model()

        var_1 = conv_opt.Variable(name='var_1')
        model.variables.append(var_1)

        var_2 = conv_opt.Variable(name='var_2')
        model.variables.append(var_2)

        var_3 = conv_opt.Variable(name='var_3')
        model.variables.append(var_3)

        model.constraints.append(conv_opt.Constraint([
            conv_opt.LinearTerm(variable=var_1, coefficient=1.),
            conv_opt.LinearTerm(variable=var_2, coefficient=1.),
            conv_opt.LinearTerm(variable=var_3, coefficient=1.),
        ], lower_bound=9.))

        model.objective_direction = conv_opt.ObjectiveDirection.minimize
        model.objective_terms = [
            conv_opt.QuadraticTerm(variable_1=var_1, variable_2=var_1, coefficient=1.),
            conv_opt.QuadraticTerm(variable_1=var_2, variable_2=var_2, coefficient=1.),
            conv_opt.QuadraticTerm(variable_1=var_3, variable_2=var_3, coefficient=1.),
        ]
        return model

    def assert_qp_3(self, model, solver, solver_options=None):
        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver[solver], solver_options=solver_options))
        result = solver_model.solve()
        self.assertAlmostEqual(result.value, 3 * 3.**2., places=4)

        solver_model.get_stats()

    def create_qp_4(self):
        model = conv_opt.Model()

        var_1 = conv_opt.Variable(name='var_1')
        model.variables.append(var_1)

        var_2 = conv_opt.Variable(name='var_2')
        model.variables.append(var_2)

        var_3 = conv_opt.Variable(name='var_3')
        model.variables.append(var_3)

        model.constraints.append(conv_opt.Constraint([
            conv_opt.LinearTerm(variable=var_1, coefficient=-4.),
            conv_opt.LinearTerm(variable=var_2, coefficient=-3.),
        ], lower_bound=-8., name='contraint_1'))

        model.constraints.append(conv_opt.Constraint([
            conv_opt.LinearTerm(variable=var_1, coefficient=2.),
            conv_opt.LinearTerm(variable=var_2, coefficient=1.),
        ], lower_bound=2., name='contraint_2'))

        model.constraints.append(conv_opt.Constraint([
            conv_opt.LinearTerm(variable=var_2, coefficient=-2.),
            conv_opt.LinearTerm(variable=var_3, coefficient=1.),
        ], lower_bound=0., name='contraint_3'))

        model.objective_direction = conv_opt.ObjectiveDirection.minimize
        model.objective_terms = [
            conv_opt.QuadraticTerm(variable_1=var_1, variable_2=var_1, coefficient=0.5),
            conv_opt.QuadraticTerm(variable_1=var_2, variable_2=var_2, coefficient=0.5),
            conv_opt.QuadraticTerm(variable_1=var_3, variable_2=var_3, coefficient=0.5),
            conv_opt.LinearTerm(variable=var_2, coefficient=-5.),
        ]
        return model

    def assert_qp_4(self, model, solver, solver_options=None):
        solver_model = model.convert(options=conv_opt.SolveOptions(solver=conv_opt.Solver[solver], solver_options=solver_options))
        result = solver_model.solve()
        self.assertAlmostEqual(result.value, -2.380952380952381, places=4)

        solver_model.get_stats()

    def create_miqp(self):
        model = self.create_qp()
        model.variables[0].type = conv_opt.VariableType.binary
        model.variables[1].type = conv_opt.VariableType.integer
        return model

    def assert_miqp(self, model, solver, presolve=conv_opt.Presolve.off,
                    status_message='optimal', check_reduced_costs=False, check_duals=False):
        self.assert_qp(model, solver, presolve=presolve,
                       status_message=status_message, check_reduced_costs=check_reduced_costs, check_duals=check_duals)

    def assert_export(self, format, solver=None):
        file, filename = tempfile.mkstemp(suffix='.' + format)
        os.close(file)
        os.remove(filename)

        self.assertFalse(os.path.isfile(filename))
        model = self.create_lp()
        model.export(filename, solver=solver)
        self.assertTrue(os.path.isfile(filename))

        os.remove(filename)


class CoreTestCase(SolverTestCase):

    def test_get_type_lp(self):
        model = self.create_lp()
        self.assertEqual(model.get_type(), conv_opt.ModelType.lp)

    def test_get_type_milp(self):
        model = self.create_lp()

        model.variables[0].type = conv_opt.VariableType.binary
        self.assertEqual(model.get_type(), conv_opt.ModelType.milp)

        model.variables[0].type = conv_opt.VariableType.integer
        self.assertEqual(model.get_type(), conv_opt.ModelType.milp)

    def test_get_type_qp(self):
        model = self.create_lp()
        model.objective_terms.append(conv_opt.QuadraticTerm(
            coefficient=1.,
            variable_1=model.variables[0],
            variable_2=model.variables[0],
        ))
        self.assertEqual(model.get_type(), conv_opt.ModelType.qp)

        model = self.create_qp()
        self.assertEqual(model.get_type(), conv_opt.ModelType.qp)

    def test_get_type_miqp(self):
        model = self.create_lp()

        model.objective_terms.append(conv_opt.QuadraticTerm(
            coefficient=1.,
            variable_1=model.variables[0],
            variable_2=model.variables[0],
        ))

        model.variables[0].type = conv_opt.VariableType.binary
        self.assertEqual(model.get_type(), conv_opt.ModelType.miqp)

        model.variables[0].type = conv_opt.VariableType.integer
        self.assertEqual(model.get_type(), conv_opt.ModelType.miqp)

    def test_get_type_unsupported_variable(self):
        model = self.create_lp()
        model.variables[0].type = None
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported variable of type '):
            model.get_type()

    def test_get_type_unsupported_objective_term(self):
        model = self.create_lp()
        model.objective_terms[0] = None
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported objective term of type '):
            model.get_type()

    def test_unsupported_solver(self):
        model = self.create_lp()
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported solver '):
            model.convert(conv_opt.SolveOptions(solver=None))

    def test__unpack_result(self):
        model = self.create_lp()

        primals = 1. * numpy.array(range(len(model.variables)))
        reduced_costs = 2. * numpy.array(range(len(model.variables)))
        duals = 3. * numpy.array(range(len(model.constraints)))
        result = conv_opt.Result(None, None, None, primals, reduced_costs, duals)

        model._unpack_result(result)

        self.assertEqual(model.variables[0].primal, 0.)
        self.assertEqual(model.variables[1].primal, 1.)
        self.assertEqual(model.variables[2].primal, 2.)
        self.assertEqual(model.variables[3].primal, 3.)
        self.assertEqual(model.variables[4].primal, 4.)
        self.assertEqual(model.variables[5].primal, 5.)
        self.assertEqual(model.variables[6].primal, 6.)

        self.assertEqual(model.variables[0].reduced_cost, 0.)
        self.assertEqual(model.variables[1].reduced_cost, 2.)
        self.assertEqual(model.variables[2].reduced_cost, 4.)
        self.assertEqual(model.variables[3].reduced_cost, 6.)
        self.assertEqual(model.variables[4].reduced_cost, 8.)
        self.assertEqual(model.variables[5].reduced_cost, 10.)
        self.assertEqual(model.variables[6].reduced_cost, 12.)

        self.assertEqual(model.constraints[0].dual, 0.)
        self.assertEqual(model.constraints[1].dual, 3.)
        self.assertEqual(model.constraints[2].dual, 6.)
        self.assertEqual(model.constraints[3].dual, 9.)
        self.assertEqual(model.constraints[4].dual, 12.)
        self.assertEqual(model.constraints[5].dual, 15.)

    def test_export(self):
        # no suggested solver
        self.assert_export('lp')

        # unsupported format
        with self.assertRaisesRegex(conv_opt.ConvOptError, '^Unsupported format '):
            self.assert_export('__xxx__')

    def test_conv_opt_error(self):
        conv_opt.ConvOptError()
