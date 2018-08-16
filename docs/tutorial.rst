Tutorial
========

#. Import the :obj:`conv_opt` package:

    .. code-block:: python

        import conv_opt

#. Create a model and, optionally, name the model:

    .. code-block:: python

        model = conv_opt.Model(name='model')

    The default name is :obj:`None`.

#. Create variables; optionally set their names, types, and upper and lower bounds; and add them to models:

    .. code-block:: python

        var_1 = conv_opt.Variable(name='var_1', type=conv_opt.VariableType.continuous, lower_bound=0, upper_bound=1)
        model.variables.append(var_1)

        var_2 = conv_opt.Variable(name='var_2', type=conv_opt.VariableType.continuous, lower_bound=0, upper_bound=1)
        model.variables.append(var_2)

    The default name is :obj:`None`.

    The :obj:`type` argument must be one of the following values. The default type is continuous.

    * :obj:`conv_opt.VariableType.binary`
    * :obj:`conv_opt.VariableType.integer`
    * :obj:`conv_opt.VariableType.continuous`
    * :obj:`conv_opt.VariableType.semi_integer`
    * :obj:`conv_opt.VariableType.semi_continuous`
    * :obj:`conv_opt.VariableType.partially_integer`

    The default upper and lower bounds are :obj:`None`.

#. Set the objective expression and direction:

    .. code-block:: python

        model.objective_terms = [
            conv_opt.LinearTerm(var_1, 1.),
            conv_opt.QuadraticTerm(var_2, var_2, 1.),
        ]
        model.objective_direction = conv_opt.ObjectiveDirection.maximize

    :obj:`objective_terms` should be a list of linear (:py:class:`conv_opt.LinearTerm <conv_opt.core.LinearTerm>`) and quadratic terms (:py:class:`conv_opt.QuadraticTerm <conv_opt.core.QuadraticTerm>`). The
    arguments to the constructors of these class are the variables involved in the term and coefficient for the term.

    :obj:`objective_direction` can be either of following values. The default direction is minimize.

        * :obj:`conv_opt.ObjectiveDirection.maximize`
        * :obj:`conv_opt.ObjectiveDirection.minimize`

#. Create constraints; optionally set their names and upper and lower bounds; and add them to models:

    .. code-block:: python

        contraint_1 = conv_opt.Constraint([
            conv_opt.LinearTerm(var_1, 1),
            conv_opt.LinearTerm(var_2, -1),
        ], name='contraint_1', upper_bound=0, lower_bound=0)
        model.constraints.append(contraint_1)

    The first argument should be a list of linear terms.

    The default name and upper and lower bounds are :obj:`None`.


#. Configure the options for solving the model:

    .. code-block:: python

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.cplex, presolve=Presolve.off, verbosity=False)

    The :obj:`solver` argument allows you to select a specific solver. The argument must be one of the following values. The default solver is GLPK.

    * :obj:`conv_opt.Solver.cbc`
    * :obj:`conv_opt.Solver.cplex`
    * :obj:`conv_opt.Solver.cvxopt`
    * :obj:`conv_opt.Solver.glpk`
    * :obj:`conv_opt.Solver.gurobi`
    * :obj:`conv_opt.Solver.minos`
    * :obj:`conv_opt.Solver.mosek`
    * :obj:`conv_opt.Solver.quadprog`
    * :obj:`conv_opt.Solver.scipy`
    * :obj:`conv_opt.Solver.soplex`
    * :obj:`conv_opt.Solver.xpress`

    The :obj:`presolve` aregument must be one of the following values. The default value is off.

    * :obj:`conv_opt.Presolve.auto`
    * :obj:`conv_opt.Presolve.on`
    * :obj:`conv_opt.Presolve.off`

    Please see the :py:class:`API documentation <conv_opt.core.SolveOptions>` for information about additional options.

#. Solve the model:

    .. code-block:: python

        result = model.solve(options)
        if result.status_code != conv_opt.StatusCode.optimal:
            raise Exception(result.status_message)
        value = result.value
        primals = result.primals

    The result will be an instance of :py:class:`conv_opt.Result <conv_opt.core.Result>`. The attributes of this class include:

    * :obj:`status_code`
    * :obj:`status_message`
    * :obj:`value`
    * :obj:`primals`
    * :obj:`reduced_costs`
    * :obj:`duals`

    :obj:`status_code` will be an instance of the :py:class:`conv_opt.StatusCode <conv_opt.core.StatusCode>` enumerated type.

#. Get diagnostic information about the model:

    .. code-block:: python

        stats = model.get_stats()

#. Convert the model to the lower level API of one of the solvers:

    .. code-block:: python

        options = conv_opt.SolveOptions(solver=conv_opt.Solver.cplex)
        cplex_model = model.convert(options)

#. Export the model to a file:

    .. code-block:: python

        filename='/path/to/file.ext'
        model.export(filename)

    :obj:`conv_opt` supports the following extensions:

    * alp
    * cbf
    * dpe: dual perturbed model
    * dua: dual
    * jtask: Jtask format
    * lp
    * mps
    * opf
    * ppe: perturbed model
    * rew: model with generic names in mps format
    * rlp: model with generic names in lp format
    * sav
    * task: Task format
    * xml: OSiL
