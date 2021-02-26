# builders.discrete_optimization.generic_tools.cp_tools

Constraint programming common utilities and class that should be used by any solver using CP

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## CPSolverName

Enum choice of underlying CP solver used by Minizinc typically

## ParametersCP

Parameters that can be used by any cp - solver

### Constructor <Badge text="ParametersCP" type="tip"/>

<skdecide-signature name= "ParametersCP" :sig="{'params': [{'name': 'time_limit'}, {'name': 'pool_solutions'}, {'name': 'intermediate_solution', 'annotation': 'bool'}, {'name': 'all_solutions', 'annotation': 'bool'}, {'name': 'nr_solutions', 'annotation': 'int'}]}"></skdecide-signature>

:param time_limit: in seconds, the time limit of solving the cp model
:param pool_solutions: TODO remove it it's not used
:param intermediate_solution: retrieve intermediate solutions
:param all_solutions: returns all solutions found by the cp solver
:param nr_solutions: max number of solution returned

## CPSolver

Additional function to be implemented by a CP Solver.

### init\_model <Badge text="CPSolver" type="tip"/>

<skdecide-signature name= "init_model" :sig="{'params': [{'name': 'self'}, {'name': 'args'}]}"></skdecide-signature>

Instantiate a CP model instance

### retrieve\_solutions <Badge text="CPSolver" type="tip"/>

<skdecide-signature name= "retrieve_solutions" :sig="{'params': [{'name': 'self'}, {'name': 'result'}, {'name': 'parameters_cp', 'annotation': 'ParametersCP'}], 'return': 'ResultStorage'}"></skdecide-signature>

Returns a storage solution coherent with the given parameters.
:param result: Result storage returned by the cp solver
:param parameters_cp: parameters of the CP solver.
:return:

