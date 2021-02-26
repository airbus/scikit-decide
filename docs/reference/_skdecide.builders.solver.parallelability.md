# builders.solver.parallelability

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## ParallelSolver

A solver must inherit this class if it wants to call several cloned parallel domains in separate concurrent processes.
The solver is meant to be called either within a 'with' context statement, or to be cleaned up using the close() method.

### Constructor <Badge text="ParallelSolver" type="tip"/>

<skdecide-signature name= "ParallelSolver" :sig="{'params': [{'name': 'domain_factory', 'annotation': 'Callable[[], Domain]'}, {'name': 'parallel', 'default': 'False', 'annotation': 'bool'}, {'name': 'shared_memory_proxy', 'default': 'None'}]}"></skdecide-signature>

Creates a parallelizable solver
#### Parameters
- **domain_factory**: A callable with no argument returning the domain to solve (factory is the domain class if None).
- **parallel**: True if the solver is run in parallel mode.
- **shared_memory_proxy**: Shared memory proxy to use if not None, otherwise run piped parallel domains.

### call\_domain\_method <Badge text="ParallelSolver" type="tip"/>

<skdecide-signature name= "call_domain_method" :sig="{'params': [{'name': 'self'}, {'name': 'name'}, {'name': 'args'}]}"></skdecide-signature>

Calls a parallel domain's method.
This is the only way to get a domain method for a parallel domain.

### close <Badge text="ParallelSolver" type="tip"/>

<skdecide-signature name= "close" :sig="{'params': [{'name': 'self'}]}"></skdecide-signature>

Joins the parallel domains' processes.
Not calling this method (or not using the 'with' context statement)
results in the solver forever waiting for the domain processes to exit.

### get\_domain <Badge text="ParallelSolver" type="tip"/>

<skdecide-signature name= "get_domain" :sig="{'params': [{'name': 'self'}]}"></skdecide-signature>

Returns the domain, optionnally creating a parallel domain if not already created.

### \_initialize <Badge text="ParallelSolver" type="tip"/>

<skdecide-signature name= "_initialize" :sig="{'params': [{'name': 'self'}]}"></skdecide-signature>

Launches the parallel domains.
This method requires to have previously recorded the self._domain_factory (e.g. after calling _init_solve),
the set of lambda functions passed to the solver's constructor (e.g. heuristic lambda for heuristic-based solvers),
and whether the parallel domain jobs should notify their status via the IPC protocol (required when interacting with
other programming languages like C++)

