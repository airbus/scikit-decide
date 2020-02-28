# hub.solver.ray_rllib.ray_rllib

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## RayRLlib

This class wraps a Ray RLlib solver (ray[rllib]) as a scikit-decide solver.

::: warning
Using this class requires Ray RLlib to be installed.
:::

### Constructor <Badge text="RayRLlib" type="tip"/>

<skdecide-signature name= "RayRLlib" :sig="{'params': [{'name': 'algo_class', 'annotation': 'Type[Trainer]'}, {'name': 'train_iterations', 'annotation': 'int'}, {'name': 'config', 'default': 'None', 'annotation': 'Optional[Dict]'}, {'name': 'policy_configs', 'default': '{\'policy\': {}}', 'annotation': 'Dict[str, Dict]'}, {'name': 'policy_mapping_fn', 'default': '<lambda function>', 'annotation': 'Callable[[str], str]'}], 'return': 'None'}"></skdecide-signature>

Initialize Ray RLlib.

#### Parameters
- **algo_class**: The class of Ray RLlib trainer/agent to wrap.
- **train_iterations**: The number of iterations to call the trainer's train() method.
- **config**: The configuration dictionary for the trainer.
- **policy_configs**: The mapping from policy id (str) to additional config (dict) (leave default for single policy).
- **policy_mapping_fn**: The function mapping agent ids to policy ids (leave default for single policy).

### check\_domain <Badge text="Solver" type="warn"/>

<skdecide-signature name= "check_domain" :sig="{'params': [{'name': 'domain', 'annotation': 'Domain'}], 'return': 'bool'}"></skdecide-signature>

Check whether a domain is compliant with this solver type.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **domain**: The domain to check.

#### Returns
True if the domain is compliant with the solver type (False otherwise).

### get\_domain\_requirements <Badge text="Solver" type="warn"/>

<skdecide-signature name= "get_domain_requirements" :sig="{'params': [], 'return': 'List[type]'}"></skdecide-signature>

Get domain requirements for this solver class to be applicable.

Domain requirements are classes from the `skdecide.builders.domain` package that the domain needs to inherit from.

#### Returns
A list of classes to inherit from.

### is\_policy\_defined\_for <Badge text="Policies" type="warn"/>

<skdecide-signature name= "is_policy_defined_for" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check whether the solver's current policy is defined for the given observation.

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the policy is defined for the given observation memory (False otherwise).

### load <Badge text="Restorable" type="warn"/>

<skdecide-signature name= "load" :sig="{'params': [{'name': 'self'}, {'name': 'path', 'annotation': 'str'}, {'name': 'domain_factory', 'annotation': 'Callable[[], D]'}], 'return': 'None'}"></skdecide-signature>

Restore the solver state from given path.

#### Parameters
- **path**: The path where the solver state was saved.
- **domain_factory**: A callable with no argument returning the domain to solve (useful in some implementations).

### reset <Badge text="Solver" type="warn"/>

<skdecide-signature name= "reset" :sig="{'params': [{'name': 'self'}], 'return': 'None'}"></skdecide-signature>

Reset whatever is needed on this solver before running a new episode.

This function does nothing by default but can be overridden if needed (e.g. to reset the hidden state of a LSTM
policy network, which carries information about past observations seen in the previous episode).

### sample\_action <Badge text="Policies" type="warn"/>

<skdecide-signature name= "sample_action" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'D.T_agent[D.T_concurrency[D.T_event]]'}"></skdecide-signature>

Sample an action for the given observation (from the solver's current policy).

#### Parameters
- **observation**: The observation for which an action must be sampled.

#### Returns
The sampled action.

### save <Badge text="Restorable" type="warn"/>

<skdecide-signature name= "save" :sig="{'params': [{'name': 'self'}, {'name': 'path', 'annotation': 'str'}], 'return': 'None'}"></skdecide-signature>

Save the solver state to given path.

#### Parameters
- **path**: The path to store the saved state.

### solve <Badge text="Solver" type="warn"/>

<skdecide-signature name= "solve" :sig="{'params': [{'name': 'self'}, {'name': 'domain_factory', 'annotation': 'Callable[[], Domain]'}], 'return': 'None'}"></skdecide-signature>

Run the solving process.

By default, `Solver.solve()` provides some boilerplate code and internally calls `Solver._solve()`. The
boilerplate code transforms the domain factory to auto-cast the new domains to the level expected by the solver.

#### Parameters
- **domain_factory**: A callable with no argument returning the domain to solve (can be just a domain class).

::: tip
The nature of the solutions produced here depends on other solver's characteristics like
`policy` and `assessibility`.
:::

### solve\_from <Badge text="Solver" type="warn"/>

<skdecide-signature name= "solve_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Run the solving process from a given state.

::: tip
Create the domain first by calling the @Solver.reset() method
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.

::: tip
The nature of the solutions produced here depends on other solver's characteristics like
`policy` and `assessibility`.
:::

### \_check\_domain <Badge text="Solver" type="warn"/>

<skdecide-signature name= "_check_domain" :sig="{'params': [{'name': 'domain', 'annotation': 'Domain'}], 'return': 'bool'}"></skdecide-signature>

Check whether a domain is compliant with this solver type.

By default, `Solver._check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden to define specific
checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all domain
requirements are met.

#### Parameters
- **domain**: The domain to check.

#### Returns
True if the domain is compliant with the solver type (False otherwise).

### \_check\_domain\_additional <Badge text="Solver" type="warn"/>

<skdecide-signature name= "_check_domain_additional" :sig="{'params': [{'name': 'domain', 'annotation': 'Domain'}], 'return': 'bool'}"></skdecide-signature>

Check whether the given domain is compliant with the specific requirements of this solver type (i.e. the
ones in addition to "domain requirements").

This is a helper function called by default from `Solver._check_domain()`. It focuses on specific checks, as
opposed to taking also into account the domain requirements for the latter.

#### Parameters
- **domain**: The domain to check.

#### Returns
True if the domain is compliant with the specific requirements of this solver type (False otherwise).

### \_get\_domain\_requirements <Badge text="Solver" type="warn"/>

<skdecide-signature name= "_get_domain_requirements" :sig="{'params': [], 'return': 'List[type]'}"></skdecide-signature>

Get domain requirements for this solver class to be applicable.

Domain requirements are classes from the `skdecide.builders.domain` package that the domain needs to inherit from.

#### Returns
A list of classes to inherit from.

### \_is\_policy\_defined\_for <Badge text="Policies" type="warn"/>

<skdecide-signature name= "_is_policy_defined_for" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check whether the solver's current policy is defined for the given observation.

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the policy is defined for the given observation memory (False otherwise).

### \_load <Badge text="Restorable" type="warn"/>

<skdecide-signature name= "_load" :sig="{'params': [{'name': 'self'}, {'name': 'path', 'annotation': 'str'}, {'name': 'domain_factory', 'annotation': 'Callable[[], D]'}]}"></skdecide-signature>

Restore the solver state from given path.

#### Parameters
- **path**: The path where the solver state was saved.
- **domain_factory**: A callable with no argument returning the domain to solve (useful in some implementations).

### \_reset <Badge text="Solver" type="warn"/>

<skdecide-signature name= "_reset" :sig="{'params': [{'name': 'self'}], 'return': 'None'}"></skdecide-signature>

Reset whatever is needed on this solver before running a new episode.

This function does nothing by default but can be overridden if needed (e.g. to reset the hidden state of a LSTM
policy network, which carries information about past observations seen in the previous episode).

### \_sample\_action <Badge text="Policies" type="warn"/>

<skdecide-signature name= "_sample_action" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'D.T_agent[D.T_concurrency[D.T_event]]'}"></skdecide-signature>

Sample an action for the given observation (from the solver's current policy).

#### Parameters
- **observation**: The observation for which an action must be sampled.

#### Returns
The sampled action.

### \_save <Badge text="Restorable" type="warn"/>

<skdecide-signature name= "_save" :sig="{'params': [{'name': 'self'}, {'name': 'path', 'annotation': 'str'}], 'return': 'None'}"></skdecide-signature>

Save the solver state to given path.

#### Parameters
- **path**: The path to store the saved state.

### \_solve <Badge text="Solver" type="warn"/>

<skdecide-signature name= "_solve" :sig="{'params': [{'name': 'self'}, {'name': 'domain_factory', 'annotation': 'Callable[[], Domain]'}], 'return': 'None'}"></skdecide-signature>

Run the solving process.

By default, `Solver._solve()` provides some boilerplate code and internally calls `Solver._solve_domain()`. The
boilerplate code transforms the domain factory to auto-cast the new domains to the level expected by the solver.

#### Parameters
- **domain_factory**: A callable with no argument returning the domain to solve (can be just a domain class).

::: tip
The nature of the solutions produced here depends on other solver's characteristics like
`policy` and `assessibility`.
:::

### \_solve\_domain <Badge text="Solver" type="warn"/>

<skdecide-signature name= "_solve_domain" :sig="{'params': [{'name': 'self'}, {'name': 'domain_factory', 'annotation': 'Callable[[], D]'}], 'return': 'None'}"></skdecide-signature>

Run the solving process.

This is a helper function called by default from `Solver._solve()`, the difference being that the domain factory
here returns domains auto-cast to the level expected by the solver.

#### Parameters
- **domain_factory**: A callable with no argument returning the domain to solve (auto-cast to expected level).

::: tip
The nature of the solutions produced here depends on other solver's characteristics like
`policy` and `assessibility`.
:::

### \_solve\_from <Badge text="Solver" type="warn"/>

<skdecide-signature name= "_solve_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Run the solving process from a given state.

::: tip
Create the domain first by calling the @Solver.reset() method
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.

::: tip
The nature of the solutions produced here depends on other solver's characteristics like
`policy` and `assessibility`.
:::

