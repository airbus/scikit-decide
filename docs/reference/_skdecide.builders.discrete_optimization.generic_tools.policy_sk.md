# builders.discrete_optimization.generic_tools.policy_sk

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## LookUpPolicy

A solver which is initialized with a stochastic policy but provides
interfaces for deterministic policies.

### Constructor <Badge text="LookUpPolicy" type="tip"/>

<skdecide-signature name= "LookUpPolicy" :sig="{'params': [{'name': 'policy_dict'}]}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

### get\_next\_action <Badge text="DeterministicPolicies" type="warn"/>

<skdecide-signature name= "get_next_action" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'D.T_agent[D.T_concurrency[D.T_event]]'}"></skdecide-signature>

Get the next deterministic action (from the solver's current policy).

#### Parameters
- **observation**: The observation for which next action is requested.

#### Returns
The next deterministic action.

### get\_next\_action\_distribution <Badge text="UncertainPolicies" type="warn"/>

<skdecide-signature name= "get_next_action_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'Distribution[D.T_agent[D.T_concurrency[D.T_event]]]'}"></skdecide-signature>

Get the probabilistic distribution of next action for the given observation (from the solver's current
policy).

#### Parameters
- **observation**: The observation to consider.

#### Returns
The probabilistic distribution of next action.

### is\_policy\_defined\_for <Badge text="Policies" type="warn"/>

<skdecide-signature name= "is_policy_defined_for" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check whether the solver's current policy is defined for the given observation.

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the policy is defined for the given observation memory (False otherwise).

### sample\_action <Badge text="Policies" type="warn"/>

<skdecide-signature name= "sample_action" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'D.T_agent[D.T_concurrency[D.T_event]]'}"></skdecide-signature>

Sample an action for the given observation (from the solver's current policy).

#### Parameters
- **observation**: The observation for which an action must be sampled.

#### Returns
The sampled action.

### \_get\_next\_action <Badge text="DeterministicPolicies" type="warn"/>

<skdecide-signature name= "_get_next_action" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'D.T_agent[D.T_concurrency[D.T_event]]'}"></skdecide-signature>

Get the next deterministic action (from the solver's current policy).

#### Parameters
- **observation**: The observation for which next action is requested.

#### Returns
The next deterministic action.

### \_get\_next\_action\_distribution <Badge text="UncertainPolicies" type="warn"/>

<skdecide-signature name= "_get_next_action_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'Distribution[D.T_agent[D.T_concurrency[D.T_event]]]'}"></skdecide-signature>

Get the probabilistic distribution of next action for the given observation (from the solver's current
policy).

#### Parameters
- **observation**: The observation to consider.

#### Returns
The probabilistic distribution of next action.

### \_is\_policy\_defined\_for <Badge text="Policies" type="warn"/>

<skdecide-signature name= "_is_policy_defined_for" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check whether the solver's current policy is defined for the given observation.

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the policy is defined for the given observation memory (False otherwise).

### \_sample\_action <Badge text="Policies" type="warn"/>

<skdecide-signature name= "_sample_action" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'D.T_agent[D.T_concurrency[D.T_event]]'}"></skdecide-signature>

Sample an action for the given observation (from the solver's current policy).

#### Parameters
- **observation**: The observation for which an action must be sampled.

#### Returns
The sampled action.

