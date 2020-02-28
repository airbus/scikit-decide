# builders.domain.goals

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## Goals

A domain must inherit this class if it has formalized goals.

### get\_goals <Badge text="Goals" type="tip"/>

<skdecide-signature name= "get_goals" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) domain goals space (finite or infinite set).

By default, `Goals.get_goals()` internally calls `Goals._get_goals_()` the first time and automatically caches its
value to make future calls more efficient (since the goals space is assumed to be constant).

::: warning
Goal states are assumed to be fully observable (i.e. observation = state) so that there is never uncertainty
about whether the goal has been reached or not. This assumption guarantees that any policy that does not
reach the goal with certainty incurs in infinite expected cost. - *Geffner, 2013: A Concise Introduction to
Models and Methods for Automated Planning*
:::

#### Returns
The goals space.

### is\_goal <Badge text="Goals" type="tip"/>

<skdecide-signature name= "is_goal" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an observation belongs to the goals.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
goals space provided by `Goals.get_goals()`, but it can be overridden for faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation is a goal (False otherwise).

### \_get\_goals <Badge text="Goals" type="tip"/>

<skdecide-signature name= "_get_goals" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) domain goals space (finite or infinite set).

By default, `Goals._get_goals()` internally calls `Goals._get_goals_()` the first time and automatically caches
its value to make future calls more efficient (since the goals space is assumed to be constant).

::: warning
Goal states are assumed to be fully observable (i.e. observation = state) so that there is never uncertainty
about whether the goal has been reached or not. This assumption guarantees that any policy that does not
reach the goal with certainty incurs in infinite expected cost. - *Geffner, 2013: A Concise Introduction to
Models and Methods for Automated Planning*
:::

#### Returns
The goals space.

### \_get\_goals\_ <Badge text="Goals" type="tip"/>

<skdecide-signature name= "_get_goals_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the domain goals space (finite or infinite set).

This is a helper function called by default from `Goals._get_goals()`, the difference being that the result is
not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The goals space.

### \_is\_goal <Badge text="Goals" type="tip"/>

<skdecide-signature name= "_is_goal" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an observation belongs to the goals.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
goals space provided by `Goals._get_goals()`, but it can be overridden for faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation is a goal (False otherwise).

