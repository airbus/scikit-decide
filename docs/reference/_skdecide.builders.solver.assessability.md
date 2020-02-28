# builders.solver.assessability

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## Utilities

A solver must inherit this class if it can provide the utility function (i.e. value function).

### get\_utility <Badge text="Utilities" type="tip"/>

<skdecide-signature name= "get_utility" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'D.T_value'}"></skdecide-signature>

Get the estimated on-policy utility of the given observation.

In mathematical terms, for a fully observable domain, this function estimates:
$$V^\pi(s)=\underset{\tau\sim\pi}{\mathbb{E}}[R(\tau)|s_0=s]$$
where $\pi$ is the current policy, any $\tau=(s_0,a_0, s_1, a_1, ...)$ represents a trajectory sampled from
the policy, $R(\tau)$ is the return (cumulative reward) and $s_0$ the initial state for the trajectories.

#### Parameters
- **observation**: The observation to consider.

#### Returns
The estimated on-policy utility of the given observation.

### \_get\_utility <Badge text="Utilities" type="tip"/>

<skdecide-signature name= "_get_utility" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'D.T_value'}"></skdecide-signature>

Get the estimated on-policy utility of the given observation.

In mathematical terms, for a fully observable domain, this function estimates:
$$V^\pi(s)=\underset{\tau\sim\pi}{\mathbb{E}}[R(\tau)|s_0=s]$$
where $\pi$ is the current policy, any $\tau=(s_0,a_0, s_1, a_1, ...)$ represents a trajectory sampled from
the policy, $R(\tau)$ is the return (cumulative reward) and $s_0$ the initial state for the trajectories.

#### Parameters
- **observation**: The observation to consider.

#### Returns
The estimated on-policy utility of the given observation.

## QValues

A solver must inherit this class if it can provide the Q function (i.e. action-value function).

### get\_q\_value <Badge text="QValues" type="tip"/>

<skdecide-signature name= "get_q_value" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'D.T_value'}"></skdecide-signature>

Get the estimated on-policy Q value of the given observation and action.

In mathematical terms, for a fully observable domain, this function estimates:
$$Q^\pi(s,a)=\underset{\tau\sim\pi}{\mathbb{E}}[R(\tau)|s_0=s,a_0=a]$$
where $\pi$ is the current policy, any $\tau=(s_0,a_0, s_1, a_1, ...)$ represents a trajectory sampled from
the policy, $R(\tau)$ is the return (cumulative reward) and $s_0$/$a_0$ the initial state/action for the
trajectories.

#### Parameters
- **observation**: The observation to consider.
- **action**: The action to consider.

#### Returns
The estimated on-policy Q value of the given observation and action.

### get\_utility <Badge text="Utilities" type="warn"/>

<skdecide-signature name= "get_utility" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'D.T_value'}"></skdecide-signature>

Get the estimated on-policy utility of the given observation.

In mathematical terms, for a fully observable domain, this function estimates:
$$V^\pi(s)=\underset{\tau\sim\pi}{\mathbb{E}}[R(\tau)|s_0=s]$$
where $\pi$ is the current policy, any $\tau=(s_0,a_0, s_1, a_1, ...)$ represents a trajectory sampled from
the policy, $R(\tau)$ is the return (cumulative reward) and $s_0$ the initial state for the trajectories.

#### Parameters
- **observation**: The observation to consider.

#### Returns
The estimated on-policy utility of the given observation.

### \_get\_q\_value <Badge text="QValues" type="tip"/>

<skdecide-signature name= "_get_q_value" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'D.T_value'}"></skdecide-signature>

Get the estimated on-policy Q value of the given observation and action.

In mathematical terms, for a fully observable domain, this function estimates:
$$Q^\pi(s,a)=\underset{\tau\sim\pi}{\mathbb{E}}[R(\tau)|s_0=s,a_0=a]$$
where $\pi$ is the current policy, any $\tau=(s_0,a_0, s_1, a_1, ...)$ represents a trajectory sampled from
the policy, $R(\tau)$ is the return (cumulative reward) and $s_0$/$a_0$ the initial state/action for the
trajectories.

#### Parameters
- **observation**: The observation to consider.
- **action**: The action to consider.

#### Returns
The estimated on-policy Q value of the given observation and action.

### \_get\_utility <Badge text="Utilities" type="warn"/>

<skdecide-signature name= "_get_utility" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'D.T_value'}"></skdecide-signature>

Get the estimated on-policy utility of the given observation.

In mathematical terms, for a fully observable domain, this function estimates:
$$V^\pi(s)=\underset{\tau\sim\pi}{\mathbb{E}}[R(\tau)|s_0=s]$$
where $\pi$ is the current policy, any $\tau=(s_0,a_0, s_1, a_1, ...)$ represents a trajectory sampled from
the policy, $R(\tau)$ is the return (cumulative reward) and $s_0$ the initial state for the trajectories.

#### Parameters
- **observation**: The observation to consider.

#### Returns
The estimated on-policy utility of the given observation.

