# builders.scheduling.scheduling_domains

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## SchedulingObjectiveEnum

Enum defining the different scheduling objectives:
- MAKESPAN: makespan (to be minimize)
- COST: cost of resources (to be minimized)

## D

Base class for any scheduling statefull domain

### check\_value <Badge text="Rewards" type="warn"/>

<skdecide-signature name= "check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### get\_action\_space <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_action_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the (cached) domain action space (finite or infinite set).

By default, `Events.get_action_space()` internally calls `Events._get_action_space_()` the first time and
automatically caches its value to make future calls more efficient (since the action space is assumed to be
constant).

#### Returns
The action space.

### get\_applicable\_actions <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_applicable_actions" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
the internal one if omitted.

By default, `Events.get_applicable_actions()` provides some boilerplate code and internally
calls `Events._get_applicable_actions()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of applicable actions.

### get\_enabled\_events <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_enabled_events" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history), or in the internal one if omitted.

By default, `Events.get_enabled_events()` provides some boilerplate code and internally
calls `Events._get_enabled_events()`. The boilerplate code automatically passes the `_memory` attribute instead of
the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of enabled events.

### get\_goals <Badge text="Goals" type="warn"/>

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

### get\_initial\_state <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the (cached) initial state.

By default, `DeterministicInitialized.get_initial_state()` internally
calls `DeterministicInitialized._get_initial_state_()` the first time and automatically caches its value to make
future calls more efficient (since the initial state is assumed to be constant).

#### Returns
The initial state.

### get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized.get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### get\_observation <Badge text="TransformedObservable" type="warn"/>

<skdecide-signature name= "get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable.get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### is\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_action" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is an action (i.e. a controllable event for the agents).

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
action space provided by `Events.get_action_space()`, but it can be overridden for faster implementations.
:::

#### Parameters
- **event**: The event to consider.

#### Returns
True if the event is an action (False otherwise).

### is\_applicable\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_applicable_action" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
omitted.

By default, `Events.is_applicable_action()` provides some boilerplate code and internally
calls `Events._is_applicable_action()`. The boilerplate code automatically passes the `_memory` attribute instead
of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the action is applicable (False otherwise).

### is\_enabled\_event <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_enabled_event" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an uncontrollable event is enabled in the given memory (state or history), or in the
internal one if omitted.

By default, `Events.is_enabled_event()` provides some boilerplate code and internally
calls `Events._is_enabled_event()`. The boilerplate code automatically passes the `_memory` attribute instead of
the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the event is enabled (False otherwise).

### is\_goal <Badge text="Goals" type="warn"/>

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

### is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable.get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable.reset()` provides some boilerplate code and internally calls `Initializable._reset()`
(which returns an initial state). The boilerplate code automatically stores the initial state into the `_memory`
attribute and samples a corresponding observation.

#### Returns
An initial observation.

### sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation.sample()` provides some boilerplate code and internally calls `Simulation._sample()`
(which returns a transition outcome). The boilerplate code automatically samples an observation corresponding to
the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation.sample()` to call the external simulator and not use
the `Simulation._sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with successive `Environment.step()`
calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain.set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain.step(my_action)
```

### solve\_with <Badge text="Domain" type="warn"/>

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver', 'annotation': 'Solver'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver**: The solver.
- **domain_factory**: A callable with no argument returning the domain to solve (factory is the domain class if None).
- **load_path**: The path to restore the solver state from (if None, the solving process will be launched instead).

#### Returns
The new solver (auto-cast to the level of the domain).

### step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment.step()` provides some boilerplate code and internally calls `Environment._step()` (which
returns a transition outcome). The boilerplate code automatically stores next state into the `_memory` attribute
and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment.step()` to call the external environment and not
use the `Environment._step()` helper function.
:::

::: warning
Before calling `Environment.step()` the first time or when the end of an episode is
reached, `Initializable.reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

### \_check\_value <Badge text="Rewards" type="warn"/>

<skdecide-signature name= "_check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### \_get\_action\_space <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_action_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the (cached) domain action space (finite or infinite set).

By default, `Events._get_action_space()` internally calls `Events._get_action_space_()` the first time and
automatically caches its value to make future calls more efficient (since the action space is assumed to be
constant).

#### Returns
The action space.

### \_get\_action\_space\_ <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_action_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the domain action space (finite or infinite set).

This is a helper function called by default from `Events._get_action_space()`, the difference being that the
result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The action space.

### \_get\_applicable\_actions <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_applicable_actions" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
the internal one if omitted.

By default, `Events._get_applicable_actions()` provides some boilerplate code and internally
calls `Events._get_applicable_actions_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of applicable actions.

### \_get\_applicable\_actions\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_applicable_actions_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history).

This is a helper function called by default from `Events._get_applicable_actions()`, the difference being that
the memory parameter is mandatory here.

#### Parameters
- **memory**: The memory to consider.

#### Returns
The space of applicable actions.

### \_get\_enabled\_events <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_enabled_events" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history), or in the internal one if omitted.

By default, `Events._get_enabled_events()` provides some boilerplate code and internally
calls `Events._get_enabled_events_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of enabled events.

### \_get\_enabled\_events\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_enabled_events_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history).

This is a helper function called by default from `Events._get_enabled_events()`, the difference being that the
memory parameter is mandatory here.

#### Parameters
- **memory**: The memory to consider.

#### Returns
The space of enabled events.

### \_get\_goals <Badge text="Goals" type="warn"/>

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

### \_get\_goals\_ <Badge text="Goals" type="warn"/>

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

### \_get\_initial\_state <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the (cached) initial state.

By default, `DeterministicInitialized._get_initial_state()` internally
calls `DeterministicInitialized._get_initial_state_()` the first time and automatically caches its value to make
future calls more efficient (since the initial state is assumed to be constant).

#### Returns
The initial state.

### \_get\_initial\_state\_ <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the initial state.

This is a helper function called by default from `DeterministicInitialized._get_initial_state()`, the difference
being that the result is not cached here.

#### Returns
The initial state.

### \_get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized._get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### \_get\_initial\_state\_distribution\_ <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_distribution_" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the probability distribution of initial states.

This is a helper function called by default from `UncertainInitialized._get_initial_state_distribution()`, the
difference being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The probability distribution of initial states.

### \_get\_memory\_maxlen <Badge text="History" type="warn"/>

<skdecide-signature name= "_get_memory_maxlen" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Get the (cached) memory max length.

By default, `FiniteHistory._get_memory_maxlen()` internally calls `FiniteHistory._get_memory_maxlen_()` the first
time and automatically caches its value to make future calls more efficient (since the memory max length is
assumed to be constant).

#### Returns
The memory max length.

### \_get\_memory\_maxlen\_ <Badge text="FiniteHistory" type="warn"/>

<skdecide-signature name= "_get_memory_maxlen_" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Get the memory max length.

This is a helper function called by default from `FiniteHistory._get_memory_maxlen()`, the difference being that
the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The memory max length.

### \_get\_observation <Badge text="TransformedObservable" type="warn"/>

<skdecide-signature name= "_get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable._get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### \_get\_observation\_space\_ <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the observation space (finite or infinite set).

This is a helper function called by default from `PartiallyObservable._get_observation_space()`, the difference
being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The observation space.

### \_init\_memory <Badge text="History" type="warn"/>

<skdecide-signature name= "_init_memory" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_memory[D.T_state]'}"></skdecide-signature>

Initialize memory (possibly with a state) according to its specification and return it.

This function is automatically called by `Initializable._reset()` to reinitialize the internal memory whenever
the domain is used as an environment.

#### Parameters
- **state**: An optional state to initialize the memory with (typically the initial state).

#### Returns
The new initialized memory.

### \_is\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_action" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is an action (i.e. a controllable event for the agents).

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
action space provided by `Events._get_action_space()`, but it can be overridden for faster implementations.
:::

#### Parameters
- **event**: The event to consider.

#### Returns
True if the event is an action (False otherwise).

### \_is\_applicable\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_applicable_action" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
omitted.

By default, `Events._is_applicable_action()` provides some boilerplate code and internally
calls `Events._is_applicable_action_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the action is applicable (False otherwise).

### \_is\_applicable\_action\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_applicable_action_from" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history).

This is a helper function called by default from `Events._is_applicable_action()`, the difference being that the
memory parameter is mandatory here.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the space of
applicable actions provided by `Events._get_applicable_actions_from()`, but it can be overridden for faster
implementations.
:::

#### Parameters
- **memory**: The memory to consider.

#### Returns
True if the action is applicable (False otherwise).

### \_is\_enabled\_event <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_enabled_event" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an uncontrollable event is enabled in the given memory (state or history), or in the
internal one if omitted.

By default, `Events._is_enabled_event()` provides some boilerplate code and internally
calls `Events._is_enabled_event_from()`. The boilerplate code automatically passes the `_memory` attribute instead
of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the event is enabled (False otherwise).

### \_is\_enabled\_event\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_enabled_event_from" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is enabled in the given memory (state or history).

This is a helper function called by default from `Events._is_enabled_event()`, the difference being that the
memory parameter is mandatory here.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the space of
enabled events provided by `Events._get_enabled_events_from()`, but it can be overridden for faster
implementations.
:::

#### Parameters
- **memory**: The memory to consider.

#### Returns
True if the event is enabled (False otherwise).

### \_is\_goal <Badge text="Goals" type="warn"/>

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

### \_is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable._get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### \_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable._reset()` provides some boilerplate code and internally
calls `Initializable._state_reset()` (which returns an initial state). The boilerplate code automatically stores
the initial state into the `_memory` attribute and samples a corresponding observation.

#### Returns
An initial observation.

### \_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation._sample()` provides some boilerplate code and internally
calls `Simulation._state_sample()` (which returns a transition outcome). The boilerplate code automatically
samples an observation corresponding to the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation._sample()` to call the external simulator and not use
the `Simulation._state_sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### \_set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with
successive `Environment._step()` calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain._set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain._step(my_action)
```

### \_state\_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_state_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Reset the state of the environment and return an initial state.

This is a helper function called by default from `Initializable._reset()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Returns
An initial state.

### \_state\_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_state_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one sample of the transition's dynamics.

This is a helper function called by default from `Simulation._sample()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The transition outcome of the sampled transition.

### \_state\_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_state_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one step of the transition's dynamics.

This is a helper function called by default from `Environment._step()`. It focuses on the state level, as opposed
to the observation one for the latter.

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The transition outcome of this step.

### \_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment._step()` provides some boilerplate code and internally
calls `Environment._state_step()` (which returns a transition outcome). The boilerplate code automatically stores
next state into the `_memory` attribute and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment._step()` to call the external environment and not
use the `Environment._state_step()` helper function.
:::

::: warning
Before calling `Environment._step()` the first time or when the end of an episode is
reached, `Initializable._reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

## SchedulingDomain

This is the highest level scheduling domain class (inheriting top-level class for each mandatory
domain characteristic).
This is where the implementation of the statefull scheduling domain is implemented,
letting to the user the possibility
to the user to define the scheduling problem without having to think of a statefull version.

### add\_to\_current\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "add_to_current_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'state'}]}"></skdecide-signature>

Samples completion conditions for a given task and add these conditions to the list of conditions in the
given state. This function should be called when a task complete.

### all\_tasks\_possible <Badge text="MixedRenewable" type="warn"/>

<skdecide-signature name= "all_tasks_possible" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}], 'return': 'bool'}"></skdecide-signature>

Return a True is for each task there is at least one mode in which the task can be executed, given the
resource configuration in the state provided as argument. Returns False otherwise.
If this function returns False, the scheduling problem is unsolvable from this state.
This is to cope with the use of non-renable resources that may lead to state from which a
task will not be possible anymore.

### check\_if\_action\_can\_be\_started <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "check_if_action_can_be_started" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}], 'return': 'Tuple[bool, Dict[str, int]]'}"></skdecide-signature>

Check if a start or resume action can be applied. It returns a boolean and a dictionary of resources to use.
        

### check\_unique\_resource\_names <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "check_unique_resource_names" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return True if there are no duplicates in resource names across both resource types
and resource units name lists.

### check\_value <Badge text="Rewards" type="warn"/>

<skdecide-signature name= "check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### find\_one\_ressource\_to\_do\_one\_task <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "find_one_ressource_to_do_one_task" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'int'}], 'return': 'List[str]'}"></skdecide-signature>

For the common case when it is possible to do the task by one resource unit.
For general case, it might just return no possible ressource unit.

### get\_action\_space <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_action_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the (cached) domain action space (finite or infinite set).

By default, `Events.get_action_space()` internally calls `Events._get_action_space_()` the first time and
automatically caches its value to make future calls more efficient (since the action space is assumed to be
constant).

#### Returns
The action space.

### get\_all\_condition\_items <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_all_condition_items" :sig="{'params': [{'name': 'self'}], 'return': 'Enum'}"></skdecide-signature>

Return an Enum with all the elements that can be used to define a condition.

Example:
    return
        ConditionElementsExample(Enum):
            OK = 0
            NC_PART_1_OPERATION_1 = 1
            NC_PART_1_OPERATION_2 = 2
            NC_PART_2_OPERATION_1 = 3
            NC_PART_2_OPERATION_2 = 4
            HARDWARE_ISSUE_MACHINE_A = 5
            HARDWARE_ISSUE_MACHINE_B = 6
    

### get\_all\_resources\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_all_resources_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, Dict[str, Any]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a resource type or resource unit
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {unit: {skill: (detail of skill)}} 

### get\_all\_tasks\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_all_tasks_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, Dict[str, Any]]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a task
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {task: {skill: (detail of skill)}} 

### get\_all\_unconditional\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_all_unconditional_tasks" :sig="{'params': [{'name': 'self'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids for which there are no conditions. These tasks are to be considered at
the start of a project (i.e. in the initial state). 

### get\_applicable\_actions <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_applicable_actions" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
the internal one if omitted.

By default, `Events.get_applicable_actions()` provides some boilerplate code and internally
calls `Events._get_applicable_actions()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of applicable actions.

### get\_available\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_available_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids that can be considered under the conditions defined in the given state.
Note that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks
that are remaining, or that have been completed, paused or started / resumed.

### get\_enabled\_events <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_enabled_events" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history), or in the internal one if omitted.

By default, `Events.get_enabled_events()` provides some boilerplate code and internally
calls `Events._get_enabled_events()`. The boilerplate code automatically passes the `_memory` attribute instead of
the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of enabled events.

### get\_goals <Badge text="Goals" type="warn"/>

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

### get\_initial\_state <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the (cached) initial state.

By default, `DeterministicInitialized.get_initial_state()` internally
calls `DeterministicInitialized._get_initial_state_()` the first time and automatically caches its value to make
future calls more efficient (since the initial state is assumed to be constant).

#### Returns
The initial state.

### get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized.get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### get\_max\_horizon <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "get_max_horizon" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Return the maximum time horizon (int)

### get\_mode\_costs <Badge text="WithModeCosts" type="warn"/>

<skdecide-signature name= "get_mode_costs" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, float]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode
and the value indicates the cost of execution the task in the mode.

### get\_objectives <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "get_objectives" :sig="{'params': [{'name': 'self'}], 'return': 'List[SchedulingObjectiveEnum]'}"></skdecide-signature>

Return the objectives to consider as a list. The items should be of SchedulingObjectiveEnum type.

### get\_observation <Badge text="TransformedObservable" type="warn"/>

<skdecide-signature name= "get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable.get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### get\_preallocations <Badge text="WithPreallocations" type="warn"/>

<skdecide-signature name= "get_preallocations" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[str]]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value indicates the pre-allocated resources for this task (as a list of str)

### get\_predecessors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "get_predecessors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the predecessors of the task. Successors are given as a list for a task given as a key.

### get\_resource\_cost\_per\_time\_unit <Badge text="WithResourceCosts" type="warn"/>

<skdecide-signature name= "get_resource_cost_per_time_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, float]'}"></skdecide-signature>

Return a dictionary where the key is the name of a resource (str)
and the value indicates the cost of using this resource per time unit.

### get\_resource\_renewability <Badge text="MixedRenewable" type="warn"/>

<skdecide-signature name= "get_resource_renewability" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, bool]'}"></skdecide-signature>

Return a dictionary where the key is a resource name (string)
and the value whether this resource is renewable (True) or not (False).

### get\_resource\_type\_for\_unit <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "get_resource_type_for_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, str]'}"></skdecide-signature>

Return a dictionary where the key is a resource unit name and the value a resource type name.
An empty dictionary can be used if there are no resource unit matching a resource type.

### get\_resource\_types\_names <Badge text="WithResourceTypes" type="warn"/>

<skdecide-signature name= "get_resource_types_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource types as a list.

### get\_resource\_units\_names <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "get_resource_units_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource units as a list.

### get\_skills\_names <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_skills_names" :sig="{'params': [{'name': 'self'}], 'return': 'Set[str]'}"></skdecide-signature>

Return a list of all skill names as a list of str. Skill names are defined in the 2 dictionaries returned
by the get_all_resources_skills and get_all_tasks_skills functions.

### get\_skills\_of\_resource <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_skills_of_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}], 'return': 'Dict[str, Any]'}"></skdecide-signature>

Return the skills of a given resource

### get\_skills\_of\_task <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_skills_of_task" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'int'}], 'return': 'Dict[str, Any]'}"></skdecide-signature>

Return the skill requirements for a given task

### get\_successors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "get_successors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the successors of the tasks. Successors are given as a list for a task given as a key.

### get\_task\_existence\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_task_existence_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)
for the task to be part of the schedule. If a task has no entry in the dictionary,
there is no conditions for that task.

Example:
    return
         {
            20: [get_all_condition_items().NC_PART_1_OPERATION_1],
            21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]
            22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]
         }e

 

### get\_task\_on\_completion\_added\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_task_on_completion_added_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[Distribution]]'}"></skdecide-signature>

Return a dict of list. The key of the dict is the task id and each list is composed of a list of tuples.
Each tuple contains the probability (first item in tuple) that the conditionElement (second item in tuple)
is True. The probabilities in the inner list should sum up to 1. The dictionary should only contains the keys
of tasks that can create conditions.

Example:
     return
        {
            12:
                [
                DiscreteDistribution([(ConditionElementsExample.NC_PART_1_OPERATION_1, 0.1), (ConditionElementsExample.OK, 0.9)]),
                DiscreteDistribution([(ConditionElementsExample.HARDWARE_ISSUE_MACHINE_A, 0.05), ('paper', 0.1), (ConditionElementsExample.OK, 0.95)])
                ]
        }
    

### get\_task\_paused\_non\_renewable\_resource\_returned <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "get_task_paused_non_renewable_resource_returned" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type bool indicating
if the non-renewable resources are consumed when the task is paused (False) or made available again (True).
E.g. {
        2: False  # if paused, non-renewable resource will be consumed
        5: True  # if paused, the non-renewable resource will be available again
        }

### get\_task\_preemptivity <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "get_task_preemptivity" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a boolean indicating
if the task can be paused or stopped.
E.g. {
        1: False
        2: True
        3: False
        4: False
        5: True
        6: False
        }

### get\_task\_progress <Badge text="CustomTaskProgress" type="warn"/>

<skdecide-signature name= "get_task_progress" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 't_from', 'annotation': 'int'}, {'name': 't_to', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'Optional[int]'}, {'name': 'sampled_duration', 'default': 'None', 'annotation': 'Optional[int]'}], 'return': 'float'}"></skdecide-signature>

#### Returns
 The task progress (float) between t_from and t_to.
 

### get\_task\_resuming\_type <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "get_task_resuming_type" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, ResumeType]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type ResumeType indicating
if the task can be resumed (restarted from where it was paused with no time loss)
or restarted (restarted from the start).
E.g. {
        1: ResumeType.NA
        2: ResumeType.Resume
        3: ResumeType.NA
        4: ResumeType.NA
        5: ResumeType.Restart
        6: ResumeType.NA
        }

### get\_time\_lags <Badge text="WithTimeLag" type="warn"/>

<skdecide-signature name= "get_time_lags" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, TimeLag]]'}"></skdecide-signature>

Return nested dictionaries where the first key is the id of a task (int)
and the second key is the id of another task (int).
The value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end
of the first task to the start of the second task.

e.g.
    {
        12:{
            15: TimeLag(5, 10),
            16: TimeLag(5, 20),
            17: MinimumOnlyTimeLag(5),
            18: MaximumOnlyTimeLag(15),
        }
    }

#### Returns
A dictionary of TimeLag objects.

### get\_time\_window <Badge text="WithTimeWindow" type="warn"/>

<skdecide-signature name= "get_time_window" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, TimeWindow]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value is a TimeWindow object.
Note that the max time horizon needs to be provided to the TimeWindow constructors
e.g.
    {
        1: TimeWindow(10, 15, 20, 30, self.get_max_horizon())
        2: EmptyTimeWindow(self.get_max_horizon())
        3: EndTimeWindow(20, 25, self.get_max_horizon())
        4: EndBeforeOnlyTimeWindow(40, self.get_max_horizon())
    }

#### Returns
A dictionary of TimeWindow objects.

### get\_variable\_resource\_consumption <Badge text="VariableResourceConsumption" type="warn"/>

<skdecide-signature name= "get_variable_resource_consumption" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return true if the domain has variable resource consumption,
false if the consumption of resource does not vary in time for any of the tasks

### initialize\_domain <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "initialize_domain" :sig="{'params': [{'name': 'self'}]}"></skdecide-signature>

Initialize a scheduling domain. This function needs to be called when instantiating a scheduling domain.

### is\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_action" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is an action (i.e. a controllable event for the agents).

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
action space provided by `Events.get_action_space()`, but it can be overridden for faster implementations.
:::

#### Parameters
- **event**: The event to consider.

#### Returns
True if the event is an action (False otherwise).

### is\_applicable\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_applicable_action" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
omitted.

By default, `Events.is_applicable_action()` provides some boilerplate code and internally
calls `Events._is_applicable_action()`. The boilerplate code automatically passes the `_memory` attribute instead
of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the action is applicable (False otherwise).

### is\_enabled\_event <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_enabled_event" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an uncontrollable event is enabled in the given memory (state or history), or in the
internal one if omitted.

By default, `Events.is_enabled_event()` provides some boilerplate code and internally
calls `Events._is_enabled_event()`. The boilerplate code automatically passes the `_memory` attribute instead of
the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the event is enabled (False otherwise).

### is\_goal <Badge text="Goals" type="warn"/>

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

### is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable.get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable.reset()` provides some boilerplate code and internally calls `Initializable._reset()`
(which returns an initial state). The boilerplate code automatically stores the initial state into the `_memory`
attribute and samples a corresponding observation.

#### Returns
An initial observation.

### sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation.sample()` provides some boilerplate code and internally calls `Simulation._sample()`
(which returns a transition outcome). The boilerplate code automatically samples an observation corresponding to
the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation.sample()` to call the external simulator and not use
the `Simulation._sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### sample\_completion\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "sample_completion_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}], 'return': 'List[int]'}"></skdecide-signature>

Samples the condition distributions associated with the given task and return a list of sampled
conditions.

### sample\_quantity\_resource <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "sample_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Sample an amount of resource availability (int) for the given resource
(either resource type or resource unit) at the given time. This number should be the sum of the number of
resource available at time t and the number of resource of this type consumed so far).

### sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Sample, store and return task duration for the given task in the given mode.

### set\_inplace\_environment <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "set_inplace_environment" :sig="{'params': [{'name': 'self'}, {'name': 'inplace_environment', 'annotation': 'bool'}]}"></skdecide-signature>

Activate or not the fact that the simulator modifies the given state inplace or create a copy before.
The inplace version is several times faster but will lead to bugs in graph search solver.

### set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with successive `Environment.step()`
calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain.set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain.step(my_action)
```

### solve\_with <Badge text="Domain" type="warn"/>

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver', 'annotation': 'Solver'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver**: The solver.
- **domain_factory**: A callable with no argument returning the domain to solve (factory is the domain class if None).
- **load_path**: The path to restore the solver state from (if None, the solving process will be launched instead).

#### Returns
The new solver (auto-cast to the level of the domain).

### step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment.step()` provides some boilerplate code and internally calls `Environment._step()` (which
returns a transition outcome). The boilerplate code automatically stores next state into the `_memory` attribute
and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment.step()` to call the external environment and not
use the `Environment._step()` helper function.
:::

::: warning
Before calling `Environment.step()` the first time or when the end of an episode is
reached, `Initializable.reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

### update\_complete\_dummy\_tasks <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_complete_dummy_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the status of newly started tasks whose duration is 0 from ongoing to complete.

### update\_complete\_dummy\_tasks\_simulation <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_complete_dummy_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update the status of newly started tasks whose duration is 0
from ongoing to complete.

### update\_complete\_dummy\_tasks\_uncertain <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_complete_dummy_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of newly started tasks whose duration is 0
from ongoing to complete.

### update\_complete\_tasks <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_complete_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}]}"></skdecide-signature>

Update the status of newly completed tasks in the state from ongoing to complete
and update resource availability. This function will also log in task_details the time it was complete

### update\_complete\_tasks\_simulation <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_complete_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}]}"></skdecide-signature>

In a simulated scheduling environment, update the status of newly completed tasks in the state from ongoing to complete
and update resource availability. This function will also log in task_details the time it was complete

### update\_complete\_tasks\_uncertain <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_complete_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of newly completed tasks in the state from ongoing
to complete, update resource availability and update on-completion conditions.
This function will also log in task_details the time it was complete.

### update\_conditional\_tasks <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_conditional_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update remaining tasks by checking conditions and potentially adding conditional tasks.

### update\_conditional\_tasks\_simulation <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_conditional_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update remaining tasks by checking conditions and potentially
adding conditional tasks.

### update\_conditional\_tasks\_uncertain <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_conditional_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update remaining tasks by checking conditions and potentially adding conditional tasks.

### update\_pause\_tasks <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_pause_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the status of a task from ongoing to paused if specified in the action
and update resource availability. This function will also log in task_details the time it was paused.

### update\_pause\_tasks\_simulation <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_pause_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulation scheduling environment, update the status of a task from ongoing to paused if
specified in the action and update resource availability. This function will also log in task_details
the time it was paused.

### update\_pause\_tasks\_uncertain <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_pause_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of a task from ongoing to paused if
specified in the action and update resource availability. This function will also log in task_details
the time it was paused.

### update\_progress <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_progress" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}]}"></skdecide-signature>

Update the progress of all ongoing tasks in the state.

### update\_progress\_simulation <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_progress_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}]}"></skdecide-signature>

In a simulation scheduling environment, update the progress of all ongoing tasks in the state.

### update\_progress\_uncertain <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_progress_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the progress of all ongoing tasks in the state.

### update\_resource\_availability <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_resource_availability" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update resource availability for next time step. This should be called after update_time().

### update\_resource\_availability\_simulation <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_resource_availability_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update resource availability for next time step.
This should be called after update_time().

### update\_resource\_availability\_uncertain <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_resource_availability_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update resource availability for next time step. This should be called after update_time().

### update\_resume\_tasks <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_resume_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the status of a task from paused to ongoing if specified in the action
and update resource availability. This function will also log in task_details the time it was resumed

### update\_resume\_tasks\_simulation <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_resume_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulationn scheduling environment, update the status of a task from paused to ongoing if specified
in the action and update resource availability. This function will also log in task_details the time it was
resumed.

### update\_resume\_tasks\_uncertain <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_resume_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of a task from paused to ongoing if specified
in the action and update resource availability. This function will also log in task_details the time it was
resumed.

### update\_start\_tasks <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_start_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the status of a task from remaining to ongoing if specified in the action
and update resource availability. This function will also log in task_details the time it was started.

### update\_start\_tasks\_simulation <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_start_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update the status of a task from remaining to ongoing if
specified in the action and update resource availability. This function will also log in task_details the
time it was started.

### update\_start\_tasks\_uncertain <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_start_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of a task from remaining to ongoing
if specified in the action and update resource availability.
This function returns a DsicreteDistribution of State.
This function will also log in task_details the time it was started.

### update\_time <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_time" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the time of the state if the time_progress attribute of the given EnumerableAction is True.

### update\_time\_simulation <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_time_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update the time of the state if the time_progress attribute of the
given EnumerableAction is True.

### update\_time\_uncertain <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "update_time_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the time of the state if the time_progress attribute of the given EnumerableAction is True.

### \_add\_to\_current\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_add_to_current_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'state'}]}"></skdecide-signature>

Samples completion conditions for a given task and add these conditions to the list of conditions in the
given state. This function should be called when a task complete.

### \_check\_value <Badge text="Rewards" type="warn"/>

<skdecide-signature name= "_check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### \_get\_action\_space <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_action_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the (cached) domain action space (finite or infinite set).

By default, `Events._get_action_space()` internally calls `Events._get_action_space_()` the first time and
automatically caches its value to make future calls more efficient (since the action space is assumed to be
constant).

#### Returns
The action space.

### \_get\_action\_space\_ <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_action_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

To be implemented if needed one day.

### \_get\_all\_resources\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "_get_all_resources_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, Dict[str, Any]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a resource type or resource unit
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {unit: {skill: (detail of skill)}} 

### \_get\_all\_tasks\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "_get_all_tasks_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, Dict[str, Any]]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a task
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {task: {skill: (detail of skill)}} 

### \_get\_all\_unconditional\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_get_all_unconditional_tasks" :sig="{'params': [{'name': 'self'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids for which there are no conditions. These tasks are to be considered at
the start of a project (i.e. in the initial state). 

### \_get\_applicable\_actions <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_applicable_actions" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
the internal one if omitted.

By default, `Events._get_applicable_actions()` provides some boilerplate code and internally
calls `Events._get_applicable_actions_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of applicable actions.

### \_get\_applicable\_actions\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_applicable_actions_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Returns the action space from a state.
TODO : think about a way to avoid the instaceof usage.

### \_get\_available\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_get_available_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids that can be considered under the conditions defined in the given state.
Note that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks
that are remaining, or that have been completed, paused or started / resumed.

### \_get\_enabled\_events <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_enabled_events" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history), or in the internal one if omitted.

By default, `Events._get_enabled_events()` provides some boilerplate code and internally
calls `Events._get_enabled_events_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of enabled events.

### \_get\_enabled\_events\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_enabled_events_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history).

This is a helper function called by default from `Events._get_enabled_events()`, the difference being that the
memory parameter is mandatory here.

#### Parameters
- **memory**: The memory to consider.

#### Returns
The space of enabled events.

### \_get\_goals <Badge text="Goals" type="warn"/>

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

### \_get\_goals\_ <Badge text="Goals" type="warn"/>

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

### \_get\_initial\_state <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the (cached) initial state.

By default, `DeterministicInitialized._get_initial_state()` internally
calls `DeterministicInitialized._get_initial_state_()` the first time and automatically caches its value to make
future calls more efficient (since the initial state is assumed to be constant).

#### Returns
The initial state.

### \_get\_initial\_state\_ <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Create and return an empty initial state

### \_get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized._get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### \_get\_initial\_state\_distribution\_ <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_distribution_" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the probability distribution of initial states.

This is a helper function called by default from `UncertainInitialized._get_initial_state_distribution()`, the
difference being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The probability distribution of initial states.

### \_get\_max\_horizon <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "_get_max_horizon" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Return the maximum time horizon (int)

### \_get\_memory\_maxlen <Badge text="History" type="warn"/>

<skdecide-signature name= "_get_memory_maxlen" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Get the (cached) memory max length.

By default, `FiniteHistory._get_memory_maxlen()` internally calls `FiniteHistory._get_memory_maxlen_()` the first
time and automatically caches its value to make future calls more efficient (since the memory max length is
assumed to be constant).

#### Returns
The memory max length.

### \_get\_memory\_maxlen\_ <Badge text="FiniteHistory" type="warn"/>

<skdecide-signature name= "_get_memory_maxlen_" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Get the memory max length.

This is a helper function called by default from `FiniteHistory._get_memory_maxlen()`, the difference being that
the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The memory max length.

### \_get\_mode\_costs <Badge text="WithModeCosts" type="warn"/>

<skdecide-signature name= "_get_mode_costs" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, float]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode
and the value indicates the cost of execution the task in the mode.

### \_get\_next\_state <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "_get_next_state" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'D.T_state'}"></skdecide-signature>

This function will be used if the domain is defined with DeterministicTransitions. This function will be ignored
if the domain is defined as having UncertainTransitions or Simulation. 

### \_get\_next\_state\_distribution <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "_get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

This function will be used if the domain is defined with UncertainTransitions. This function will be ignored
if the domain is defined as a Simulation. This function may also be used by uncertainty-specialised solvers
 on deterministic domains.

### \_get\_objectives <Badge text="SchedulingDomain" type="tip"/>

<skdecide-signature name= "_get_objectives" :sig="{'params': [{'name': 'self'}], 'return': 'List[SchedulingObjectiveEnum]'}"></skdecide-signature>

Return the objectives to consider as a list. The items should be of SchedulingObjectiveEnum type.

### \_get\_observation <Badge text="TransformedObservable" type="warn"/>

<skdecide-signature name= "_get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable._get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### \_get\_observation\_space\_ <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

To be implemented if needed one day.

### \_get\_preallocations <Badge text="WithPreallocations" type="warn"/>

<skdecide-signature name= "_get_preallocations" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[str]]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value indicates the pre-allocated resources for this task (as a list of str)

### \_get\_predecessors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "_get_predecessors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the predecessors of the task. Successors are given as a list for a task given as a key.

### \_get\_resource\_cost\_per\_time\_unit <Badge text="WithResourceCosts" type="warn"/>

<skdecide-signature name= "_get_resource_cost_per_time_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, float]'}"></skdecide-signature>

Return a dictionary where the key is the name of a resource (str)
and the value indicates the cost of using this resource per time unit.

### \_get\_resource\_renewability <Badge text="MixedRenewable" type="warn"/>

<skdecide-signature name= "_get_resource_renewability" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, bool]'}"></skdecide-signature>

Return a dictionary where the key is a resource name (string)
and the value whether this resource is renewable (True) or not (False).

### \_get\_resource\_type\_for\_unit <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "_get_resource_type_for_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, str]'}"></skdecide-signature>

Return a dictionary where the key is a resource unit name and the value a resource type name.
An empty dictionary can be used if there are no resource unit matching a resource type.

### \_get\_resource\_types\_names <Badge text="WithResourceTypes" type="warn"/>

<skdecide-signature name= "_get_resource_types_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource types as a list.

### \_get\_resource\_units\_names <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "_get_resource_units_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource units as a list.

### \_get\_successors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "_get_successors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the successors of the tasks. Successors are given as a list for a task given as a key.

### \_get\_task\_existence\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_get_task_existence_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)
for the task to be part of the schedule. If a task has no entry in the dictionary,
there is no conditions for that task.

Example:
    return
         {
            20: [get_all_condition_items().NC_PART_1_OPERATION_1],
            21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]
            22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]
         }e

### \_get\_task\_paused\_non\_renewable\_resource\_returned <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "_get_task_paused_non_renewable_resource_returned" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type bool indicating
if the non-renewable resources are consumed when the task is paused (False) or made available again (True).
E.g. {
        2: False  # if paused, non-renewable resource will be consumed
        5: True  # if paused, the non-renewable resource will be available again
        }

### \_get\_task\_preemptivity <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "_get_task_preemptivity" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a boolean indicating
if the task can be paused or stopped.
E.g. {
        1: False
        2: True
        3: False
        4: False
        5: True
        6: False
        }

### \_get\_task\_progress <Badge text="CustomTaskProgress" type="warn"/>

<skdecide-signature name= "_get_task_progress" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 't_from', 'annotation': 'int'}, {'name': 't_to', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'Optional[int]'}, {'name': 'sampled_duration', 'default': 'None', 'annotation': 'Optional[int]'}], 'return': 'float'}"></skdecide-signature>

#### Returns
 The task progress (float) between t_from and t_to.
 

### \_get\_task\_resuming\_type <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "_get_task_resuming_type" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, ResumeType]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type ResumeType indicating
if the task can be resumed (restarted from where it was paused with no time loss)
or restarted (restarted from the start).
E.g. {
        1: ResumeType.NA
        2: ResumeType.Resume
        3: ResumeType.NA
        4: ResumeType.NA
        5: ResumeType.Restart
        6: ResumeType.NA
        }

### \_get\_tasks\_ids <Badge text="MultiMode" type="warn"/>

<skdecide-signature name= "_get_tasks_ids" :sig="{'params': [{'name': 'self'}], 'return': 'Union[Set[int], Dict[int, Any], List[int]]'}"></skdecide-signature>

Return a set or dict of int = id of tasks

### \_get\_tasks\_modes <Badge text="MultiMode" type="warn"/>

<skdecide-signature name= "_get_tasks_modes" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, ModeConsumption]]'}"></skdecide-signature>

Return a nested dictionary where the first key is a task id and the second key is a mode id.
 The value is a Mode object defining the resource consumption.
If the domain is an instance of VariableResourceConsumption, VaryingModeConsumption objects should be used.
If this is not the case (i.e. the domain is an instance of ConstantResourceConsumption),
then ConstantModeConsumption should be used.

E.g. with constant resource consumption
    {
        12: {
                1: ConstantModeConsumption({'rt_1': 2, 'rt_2': 0, 'ru_1': 1}),
                2: ConstantModeConsumption({'rt_1': 0, 'rt_2': 3, 'ru_1': 1}),
            }
    }

E.g. with time varying resource consumption
    {
    12: {
        1: VaryingModeConsumption({'rt_1': [2,2,2,2,3], 'rt_2': [0,0,0,0,0], 'ru_1': [1,1,1,1,1]}),
        2: VaryingModeConsumption({'rt_1': [1,1,1,1,2,2,2], 'rt_2': [0,0,0,0,0,0,0], 'ru_1': [1,1,1,1,1,1,1]}),
        }
    }

### \_get\_time\_lags <Badge text="WithTimeLag" type="warn"/>

<skdecide-signature name= "_get_time_lags" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, TimeLag]]'}"></skdecide-signature>

Return nested dictionaries where the first key is the id of a task (int)
and the second key is the id of another task (int).
The value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end
of the first task to the start of the second task.

e.g.
    {
        12:{
            15: TimeLag(5, 10),
            16: TimeLag(5, 20),
            17: MinimumOnlyTimeLag(5),
            18: MaximumOnlyTimeLag(15),
        }
    }

#### Returns
A dictionary of TimeLag objects.

### \_get\_time\_window <Badge text="WithTimeWindow" type="warn"/>

<skdecide-signature name= "_get_time_window" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, TimeWindow]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value is a TimeWindow object.
Note that the max time horizon needs to be provided to the TimeWindow constructors
e.g.
    {
        1: TimeWindow(10, 15, 20, 30, self.get_max_horizon())
        2: EmptyTimeWindow(self.get_max_horizon())
        3: EndTimeWindow(20, 25, self.get_max_horizon())
        4: EndBeforeOnlyTimeWindow(40, self.get_max_horizon())
    }

#### Returns
A dictionary of TimeWindow objects.

### \_get\_variable\_resource\_consumption <Badge text="VariableResourceConsumption" type="warn"/>

<skdecide-signature name= "_get_variable_resource_consumption" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return true if the domain has variable resource consumption,
false if the consumption of resource does not vary in time for any of the tasks

### \_init\_memory <Badge text="History" type="warn"/>

<skdecide-signature name= "_init_memory" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_memory[D.T_state]'}"></skdecide-signature>

Initialize memory (possibly with a state) according to its specification and return it.

This function is automatically called by `Initializable._reset()` to reinitialize the internal memory whenever
the domain is used as an environment.

#### Parameters
- **state**: An optional state to initialize the memory with (typically the initial state).

#### Returns
The new initialized memory.

### \_is\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_action" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is an action (i.e. a controllable event for the agents).

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
action space provided by `Events._get_action_space()`, but it can be overridden for faster implementations.
:::

#### Parameters
- **event**: The event to consider.

#### Returns
True if the event is an action (False otherwise).

### \_is\_applicable\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_applicable_action" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
omitted.

By default, `Events._is_applicable_action()` provides some boilerplate code and internally
calls `Events._is_applicable_action_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the action is applicable (False otherwise).

### \_is\_applicable\_action\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_applicable_action_from" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history).

This is a helper function called by default from `Events._is_applicable_action()`, the difference being that the
memory parameter is mandatory here.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the space of
applicable actions provided by `Events._get_applicable_actions_from()`, but it can be overridden for faster
implementations.
:::

#### Parameters
- **memory**: The memory to consider.

#### Returns
True if the action is applicable (False otherwise).

### \_is\_enabled\_event <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_enabled_event" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an uncontrollable event is enabled in the given memory (state or history), or in the
internal one if omitted.

By default, `Events._is_enabled_event()` provides some boilerplate code and internally
calls `Events._is_enabled_event_from()`. The boilerplate code automatically passes the `_memory` attribute instead
of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the event is enabled (False otherwise).

### \_is\_enabled\_event\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_enabled_event_from" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is enabled in the given memory (state or history).

This is a helper function called by default from `Events._is_enabled_event()`, the difference being that the
memory parameter is mandatory here.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the space of
enabled events provided by `Events._get_enabled_events_from()`, but it can be overridden for faster
implementations.
:::

#### Parameters
- **memory**: The memory to consider.

#### Returns
True if the event is enabled (False otherwise).

### \_is\_goal <Badge text="Goals" type="warn"/>

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

### \_is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable._get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### \_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable._reset()` provides some boilerplate code and internally
calls `Initializable._state_reset()` (which returns an initial state). The boilerplate code automatically stores
the initial state into the `_memory` attribute and samples a corresponding observation.

#### Returns
An initial observation.

### \_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation._sample()` provides some boilerplate code and internally
calls `Simulation._state_sample()` (which returns a transition outcome). The boilerplate code automatically
samples an observation corresponding to the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation._sample()` to call the external simulator and not use
the `Simulation._state_sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### \_sample\_completion\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_sample_completion_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}], 'return': 'List[int]'}"></skdecide-signature>

Samples the condition distributions associated with the given task and return a list of sampled
conditions.

### \_sample\_quantity\_resource <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "_sample_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Sample an amount of resource availability (int) for the given resource
(either resource type or resource unit) at the given time. This number should be the sum of the number of
resource available at time t and the number of resource of this type consumed so far).

### \_sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "_sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode.

### \_set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with
successive `Environment._step()` calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain._set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain._step(my_action)
```

### \_state\_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_state_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Reset the state of the environment and return an initial state.

This is a helper function called by default from `Initializable._reset()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Returns
An initial state.

### \_state\_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_state_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

This function will be used if the domain is defined as a Simulation (i.e. transitions are defined by call to
a simulation). This function may also be used by simulation-based solvers on non-Simulation domains.

### \_state\_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_state_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one step of the transition's dynamics.

This is a helper function called by default from `Environment._step()`. It focuses on the state level, as opposed
to the observation one for the latter.

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The transition outcome of this step.

### \_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment._step()` provides some boilerplate code and internally
calls `Environment._state_step()` (which returns a transition outcome). The boilerplate code automatically stores
next state into the `_memory` attribute and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment._step()` to call the external environment and not
use the `Environment._state_step()` helper function.
:::

::: warning
Before calling `Environment._step()` the first time or when the end of an episode is
reached, `Initializable._reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

## D\_det

Base class for deterministic scheduling problems

### check\_value <Badge text="Rewards" type="warn"/>

<skdecide-signature name= "check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### get\_action\_space <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_action_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the (cached) domain action space (finite or infinite set).

By default, `Events.get_action_space()` internally calls `Events._get_action_space_()` the first time and
automatically caches its value to make future calls more efficient (since the action space is assumed to be
constant).

#### Returns
The action space.

### get\_applicable\_actions <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_applicable_actions" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
the internal one if omitted.

By default, `Events.get_applicable_actions()` provides some boilerplate code and internally
calls `Events._get_applicable_actions()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of applicable actions.

### get\_enabled\_events <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_enabled_events" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history), or in the internal one if omitted.

By default, `Events.get_enabled_events()` provides some boilerplate code and internally
calls `Events._get_enabled_events()`. The boilerplate code automatically passes the `_memory` attribute instead of
the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of enabled events.

### get\_goals <Badge text="Goals" type="warn"/>

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

### get\_initial\_state <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the (cached) initial state.

By default, `DeterministicInitialized.get_initial_state()` internally
calls `DeterministicInitialized._get_initial_state_()` the first time and automatically caches its value to make
future calls more efficient (since the initial state is assumed to be constant).

#### Returns
The initial state.

### get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized.get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### get\_next\_state <Badge text="DeterministicTransitions" type="warn"/>

<skdecide-signature name= "get_next_state" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the next state given a memory and action.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The deterministic next state.

### get\_next\_state\_distribution <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'DiscreteDistribution[D.T_state]'}"></skdecide-signature>

Get the discrete probability distribution of next state given a memory and action.

::: tip
In the Markovian case (memory only holds last state $s$), given an action $a$, this function can
be mathematically represented by $P(S'|s, a)$, where $S'$ is the next state random variable.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The discrete probability distribution of next state.

### get\_observation <Badge text="TransformedObservable" type="warn"/>

<skdecide-signature name= "get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable.get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### get\_transition\_value <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "get_transition_value" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_agent[TransitionValue[D.T_value]]'}"></skdecide-signature>

Get the value (reward or cost) of a transition.

The transition to consider is defined by the function parameters.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `UncertainTransitions._is_transition_value_dependent_on_next_state_()` to return
False. This information can then be exploited by solvers to avoid computing next state to evaluate a
transition value (more efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
The transition value (reward or cost).

### is\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_action" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is an action (i.e. a controllable event for the agents).

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
action space provided by `Events.get_action_space()`, but it can be overridden for faster implementations.
:::

#### Parameters
- **event**: The event to consider.

#### Returns
True if the event is an action (False otherwise).

### is\_applicable\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_applicable_action" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
omitted.

By default, `Events.is_applicable_action()` provides some boilerplate code and internally
calls `Events._is_applicable_action()`. The boilerplate code automatically passes the `_memory` attribute instead
of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the action is applicable (False otherwise).

### is\_enabled\_event <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_enabled_event" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an uncontrollable event is enabled in the given memory (state or history), or in the
internal one if omitted.

By default, `Events.is_enabled_event()` provides some boilerplate code and internally
calls `Events._is_enabled_event()`. The boilerplate code automatically passes the `_memory` attribute instead of
the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the event is enabled (False otherwise).

### is\_goal <Badge text="Goals" type="warn"/>

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

### is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable.get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### is\_terminal <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "is_terminal" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether a state is terminal.

A terminal state is a state with no outgoing transition (except to itself with value 0).

#### Parameters
- **state**: The state to consider.

#### Returns
True if the state is terminal (False otherwise).

### is\_transition\_value\_dependent\_on\_next\_state <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "is_transition_value_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether get_transition_value() requires the next_state parameter for its computation (cached).

By default, `UncertainTransitions.is_transition_value_dependent_on_next_state()` internally
calls `UncertainTransitions._is_transition_value_dependent_on_next_state_()` the first time and automatically
caches its value to make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable.reset()` provides some boilerplate code and internally calls `Initializable._reset()`
(which returns an initial state). The boilerplate code automatically stores the initial state into the `_memory`
attribute and samples a corresponding observation.

#### Returns
An initial observation.

### sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation.sample()` provides some boilerplate code and internally calls `Simulation._sample()`
(which returns a transition outcome). The boilerplate code automatically samples an observation corresponding to
the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation.sample()` to call the external simulator and not use
the `Simulation._sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with successive `Environment.step()`
calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain.set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain.step(my_action)
```

### solve\_with <Badge text="Domain" type="warn"/>

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver', 'annotation': 'Solver'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver**: The solver.
- **domain_factory**: A callable with no argument returning the domain to solve (factory is the domain class if None).
- **load_path**: The path to restore the solver state from (if None, the solving process will be launched instead).

#### Returns
The new solver (auto-cast to the level of the domain).

### step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment.step()` provides some boilerplate code and internally calls `Environment._step()` (which
returns a transition outcome). The boilerplate code automatically stores next state into the `_memory` attribute
and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment.step()` to call the external environment and not
use the `Environment._step()` helper function.
:::

::: warning
Before calling `Environment.step()` the first time or when the end of an episode is
reached, `Initializable.reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

### \_check\_value <Badge text="Rewards" type="warn"/>

<skdecide-signature name= "_check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### \_get\_action\_space <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_action_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the (cached) domain action space (finite or infinite set).

By default, `Events._get_action_space()` internally calls `Events._get_action_space_()` the first time and
automatically caches its value to make future calls more efficient (since the action space is assumed to be
constant).

#### Returns
The action space.

### \_get\_action\_space\_ <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_action_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the domain action space (finite or infinite set).

This is a helper function called by default from `Events._get_action_space()`, the difference being that the
result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The action space.

### \_get\_applicable\_actions <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_applicable_actions" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
the internal one if omitted.

By default, `Events._get_applicable_actions()` provides some boilerplate code and internally
calls `Events._get_applicable_actions_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of applicable actions.

### \_get\_applicable\_actions\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_applicable_actions_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history).

This is a helper function called by default from `Events._get_applicable_actions()`, the difference being that
the memory parameter is mandatory here.

#### Parameters
- **memory**: The memory to consider.

#### Returns
The space of applicable actions.

### \_get\_enabled\_events <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_enabled_events" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history), or in the internal one if omitted.

By default, `Events._get_enabled_events()` provides some boilerplate code and internally
calls `Events._get_enabled_events_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of enabled events.

### \_get\_enabled\_events\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_enabled_events_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history).

This is a helper function called by default from `Events._get_enabled_events()`, the difference being that the
memory parameter is mandatory here.

#### Parameters
- **memory**: The memory to consider.

#### Returns
The space of enabled events.

### \_get\_goals <Badge text="Goals" type="warn"/>

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

### \_get\_goals\_ <Badge text="Goals" type="warn"/>

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

### \_get\_initial\_state <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the (cached) initial state.

By default, `DeterministicInitialized._get_initial_state()` internally
calls `DeterministicInitialized._get_initial_state_()` the first time and automatically caches its value to make
future calls more efficient (since the initial state is assumed to be constant).

#### Returns
The initial state.

### \_get\_initial\_state\_ <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the initial state.

This is a helper function called by default from `DeterministicInitialized._get_initial_state()`, the difference
being that the result is not cached here.

#### Returns
The initial state.

### \_get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized._get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### \_get\_initial\_state\_distribution\_ <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_distribution_" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the probability distribution of initial states.

This is a helper function called by default from `UncertainInitialized._get_initial_state_distribution()`, the
difference being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The probability distribution of initial states.

### \_get\_memory\_maxlen <Badge text="History" type="warn"/>

<skdecide-signature name= "_get_memory_maxlen" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Get the (cached) memory max length.

By default, `FiniteHistory._get_memory_maxlen()` internally calls `FiniteHistory._get_memory_maxlen_()` the first
time and automatically caches its value to make future calls more efficient (since the memory max length is
assumed to be constant).

#### Returns
The memory max length.

### \_get\_memory\_maxlen\_ <Badge text="FiniteHistory" type="warn"/>

<skdecide-signature name= "_get_memory_maxlen_" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Get the memory max length.

This is a helper function called by default from `FiniteHistory._get_memory_maxlen()`, the difference being that
the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The memory max length.

### \_get\_next\_state <Badge text="DeterministicTransitions" type="warn"/>

<skdecide-signature name= "_get_next_state" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the next state given a memory and action.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The deterministic next state.

### \_get\_next\_state\_distribution <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'SingleValueDistribution[D.T_state]'}"></skdecide-signature>

Get the discrete probability distribution of next state given a memory and action.

::: tip
In the Markovian case (memory only holds last state $s$), given an action $a$, this function can
be mathematically represented by $P(S'|s, a)$, where $S'$ is the next state random variable.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The discrete probability distribution of next state.

### \_get\_observation <Badge text="TransformedObservable" type="warn"/>

<skdecide-signature name= "_get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable._get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### \_get\_observation\_space\_ <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the observation space (finite or infinite set).

This is a helper function called by default from `PartiallyObservable._get_observation_space()`, the difference
being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The observation space.

### \_get\_transition\_value <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_get_transition_value" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_agent[TransitionValue[D.T_value]]'}"></skdecide-signature>

Get the value (reward or cost) of a transition.

The transition to consider is defined by the function parameters.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `UncertainTransitions._is_transition_value_dependent_on_next_state_()` to return
False. This information can then be exploited by solvers to avoid computing next state to evaluate a
transition value (more efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
The transition value (reward or cost).

### \_init\_memory <Badge text="History" type="warn"/>

<skdecide-signature name= "_init_memory" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_memory[D.T_state]'}"></skdecide-signature>

Initialize memory (possibly with a state) according to its specification and return it.

This function is automatically called by `Initializable._reset()` to reinitialize the internal memory whenever
the domain is used as an environment.

#### Parameters
- **state**: An optional state to initialize the memory with (typically the initial state).

#### Returns
The new initialized memory.

### \_is\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_action" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is an action (i.e. a controllable event for the agents).

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
action space provided by `Events._get_action_space()`, but it can be overridden for faster implementations.
:::

#### Parameters
- **event**: The event to consider.

#### Returns
True if the event is an action (False otherwise).

### \_is\_applicable\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_applicable_action" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
omitted.

By default, `Events._is_applicable_action()` provides some boilerplate code and internally
calls `Events._is_applicable_action_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the action is applicable (False otherwise).

### \_is\_applicable\_action\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_applicable_action_from" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history).

This is a helper function called by default from `Events._is_applicable_action()`, the difference being that the
memory parameter is mandatory here.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the space of
applicable actions provided by `Events._get_applicable_actions_from()`, but it can be overridden for faster
implementations.
:::

#### Parameters
- **memory**: The memory to consider.

#### Returns
True if the action is applicable (False otherwise).

### \_is\_enabled\_event <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_enabled_event" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an uncontrollable event is enabled in the given memory (state or history), or in the
internal one if omitted.

By default, `Events._is_enabled_event()` provides some boilerplate code and internally
calls `Events._is_enabled_event_from()`. The boilerplate code automatically passes the `_memory` attribute instead
of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the event is enabled (False otherwise).

### \_is\_enabled\_event\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_enabled_event_from" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is enabled in the given memory (state or history).

This is a helper function called by default from `Events._is_enabled_event()`, the difference being that the
memory parameter is mandatory here.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the space of
enabled events provided by `Events._get_enabled_events_from()`, but it can be overridden for faster
implementations.
:::

#### Parameters
- **memory**: The memory to consider.

#### Returns
True if the event is enabled (False otherwise).

### \_is\_goal <Badge text="Goals" type="warn"/>

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

### \_is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable._get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### \_is\_terminal <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_terminal" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether a state is terminal.

A terminal state is a state with no outgoing transition (except to itself with value 0).

#### Parameters
- **state**: The state to consider.

#### Returns
True if the state is terminal (False otherwise).

### \_is\_transition\_value\_dependent\_on\_next\_state <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_transition_value_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether _get_transition_value() requires the next_state parameter for its computation (cached).

By default, `UncertainTransitions._is_transition_value_dependent_on_next_state()` internally
calls `UncertainTransitions._is_transition_value_dependent_on_next_state_()` the first time and automatically
caches its value to make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### \_is\_transition\_value\_dependent\_on\_next\_state\_ <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_transition_value_dependent_on_next_state_" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether _get_transition_value() requires the next_state parameter for its computation.

This is a helper function called by default
from `UncertainTransitions._is_transition_value_dependent_on_next_state()`, the difference being that the result
is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### \_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable._reset()` provides some boilerplate code and internally
calls `Initializable._state_reset()` (which returns an initial state). The boilerplate code automatically stores
the initial state into the `_memory` attribute and samples a corresponding observation.

#### Returns
An initial observation.

### \_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation._sample()` provides some boilerplate code and internally
calls `Simulation._state_sample()` (which returns a transition outcome). The boilerplate code automatically
samples an observation corresponding to the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation._sample()` to call the external simulator and not use
the `Simulation._state_sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### \_set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with
successive `Environment._step()` calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain._set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain._step(my_action)
```

### \_state\_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_state_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Reset the state of the environment and return an initial state.

This is a helper function called by default from `Initializable._reset()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Returns
An initial state.

### \_state\_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_state_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one sample of the transition's dynamics.

This is a helper function called by default from `Simulation._sample()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The transition outcome of the sampled transition.

### \_state\_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_state_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one step of the transition's dynamics.

This is a helper function called by default from `Environment._step()`. It focuses on the state level, as opposed
to the observation one for the latter.

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The transition outcome of this step.

### \_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment._step()` provides some boilerplate code and internally
calls `Environment._state_step()` (which returns a transition outcome). The boilerplate code automatically stores
next state into the `_memory` attribute and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment._step()` to call the external environment and not
use the `Environment._state_step()` helper function.
:::

::: warning
Before calling `Environment._step()` the first time or when the end of an episode is
reached, `Initializable._reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

## D\_uncertain

Base class for uncertain scheduling problems where we can compute distributions

### check\_value <Badge text="Rewards" type="warn"/>

<skdecide-signature name= "check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### get\_action\_space <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_action_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the (cached) domain action space (finite or infinite set).

By default, `Events.get_action_space()` internally calls `Events._get_action_space_()` the first time and
automatically caches its value to make future calls more efficient (since the action space is assumed to be
constant).

#### Returns
The action space.

### get\_applicable\_actions <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_applicable_actions" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
the internal one if omitted.

By default, `Events.get_applicable_actions()` provides some boilerplate code and internally
calls `Events._get_applicable_actions()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of applicable actions.

### get\_enabled\_events <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_enabled_events" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history), or in the internal one if omitted.

By default, `Events.get_enabled_events()` provides some boilerplate code and internally
calls `Events._get_enabled_events()`. The boilerplate code automatically passes the `_memory` attribute instead of
the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of enabled events.

### get\_goals <Badge text="Goals" type="warn"/>

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

### get\_initial\_state <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the (cached) initial state.

By default, `DeterministicInitialized.get_initial_state()` internally
calls `DeterministicInitialized._get_initial_state_()` the first time and automatically caches its value to make
future calls more efficient (since the initial state is assumed to be constant).

#### Returns
The initial state.

### get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized.get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### get\_next\_state\_distribution <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the probability distribution of next state given a memory and action.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The probability distribution of next state.

### get\_observation <Badge text="TransformedObservable" type="warn"/>

<skdecide-signature name= "get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable.get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### get\_transition\_value <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "get_transition_value" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_agent[TransitionValue[D.T_value]]'}"></skdecide-signature>

Get the value (reward or cost) of a transition.

The transition to consider is defined by the function parameters.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `UncertainTransitions._is_transition_value_dependent_on_next_state_()` to return
False. This information can then be exploited by solvers to avoid computing next state to evaluate a
transition value (more efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
The transition value (reward or cost).

### is\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_action" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is an action (i.e. a controllable event for the agents).

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
action space provided by `Events.get_action_space()`, but it can be overridden for faster implementations.
:::

#### Parameters
- **event**: The event to consider.

#### Returns
True if the event is an action (False otherwise).

### is\_applicable\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_applicable_action" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
omitted.

By default, `Events.is_applicable_action()` provides some boilerplate code and internally
calls `Events._is_applicable_action()`. The boilerplate code automatically passes the `_memory` attribute instead
of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the action is applicable (False otherwise).

### is\_enabled\_event <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_enabled_event" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an uncontrollable event is enabled in the given memory (state or history), or in the
internal one if omitted.

By default, `Events.is_enabled_event()` provides some boilerplate code and internally
calls `Events._is_enabled_event()`. The boilerplate code automatically passes the `_memory` attribute instead of
the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the event is enabled (False otherwise).

### is\_goal <Badge text="Goals" type="warn"/>

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

### is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable.get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### is\_terminal <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "is_terminal" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether a state is terminal.

A terminal state is a state with no outgoing transition (except to itself with value 0).

#### Parameters
- **state**: The state to consider.

#### Returns
True if the state is terminal (False otherwise).

### is\_transition\_value\_dependent\_on\_next\_state <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "is_transition_value_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether get_transition_value() requires the next_state parameter for its computation (cached).

By default, `UncertainTransitions.is_transition_value_dependent_on_next_state()` internally
calls `UncertainTransitions._is_transition_value_dependent_on_next_state_()` the first time and automatically
caches its value to make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable.reset()` provides some boilerplate code and internally calls `Initializable._reset()`
(which returns an initial state). The boilerplate code automatically stores the initial state into the `_memory`
attribute and samples a corresponding observation.

#### Returns
An initial observation.

### sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation.sample()` provides some boilerplate code and internally calls `Simulation._sample()`
(which returns a transition outcome). The boilerplate code automatically samples an observation corresponding to
the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation.sample()` to call the external simulator and not use
the `Simulation._sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with successive `Environment.step()`
calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain.set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain.step(my_action)
```

### solve\_with <Badge text="Domain" type="warn"/>

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver', 'annotation': 'Solver'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver**: The solver.
- **domain_factory**: A callable with no argument returning the domain to solve (factory is the domain class if None).
- **load_path**: The path to restore the solver state from (if None, the solving process will be launched instead).

#### Returns
The new solver (auto-cast to the level of the domain).

### step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment.step()` provides some boilerplate code and internally calls `Environment._step()` (which
returns a transition outcome). The boilerplate code automatically stores next state into the `_memory` attribute
and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment.step()` to call the external environment and not
use the `Environment._step()` helper function.
:::

::: warning
Before calling `Environment.step()` the first time or when the end of an episode is
reached, `Initializable.reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

### \_check\_value <Badge text="Rewards" type="warn"/>

<skdecide-signature name= "_check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### \_get\_action\_space <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_action_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the (cached) domain action space (finite or infinite set).

By default, `Events._get_action_space()` internally calls `Events._get_action_space_()` the first time and
automatically caches its value to make future calls more efficient (since the action space is assumed to be
constant).

#### Returns
The action space.

### \_get\_action\_space\_ <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_action_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the domain action space (finite or infinite set).

This is a helper function called by default from `Events._get_action_space()`, the difference being that the
result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The action space.

### \_get\_applicable\_actions <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_applicable_actions" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
the internal one if omitted.

By default, `Events._get_applicable_actions()` provides some boilerplate code and internally
calls `Events._get_applicable_actions_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of applicable actions.

### \_get\_applicable\_actions\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_applicable_actions_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history).

This is a helper function called by default from `Events._get_applicable_actions()`, the difference being that
the memory parameter is mandatory here.

#### Parameters
- **memory**: The memory to consider.

#### Returns
The space of applicable actions.

### \_get\_enabled\_events <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_enabled_events" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history), or in the internal one if omitted.

By default, `Events._get_enabled_events()` provides some boilerplate code and internally
calls `Events._get_enabled_events_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of enabled events.

### \_get\_enabled\_events\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_enabled_events_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history).

This is a helper function called by default from `Events._get_enabled_events()`, the difference being that the
memory parameter is mandatory here.

#### Parameters
- **memory**: The memory to consider.

#### Returns
The space of enabled events.

### \_get\_goals <Badge text="Goals" type="warn"/>

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

### \_get\_goals\_ <Badge text="Goals" type="warn"/>

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

### \_get\_initial\_state <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the (cached) initial state.

By default, `DeterministicInitialized._get_initial_state()` internally
calls `DeterministicInitialized._get_initial_state_()` the first time and automatically caches its value to make
future calls more efficient (since the initial state is assumed to be constant).

#### Returns
The initial state.

### \_get\_initial\_state\_ <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the initial state.

This is a helper function called by default from `DeterministicInitialized._get_initial_state()`, the difference
being that the result is not cached here.

#### Returns
The initial state.

### \_get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized._get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### \_get\_initial\_state\_distribution\_ <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_distribution_" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the probability distribution of initial states.

This is a helper function called by default from `UncertainInitialized._get_initial_state_distribution()`, the
difference being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The probability distribution of initial states.

### \_get\_memory\_maxlen <Badge text="History" type="warn"/>

<skdecide-signature name= "_get_memory_maxlen" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Get the (cached) memory max length.

By default, `FiniteHistory._get_memory_maxlen()` internally calls `FiniteHistory._get_memory_maxlen_()` the first
time and automatically caches its value to make future calls more efficient (since the memory max length is
assumed to be constant).

#### Returns
The memory max length.

### \_get\_memory\_maxlen\_ <Badge text="FiniteHistory" type="warn"/>

<skdecide-signature name= "_get_memory_maxlen_" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Get the memory max length.

This is a helper function called by default from `FiniteHistory._get_memory_maxlen()`, the difference being that
the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The memory max length.

### \_get\_next\_state\_distribution <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the probability distribution of next state given a memory and action.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The probability distribution of next state.

### \_get\_observation <Badge text="TransformedObservable" type="warn"/>

<skdecide-signature name= "_get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable._get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### \_get\_observation\_space\_ <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the observation space (finite or infinite set).

This is a helper function called by default from `PartiallyObservable._get_observation_space()`, the difference
being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The observation space.

### \_get\_transition\_value <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_get_transition_value" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_agent[TransitionValue[D.T_value]]'}"></skdecide-signature>

Get the value (reward or cost) of a transition.

The transition to consider is defined by the function parameters.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `UncertainTransitions._is_transition_value_dependent_on_next_state_()` to return
False. This information can then be exploited by solvers to avoid computing next state to evaluate a
transition value (more efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
The transition value (reward or cost).

### \_init\_memory <Badge text="History" type="warn"/>

<skdecide-signature name= "_init_memory" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_memory[D.T_state]'}"></skdecide-signature>

Initialize memory (possibly with a state) according to its specification and return it.

This function is automatically called by `Initializable._reset()` to reinitialize the internal memory whenever
the domain is used as an environment.

#### Parameters
- **state**: An optional state to initialize the memory with (typically the initial state).

#### Returns
The new initialized memory.

### \_is\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_action" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is an action (i.e. a controllable event for the agents).

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
action space provided by `Events._get_action_space()`, but it can be overridden for faster implementations.
:::

#### Parameters
- **event**: The event to consider.

#### Returns
True if the event is an action (False otherwise).

### \_is\_applicable\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_applicable_action" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
omitted.

By default, `Events._is_applicable_action()` provides some boilerplate code and internally
calls `Events._is_applicable_action_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the action is applicable (False otherwise).

### \_is\_applicable\_action\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_applicable_action_from" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history).

This is a helper function called by default from `Events._is_applicable_action()`, the difference being that the
memory parameter is mandatory here.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the space of
applicable actions provided by `Events._get_applicable_actions_from()`, but it can be overridden for faster
implementations.
:::

#### Parameters
- **memory**: The memory to consider.

#### Returns
True if the action is applicable (False otherwise).

### \_is\_enabled\_event <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_enabled_event" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an uncontrollable event is enabled in the given memory (state or history), or in the
internal one if omitted.

By default, `Events._is_enabled_event()` provides some boilerplate code and internally
calls `Events._is_enabled_event_from()`. The boilerplate code automatically passes the `_memory` attribute instead
of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the event is enabled (False otherwise).

### \_is\_enabled\_event\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_enabled_event_from" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is enabled in the given memory (state or history).

This is a helper function called by default from `Events._is_enabled_event()`, the difference being that the
memory parameter is mandatory here.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the space of
enabled events provided by `Events._get_enabled_events_from()`, but it can be overridden for faster
implementations.
:::

#### Parameters
- **memory**: The memory to consider.

#### Returns
True if the event is enabled (False otherwise).

### \_is\_goal <Badge text="Goals" type="warn"/>

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

### \_is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable._get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### \_is\_terminal <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_terminal" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether a state is terminal.

A terminal state is a state with no outgoing transition (except to itself with value 0).

#### Parameters
- **state**: The state to consider.

#### Returns
True if the state is terminal (False otherwise).

### \_is\_transition\_value\_dependent\_on\_next\_state <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_transition_value_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether _get_transition_value() requires the next_state parameter for its computation (cached).

By default, `UncertainTransitions._is_transition_value_dependent_on_next_state()` internally
calls `UncertainTransitions._is_transition_value_dependent_on_next_state_()` the first time and automatically
caches its value to make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### \_is\_transition\_value\_dependent\_on\_next\_state\_ <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_transition_value_dependent_on_next_state_" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether _get_transition_value() requires the next_state parameter for its computation.

This is a helper function called by default
from `UncertainTransitions._is_transition_value_dependent_on_next_state()`, the difference being that the result
is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### \_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable._reset()` provides some boilerplate code and internally
calls `Initializable._state_reset()` (which returns an initial state). The boilerplate code automatically stores
the initial state into the `_memory` attribute and samples a corresponding observation.

#### Returns
An initial observation.

### \_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation._sample()` provides some boilerplate code and internally
calls `Simulation._state_sample()` (which returns a transition outcome). The boilerplate code automatically
samples an observation corresponding to the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation._sample()` to call the external simulator and not use
the `Simulation._state_sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### \_set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with
successive `Environment._step()` calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain._set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain._step(my_action)
```

### \_state\_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_state_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Reset the state of the environment and return an initial state.

This is a helper function called by default from `Initializable._reset()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Returns
An initial state.

### \_state\_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_state_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one sample of the transition's dynamics.

This is a helper function called by default from `Simulation._sample()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The transition outcome of the sampled transition.

### \_state\_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_state_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one step of the transition's dynamics.

This is a helper function called by default from `Environment._step()`. It focuses on the state level, as opposed
to the observation one for the latter.

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The transition outcome of this step.

### \_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment._step()` provides some boilerplate code and internally
calls `Environment._state_step()` (which returns a transition outcome). The boilerplate code automatically stores
next state into the `_memory` attribute and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment._step()` to call the external environment and not
use the `Environment._state_step()` helper function.
:::

::: warning
Before calling `Environment._step()` the first time or when the end of an episode is
reached, `Initializable._reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

## UncertainSchedulingDomain

This is the highest level scheduling domain class (inheriting top-level class for each mandatory
domain characteristic).

### add\_to\_current\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "add_to_current_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'state'}]}"></skdecide-signature>

Samples completion conditions for a given task and add these conditions to the list of conditions in the
given state. This function should be called when a task complete.

### all\_tasks\_possible <Badge text="MixedRenewable" type="warn"/>

<skdecide-signature name= "all_tasks_possible" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}], 'return': 'bool'}"></skdecide-signature>

Return a True is for each task there is at least one mode in which the task can be executed, given the
resource configuration in the state provided as argument. Returns False otherwise.
If this function returns False, the scheduling problem is unsolvable from this state.
This is to cope with the use of non-renable resources that may lead to state from which a
task will not be possible anymore.

### check\_if\_action\_can\_be\_started <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "check_if_action_can_be_started" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}], 'return': 'Tuple[bool, Dict[str, int]]'}"></skdecide-signature>

Check if a start or resume action can be applied. It returns a boolean and a dictionary of resources to use.
        

### check\_unique\_resource\_names <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "check_unique_resource_names" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return True if there are no duplicates in resource names across both resource types
and resource units name lists.

### check\_value <Badge text="Rewards" type="warn"/>

<skdecide-signature name= "check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### find\_one\_ressource\_to\_do\_one\_task <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "find_one_ressource_to_do_one_task" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'int'}], 'return': 'List[str]'}"></skdecide-signature>

For the common case when it is possible to do the task by one resource unit.
For general case, it might just return no possible ressource unit.

### get\_action\_space <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_action_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the (cached) domain action space (finite or infinite set).

By default, `Events.get_action_space()` internally calls `Events._get_action_space_()` the first time and
automatically caches its value to make future calls more efficient (since the action space is assumed to be
constant).

#### Returns
The action space.

### get\_all\_condition\_items <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_all_condition_items" :sig="{'params': [{'name': 'self'}], 'return': 'Enum'}"></skdecide-signature>

Return an Enum with all the elements that can be used to define a condition.

Example:
    return
        ConditionElementsExample(Enum):
            OK = 0
            NC_PART_1_OPERATION_1 = 1
            NC_PART_1_OPERATION_2 = 2
            NC_PART_2_OPERATION_1 = 3
            NC_PART_2_OPERATION_2 = 4
            HARDWARE_ISSUE_MACHINE_A = 5
            HARDWARE_ISSUE_MACHINE_B = 6
    

### get\_all\_resources\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_all_resources_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, Dict[str, Any]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a resource type or resource unit
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {unit: {skill: (detail of skill)}} 

### get\_all\_tasks\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_all_tasks_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, Dict[str, Any]]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a task
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {task: {skill: (detail of skill)}} 

### get\_all\_unconditional\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_all_unconditional_tasks" :sig="{'params': [{'name': 'self'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids for which there are no conditions. These tasks are to be considered at
the start of a project (i.e. in the initial state). 

### get\_applicable\_actions <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_applicable_actions" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
the internal one if omitted.

By default, `Events.get_applicable_actions()` provides some boilerplate code and internally
calls `Events._get_applicable_actions()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of applicable actions.

### get\_available\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_available_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids that can be considered under the conditions defined in the given state.
Note that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks
that are remaining, or that have been completed, paused or started / resumed.

### get\_enabled\_events <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_enabled_events" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history), or in the internal one if omitted.

By default, `Events.get_enabled_events()` provides some boilerplate code and internally
calls `Events._get_enabled_events()`. The boilerplate code automatically passes the `_memory` attribute instead of
the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of enabled events.

### get\_goals <Badge text="Goals" type="warn"/>

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

### get\_initial\_state <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the (cached) initial state.

By default, `DeterministicInitialized.get_initial_state()` internally
calls `DeterministicInitialized._get_initial_state_()` the first time and automatically caches its value to make
future calls more efficient (since the initial state is assumed to be constant).

#### Returns
The initial state.

### get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized.get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### get\_max\_horizon <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "get_max_horizon" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Return the maximum time horizon (int)

### get\_mode\_costs <Badge text="WithModeCosts" type="warn"/>

<skdecide-signature name= "get_mode_costs" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, float]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode
and the value indicates the cost of execution the task in the mode.

### get\_next\_state\_distribution <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the probability distribution of next state given a memory and action.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The probability distribution of next state.

### get\_objectives <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "get_objectives" :sig="{'params': [{'name': 'self'}], 'return': 'List[SchedulingObjectiveEnum]'}"></skdecide-signature>

Return the objectives to consider as a list. The items should be of SchedulingObjectiveEnum type.

### get\_observation <Badge text="TransformedObservable" type="warn"/>

<skdecide-signature name= "get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable.get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### get\_preallocations <Badge text="WithPreallocations" type="warn"/>

<skdecide-signature name= "get_preallocations" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[str]]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value indicates the pre-allocated resources for this task (as a list of str)

### get\_predecessors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "get_predecessors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the predecessors of the task. Successors are given as a list for a task given as a key.

### get\_resource\_cost\_per\_time\_unit <Badge text="WithResourceCosts" type="warn"/>

<skdecide-signature name= "get_resource_cost_per_time_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, float]'}"></skdecide-signature>

Return a dictionary where the key is the name of a resource (str)
and the value indicates the cost of using this resource per time unit.

### get\_resource\_renewability <Badge text="MixedRenewable" type="warn"/>

<skdecide-signature name= "get_resource_renewability" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, bool]'}"></skdecide-signature>

Return a dictionary where the key is a resource name (string)
and the value whether this resource is renewable (True) or not (False).

### get\_resource\_type\_for\_unit <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "get_resource_type_for_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, str]'}"></skdecide-signature>

Return a dictionary where the key is a resource unit name and the value a resource type name.
An empty dictionary can be used if there are no resource unit matching a resource type.

### get\_resource\_types\_names <Badge text="WithResourceTypes" type="warn"/>

<skdecide-signature name= "get_resource_types_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource types as a list.

### get\_resource\_units\_names <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "get_resource_units_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource units as a list.

### get\_skills\_names <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_skills_names" :sig="{'params': [{'name': 'self'}], 'return': 'Set[str]'}"></skdecide-signature>

Return a list of all skill names as a list of str. Skill names are defined in the 2 dictionaries returned
by the get_all_resources_skills and get_all_tasks_skills functions.

### get\_skills\_of\_resource <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_skills_of_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}], 'return': 'Dict[str, Any]'}"></skdecide-signature>

Return the skills of a given resource

### get\_skills\_of\_task <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_skills_of_task" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'int'}], 'return': 'Dict[str, Any]'}"></skdecide-signature>

Return the skill requirements for a given task

### get\_successors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "get_successors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the successors of the tasks. Successors are given as a list for a task given as a key.

### get\_task\_existence\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_task_existence_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)
for the task to be part of the schedule. If a task has no entry in the dictionary,
there is no conditions for that task.

Example:
    return
         {
            20: [get_all_condition_items().NC_PART_1_OPERATION_1],
            21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]
            22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]
         }e

 

### get\_task\_on\_completion\_added\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_task_on_completion_added_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[Distribution]]'}"></skdecide-signature>

Return a dict of list. The key of the dict is the task id and each list is composed of a list of tuples.
Each tuple contains the probability (first item in tuple) that the conditionElement (second item in tuple)
is True. The probabilities in the inner list should sum up to 1. The dictionary should only contains the keys
of tasks that can create conditions.

Example:
     return
        {
            12:
                [
                DiscreteDistribution([(ConditionElementsExample.NC_PART_1_OPERATION_1, 0.1), (ConditionElementsExample.OK, 0.9)]),
                DiscreteDistribution([(ConditionElementsExample.HARDWARE_ISSUE_MACHINE_A, 0.05), ('paper', 0.1), (ConditionElementsExample.OK, 0.95)])
                ]
        }
    

### get\_task\_paused\_non\_renewable\_resource\_returned <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "get_task_paused_non_renewable_resource_returned" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type bool indicating
if the non-renewable resources are consumed when the task is paused (False) or made available again (True).
E.g. {
        2: False  # if paused, non-renewable resource will be consumed
        5: True  # if paused, the non-renewable resource will be available again
        }

### get\_task\_preemptivity <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "get_task_preemptivity" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a boolean indicating
if the task can be paused or stopped.
E.g. {
        1: False
        2: True
        3: False
        4: False
        5: True
        6: False
        }

### get\_task\_progress <Badge text="CustomTaskProgress" type="warn"/>

<skdecide-signature name= "get_task_progress" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 't_from', 'annotation': 'int'}, {'name': 't_to', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'Optional[int]'}, {'name': 'sampled_duration', 'default': 'None', 'annotation': 'Optional[int]'}], 'return': 'float'}"></skdecide-signature>

#### Returns
 The task progress (float) between t_from and t_to.
 

### get\_task\_resuming\_type <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "get_task_resuming_type" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, ResumeType]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type ResumeType indicating
if the task can be resumed (restarted from where it was paused with no time loss)
or restarted (restarted from the start).
E.g. {
        1: ResumeType.NA
        2: ResumeType.Resume
        3: ResumeType.NA
        4: ResumeType.NA
        5: ResumeType.Restart
        6: ResumeType.NA
        }

### get\_time\_lags <Badge text="WithTimeLag" type="warn"/>

<skdecide-signature name= "get_time_lags" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, TimeLag]]'}"></skdecide-signature>

Return nested dictionaries where the first key is the id of a task (int)
and the second key is the id of another task (int).
The value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end
of the first task to the start of the second task.

e.g.
    {
        12:{
            15: TimeLag(5, 10),
            16: TimeLag(5, 20),
            17: MinimumOnlyTimeLag(5),
            18: MaximumOnlyTimeLag(15),
        }
    }

#### Returns
A dictionary of TimeLag objects.

### get\_time\_window <Badge text="WithTimeWindow" type="warn"/>

<skdecide-signature name= "get_time_window" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, TimeWindow]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value is a TimeWindow object.
Note that the max time horizon needs to be provided to the TimeWindow constructors
e.g.
    {
        1: TimeWindow(10, 15, 20, 30, self.get_max_horizon())
        2: EmptyTimeWindow(self.get_max_horizon())
        3: EndTimeWindow(20, 25, self.get_max_horizon())
        4: EndBeforeOnlyTimeWindow(40, self.get_max_horizon())
    }

#### Returns
A dictionary of TimeWindow objects.

### get\_transition\_value <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "get_transition_value" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_agent[TransitionValue[D.T_value]]'}"></skdecide-signature>

Get the value (reward or cost) of a transition.

The transition to consider is defined by the function parameters.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `UncertainTransitions._is_transition_value_dependent_on_next_state_()` to return
False. This information can then be exploited by solvers to avoid computing next state to evaluate a
transition value (more efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
The transition value (reward or cost).

### get\_variable\_resource\_consumption <Badge text="VariableResourceConsumption" type="warn"/>

<skdecide-signature name= "get_variable_resource_consumption" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return true if the domain has variable resource consumption,
false if the consumption of resource does not vary in time for any of the tasks

### initialize\_domain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "initialize_domain" :sig="{'params': [{'name': 'self'}]}"></skdecide-signature>

Initialize a scheduling domain. This function needs to be called when instantiating a scheduling domain.

### is\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_action" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is an action (i.e. a controllable event for the agents).

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
action space provided by `Events.get_action_space()`, but it can be overridden for faster implementations.
:::

#### Parameters
- **event**: The event to consider.

#### Returns
True if the event is an action (False otherwise).

### is\_applicable\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_applicable_action" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
omitted.

By default, `Events.is_applicable_action()` provides some boilerplate code and internally
calls `Events._is_applicable_action()`. The boilerplate code automatically passes the `_memory` attribute instead
of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the action is applicable (False otherwise).

### is\_enabled\_event <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_enabled_event" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an uncontrollable event is enabled in the given memory (state or history), or in the
internal one if omitted.

By default, `Events.is_enabled_event()` provides some boilerplate code and internally
calls `Events._is_enabled_event()`. The boilerplate code automatically passes the `_memory` attribute instead of
the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the event is enabled (False otherwise).

### is\_goal <Badge text="Goals" type="warn"/>

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

### is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable.get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### is\_terminal <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "is_terminal" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether a state is terminal.

A terminal state is a state with no outgoing transition (except to itself with value 0).

#### Parameters
- **state**: The state to consider.

#### Returns
True if the state is terminal (False otherwise).

### is\_transition\_value\_dependent\_on\_next\_state <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "is_transition_value_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether get_transition_value() requires the next_state parameter for its computation (cached).

By default, `UncertainTransitions.is_transition_value_dependent_on_next_state()` internally
calls `UncertainTransitions._is_transition_value_dependent_on_next_state_()` the first time and automatically
caches its value to make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable.reset()` provides some boilerplate code and internally calls `Initializable._reset()`
(which returns an initial state). The boilerplate code automatically stores the initial state into the `_memory`
attribute and samples a corresponding observation.

#### Returns
An initial observation.

### sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation.sample()` provides some boilerplate code and internally calls `Simulation._sample()`
(which returns a transition outcome). The boilerplate code automatically samples an observation corresponding to
the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation.sample()` to call the external simulator and not use
the `Simulation._sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### sample\_completion\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "sample_completion_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}], 'return': 'List[int]'}"></skdecide-signature>

Samples the condition distributions associated with the given task and return a list of sampled
conditions.

### sample\_quantity\_resource <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "sample_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Sample an amount of resource availability (int) for the given resource
(either resource type or resource unit) at the given time. This number should be the sum of the number of
resource available at time t and the number of resource of this type consumed so far).

### sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Sample, store and return task duration for the given task in the given mode.

### set\_inplace\_environment <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "set_inplace_environment" :sig="{'params': [{'name': 'self'}, {'name': 'inplace_environment', 'annotation': 'bool'}]}"></skdecide-signature>

Activate or not the fact that the simulator modifies the given state inplace or create a copy before.
The inplace version is several times faster but will lead to bugs in graph search solver.

### set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with successive `Environment.step()`
calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain.set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain.step(my_action)
```

### solve\_with <Badge text="Domain" type="warn"/>

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver', 'annotation': 'Solver'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver**: The solver.
- **domain_factory**: A callable with no argument returning the domain to solve (factory is the domain class if None).
- **load_path**: The path to restore the solver state from (if None, the solving process will be launched instead).

#### Returns
The new solver (auto-cast to the level of the domain).

### step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment.step()` provides some boilerplate code and internally calls `Environment._step()` (which
returns a transition outcome). The boilerplate code automatically stores next state into the `_memory` attribute
and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment.step()` to call the external environment and not
use the `Environment._step()` helper function.
:::

::: warning
Before calling `Environment.step()` the first time or when the end of an episode is
reached, `Initializable.reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

### update\_complete\_dummy\_tasks <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_complete_dummy_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the status of newly started tasks whose duration is 0 from ongoing to complete.

### update\_complete\_dummy\_tasks\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_complete_dummy_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update the status of newly started tasks whose duration is 0
from ongoing to complete.

### update\_complete\_dummy\_tasks\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_complete_dummy_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of newly started tasks whose duration is 0
from ongoing to complete.

### update\_complete\_tasks <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_complete_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}]}"></skdecide-signature>

Update the status of newly completed tasks in the state from ongoing to complete
and update resource availability. This function will also log in task_details the time it was complete

### update\_complete\_tasks\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_complete_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}]}"></skdecide-signature>

In a simulated scheduling environment, update the status of newly completed tasks in the state from ongoing to complete
and update resource availability. This function will also log in task_details the time it was complete

### update\_complete\_tasks\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_complete_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of newly completed tasks in the state from ongoing
to complete, update resource availability and update on-completion conditions.
This function will also log in task_details the time it was complete.

### update\_conditional\_tasks <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_conditional_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update remaining tasks by checking conditions and potentially adding conditional tasks.

### update\_conditional\_tasks\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_conditional_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update remaining tasks by checking conditions and potentially
adding conditional tasks.

### update\_conditional\_tasks\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_conditional_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update remaining tasks by checking conditions and potentially adding conditional tasks.

### update\_pause\_tasks <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_pause_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the status of a task from ongoing to paused if specified in the action
and update resource availability. This function will also log in task_details the time it was paused.

### update\_pause\_tasks\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_pause_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulation scheduling environment, update the status of a task from ongoing to paused if
specified in the action and update resource availability. This function will also log in task_details
the time it was paused.

### update\_pause\_tasks\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_pause_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of a task from ongoing to paused if
specified in the action and update resource availability. This function will also log in task_details
the time it was paused.

### update\_progress <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_progress" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}]}"></skdecide-signature>

Update the progress of all ongoing tasks in the state.

### update\_progress\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_progress_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}]}"></skdecide-signature>

In a simulation scheduling environment, update the progress of all ongoing tasks in the state.

### update\_progress\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_progress_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the progress of all ongoing tasks in the state.

### update\_resource\_availability <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_resource_availability" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update resource availability for next time step. This should be called after update_time().

### update\_resource\_availability\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_resource_availability_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update resource availability for next time step.
This should be called after update_time().

### update\_resource\_availability\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_resource_availability_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update resource availability for next time step. This should be called after update_time().

### update\_resume\_tasks <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_resume_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the status of a task from paused to ongoing if specified in the action
and update resource availability. This function will also log in task_details the time it was resumed

### update\_resume\_tasks\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_resume_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulationn scheduling environment, update the status of a task from paused to ongoing if specified
in the action and update resource availability. This function will also log in task_details the time it was
resumed.

### update\_resume\_tasks\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_resume_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of a task from paused to ongoing if specified
in the action and update resource availability. This function will also log in task_details the time it was
resumed.

### update\_start\_tasks <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_start_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the status of a task from remaining to ongoing if specified in the action
and update resource availability. This function will also log in task_details the time it was started.

### update\_start\_tasks\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_start_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update the status of a task from remaining to ongoing if
specified in the action and update resource availability. This function will also log in task_details the
time it was started.

### update\_start\_tasks\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_start_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of a task from remaining to ongoing
if specified in the action and update resource availability.
This function returns a DsicreteDistribution of State.
This function will also log in task_details the time it was started.

### update\_time <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_time" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the time of the state if the time_progress attribute of the given EnumerableAction is True.

### update\_time\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_time_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update the time of the state if the time_progress attribute of the
given EnumerableAction is True.

### update\_time\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_time_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the time of the state if the time_progress attribute of the given EnumerableAction is True.

### \_add\_to\_current\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_add_to_current_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'state'}]}"></skdecide-signature>

Samples completion conditions for a given task and add these conditions to the list of conditions in the
given state. This function should be called when a task complete.

### \_check\_value <Badge text="Rewards" type="warn"/>

<skdecide-signature name= "_check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### \_get\_action\_space <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_action_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the (cached) domain action space (finite or infinite set).

By default, `Events._get_action_space()` internally calls `Events._get_action_space_()` the first time and
automatically caches its value to make future calls more efficient (since the action space is assumed to be
constant).

#### Returns
The action space.

### \_get\_action\_space\_ <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_action_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

To be implemented if needed one day.

### \_get\_all\_resources\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "_get_all_resources_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, Dict[str, Any]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a resource type or resource unit
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {unit: {skill: (detail of skill)}} 

### \_get\_all\_tasks\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "_get_all_tasks_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, Dict[str, Any]]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a task
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {task: {skill: (detail of skill)}} 

### \_get\_all\_unconditional\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_get_all_unconditional_tasks" :sig="{'params': [{'name': 'self'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids for which there are no conditions. These tasks are to be considered at
the start of a project (i.e. in the initial state). 

### \_get\_applicable\_actions <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_applicable_actions" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
the internal one if omitted.

By default, `Events._get_applicable_actions()` provides some boilerplate code and internally
calls `Events._get_applicable_actions_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of applicable actions.

### \_get\_applicable\_actions\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_applicable_actions_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Returns the action space from a state.
TODO : think about a way to avoid the instaceof usage.

### \_get\_available\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_get_available_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids that can be considered under the conditions defined in the given state.
Note that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks
that are remaining, or that have been completed, paused or started / resumed.

### \_get\_enabled\_events <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_enabled_events" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history), or in the internal one if omitted.

By default, `Events._get_enabled_events()` provides some boilerplate code and internally
calls `Events._get_enabled_events_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of enabled events.

### \_get\_enabled\_events\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_enabled_events_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history).

This is a helper function called by default from `Events._get_enabled_events()`, the difference being that the
memory parameter is mandatory here.

#### Parameters
- **memory**: The memory to consider.

#### Returns
The space of enabled events.

### \_get\_goals <Badge text="Goals" type="warn"/>

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

### \_get\_goals\_ <Badge text="Goals" type="warn"/>

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

### \_get\_initial\_state <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the (cached) initial state.

By default, `DeterministicInitialized._get_initial_state()` internally
calls `DeterministicInitialized._get_initial_state_()` the first time and automatically caches its value to make
future calls more efficient (since the initial state is assumed to be constant).

#### Returns
The initial state.

### \_get\_initial\_state\_ <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Create and return an empty initial state

### \_get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized._get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### \_get\_initial\_state\_distribution\_ <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_distribution_" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the probability distribution of initial states.

This is a helper function called by default from `UncertainInitialized._get_initial_state_distribution()`, the
difference being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The probability distribution of initial states.

### \_get\_max\_horizon <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "_get_max_horizon" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Return the maximum time horizon (int)

### \_get\_memory\_maxlen <Badge text="History" type="warn"/>

<skdecide-signature name= "_get_memory_maxlen" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Get the (cached) memory max length.

By default, `FiniteHistory._get_memory_maxlen()` internally calls `FiniteHistory._get_memory_maxlen_()` the first
time and automatically caches its value to make future calls more efficient (since the memory max length is
assumed to be constant).

#### Returns
The memory max length.

### \_get\_memory\_maxlen\_ <Badge text="FiniteHistory" type="warn"/>

<skdecide-signature name= "_get_memory_maxlen_" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Get the memory max length.

This is a helper function called by default from `FiniteHistory._get_memory_maxlen()`, the difference being that
the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The memory max length.

### \_get\_mode\_costs <Badge text="WithModeCosts" type="warn"/>

<skdecide-signature name= "_get_mode_costs" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, float]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode
and the value indicates the cost of execution the task in the mode.

### \_get\_next\_state <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "_get_next_state" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'D.T_state'}"></skdecide-signature>

This function will be used if the domain is defined with DeterministicTransitions. This function will be ignored
if the domain is defined as having UncertainTransitions or Simulation. 

### \_get\_next\_state\_distribution <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

This function will be used if the domain is defined with UncertainTransitions. This function will be ignored
if the domain is defined as a Simulation. This function may also be used by uncertainty-specialised solvers
 on deterministic domains.

### \_get\_objectives <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "_get_objectives" :sig="{'params': [{'name': 'self'}], 'return': 'List[SchedulingObjectiveEnum]'}"></skdecide-signature>

Return the objectives to consider as a list. The items should be of SchedulingObjectiveEnum type.

### \_get\_observation <Badge text="TransformedObservable" type="warn"/>

<skdecide-signature name= "_get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable._get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### \_get\_observation\_space\_ <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

To be implemented if needed one day.

### \_get\_preallocations <Badge text="WithPreallocations" type="warn"/>

<skdecide-signature name= "_get_preallocations" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[str]]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value indicates the pre-allocated resources for this task (as a list of str)

### \_get\_predecessors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "_get_predecessors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the predecessors of the task. Successors are given as a list for a task given as a key.

### \_get\_resource\_cost\_per\_time\_unit <Badge text="WithResourceCosts" type="warn"/>

<skdecide-signature name= "_get_resource_cost_per_time_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, float]'}"></skdecide-signature>

Return a dictionary where the key is the name of a resource (str)
and the value indicates the cost of using this resource per time unit.

### \_get\_resource\_renewability <Badge text="MixedRenewable" type="warn"/>

<skdecide-signature name= "_get_resource_renewability" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, bool]'}"></skdecide-signature>

Return a dictionary where the key is a resource name (string)
and the value whether this resource is renewable (True) or not (False).

### \_get\_resource\_type\_for\_unit <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "_get_resource_type_for_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, str]'}"></skdecide-signature>

Return a dictionary where the key is a resource unit name and the value a resource type name.
An empty dictionary can be used if there are no resource unit matching a resource type.

### \_get\_resource\_types\_names <Badge text="WithResourceTypes" type="warn"/>

<skdecide-signature name= "_get_resource_types_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource types as a list.

### \_get\_resource\_units\_names <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "_get_resource_units_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource units as a list.

### \_get\_successors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "_get_successors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the successors of the tasks. Successors are given as a list for a task given as a key.

### \_get\_task\_existence\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_get_task_existence_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)
for the task to be part of the schedule. If a task has no entry in the dictionary,
there is no conditions for that task.

Example:
    return
         {
            20: [get_all_condition_items().NC_PART_1_OPERATION_1],
            21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]
            22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]
         }e

### \_get\_task\_paused\_non\_renewable\_resource\_returned <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "_get_task_paused_non_renewable_resource_returned" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type bool indicating
if the non-renewable resources are consumed when the task is paused (False) or made available again (True).
E.g. {
        2: False  # if paused, non-renewable resource will be consumed
        5: True  # if paused, the non-renewable resource will be available again
        }

### \_get\_task\_preemptivity <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "_get_task_preemptivity" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a boolean indicating
if the task can be paused or stopped.
E.g. {
        1: False
        2: True
        3: False
        4: False
        5: True
        6: False
        }

### \_get\_task\_progress <Badge text="CustomTaskProgress" type="warn"/>

<skdecide-signature name= "_get_task_progress" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 't_from', 'annotation': 'int'}, {'name': 't_to', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'Optional[int]'}, {'name': 'sampled_duration', 'default': 'None', 'annotation': 'Optional[int]'}], 'return': 'float'}"></skdecide-signature>

#### Returns
 The task progress (float) between t_from and t_to.
 

### \_get\_task\_resuming\_type <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "_get_task_resuming_type" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, ResumeType]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type ResumeType indicating
if the task can be resumed (restarted from where it was paused with no time loss)
or restarted (restarted from the start).
E.g. {
        1: ResumeType.NA
        2: ResumeType.Resume
        3: ResumeType.NA
        4: ResumeType.NA
        5: ResumeType.Restart
        6: ResumeType.NA
        }

### \_get\_tasks\_ids <Badge text="MultiMode" type="warn"/>

<skdecide-signature name= "_get_tasks_ids" :sig="{'params': [{'name': 'self'}], 'return': 'Union[Set[int], Dict[int, Any], List[int]]'}"></skdecide-signature>

Return a set or dict of int = id of tasks

### \_get\_tasks\_modes <Badge text="MultiMode" type="warn"/>

<skdecide-signature name= "_get_tasks_modes" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, ModeConsumption]]'}"></skdecide-signature>

Return a nested dictionary where the first key is a task id and the second key is a mode id.
 The value is a Mode object defining the resource consumption.
If the domain is an instance of VariableResourceConsumption, VaryingModeConsumption objects should be used.
If this is not the case (i.e. the domain is an instance of ConstantResourceConsumption),
then ConstantModeConsumption should be used.

E.g. with constant resource consumption
    {
        12: {
                1: ConstantModeConsumption({'rt_1': 2, 'rt_2': 0, 'ru_1': 1}),
                2: ConstantModeConsumption({'rt_1': 0, 'rt_2': 3, 'ru_1': 1}),
            }
    }

E.g. with time varying resource consumption
    {
    12: {
        1: VaryingModeConsumption({'rt_1': [2,2,2,2,3], 'rt_2': [0,0,0,0,0], 'ru_1': [1,1,1,1,1]}),
        2: VaryingModeConsumption({'rt_1': [1,1,1,1,2,2,2], 'rt_2': [0,0,0,0,0,0,0], 'ru_1': [1,1,1,1,1,1,1]}),
        }
    }

### \_get\_time\_lags <Badge text="WithTimeLag" type="warn"/>

<skdecide-signature name= "_get_time_lags" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, TimeLag]]'}"></skdecide-signature>

Return nested dictionaries where the first key is the id of a task (int)
and the second key is the id of another task (int).
The value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end
of the first task to the start of the second task.

e.g.
    {
        12:{
            15: TimeLag(5, 10),
            16: TimeLag(5, 20),
            17: MinimumOnlyTimeLag(5),
            18: MaximumOnlyTimeLag(15),
        }
    }

#### Returns
A dictionary of TimeLag objects.

### \_get\_time\_window <Badge text="WithTimeWindow" type="warn"/>

<skdecide-signature name= "_get_time_window" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, TimeWindow]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value is a TimeWindow object.
Note that the max time horizon needs to be provided to the TimeWindow constructors
e.g.
    {
        1: TimeWindow(10, 15, 20, 30, self.get_max_horizon())
        2: EmptyTimeWindow(self.get_max_horizon())
        3: EndTimeWindow(20, 25, self.get_max_horizon())
        4: EndBeforeOnlyTimeWindow(40, self.get_max_horizon())
    }

#### Returns
A dictionary of TimeWindow objects.

### \_get\_variable\_resource\_consumption <Badge text="VariableResourceConsumption" type="warn"/>

<skdecide-signature name= "_get_variable_resource_consumption" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return true if the domain has variable resource consumption,
false if the consumption of resource does not vary in time for any of the tasks

### \_init\_memory <Badge text="History" type="warn"/>

<skdecide-signature name= "_init_memory" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_memory[D.T_state]'}"></skdecide-signature>

Initialize memory (possibly with a state) according to its specification and return it.

This function is automatically called by `Initializable._reset()` to reinitialize the internal memory whenever
the domain is used as an environment.

#### Parameters
- **state**: An optional state to initialize the memory with (typically the initial state).

#### Returns
The new initialized memory.

### \_is\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_action" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is an action (i.e. a controllable event for the agents).

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
action space provided by `Events._get_action_space()`, but it can be overridden for faster implementations.
:::

#### Parameters
- **event**: The event to consider.

#### Returns
True if the event is an action (False otherwise).

### \_is\_applicable\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_applicable_action" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
omitted.

By default, `Events._is_applicable_action()` provides some boilerplate code and internally
calls `Events._is_applicable_action_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the action is applicable (False otherwise).

### \_is\_applicable\_action\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_applicable_action_from" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history).

This is a helper function called by default from `Events._is_applicable_action()`, the difference being that the
memory parameter is mandatory here.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the space of
applicable actions provided by `Events._get_applicable_actions_from()`, but it can be overridden for faster
implementations.
:::

#### Parameters
- **memory**: The memory to consider.

#### Returns
True if the action is applicable (False otherwise).

### \_is\_enabled\_event <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_enabled_event" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an uncontrollable event is enabled in the given memory (state or history), or in the
internal one if omitted.

By default, `Events._is_enabled_event()` provides some boilerplate code and internally
calls `Events._is_enabled_event_from()`. The boilerplate code automatically passes the `_memory` attribute instead
of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the event is enabled (False otherwise).

### \_is\_enabled\_event\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_enabled_event_from" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is enabled in the given memory (state or history).

This is a helper function called by default from `Events._is_enabled_event()`, the difference being that the
memory parameter is mandatory here.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the space of
enabled events provided by `Events._get_enabled_events_from()`, but it can be overridden for faster
implementations.
:::

#### Parameters
- **memory**: The memory to consider.

#### Returns
True if the event is enabled (False otherwise).

### \_is\_goal <Badge text="Goals" type="warn"/>

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

### \_is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable._get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### \_is\_transition\_value\_dependent\_on\_next\_state <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_transition_value_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether _get_transition_value() requires the next_state parameter for its computation (cached).

By default, `UncertainTransitions._is_transition_value_dependent_on_next_state()` internally
calls `UncertainTransitions._is_transition_value_dependent_on_next_state_()` the first time and automatically
caches its value to make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### \_is\_transition\_value\_dependent\_on\_next\_state\_ <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_transition_value_dependent_on_next_state_" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether _get_transition_value() requires the next_state parameter for its computation.

This is a helper function called by default
from `UncertainTransitions._is_transition_value_dependent_on_next_state()`, the difference being that the result
is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### \_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable._reset()` provides some boilerplate code and internally
calls `Initializable._state_reset()` (which returns an initial state). The boilerplate code automatically stores
the initial state into the `_memory` attribute and samples a corresponding observation.

#### Returns
An initial observation.

### \_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation._sample()` provides some boilerplate code and internally
calls `Simulation._state_sample()` (which returns a transition outcome). The boilerplate code automatically
samples an observation corresponding to the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation._sample()` to call the external simulator and not use
the `Simulation._state_sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### \_sample\_completion\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_sample_completion_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}], 'return': 'List[int]'}"></skdecide-signature>

Samples the condition distributions associated with the given task and return a list of sampled
conditions.

### \_sample\_quantity\_resource <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "_sample_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Sample an amount of resource availability (int) for the given resource
(either resource type or resource unit) at the given time. This number should be the sum of the number of
resource available at time t and the number of resource of this type consumed so far).

### \_sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "_sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode.

### \_set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with
successive `Environment._step()` calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain._set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain._step(my_action)
```

### \_state\_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_state_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Reset the state of the environment and return an initial state.

This is a helper function called by default from `Initializable._reset()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Returns
An initial state.

### \_state\_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_state_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

This function will be used if the domain is defined as a Simulation (i.e. transitions are defined by call to
a simulation). This function may also be used by simulation-based solvers on non-Simulation domains.

### \_state\_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_state_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one step of the transition's dynamics.

This is a helper function called by default from `Environment._step()`. It focuses on the state level, as opposed
to the observation one for the latter.

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The transition outcome of this step.

### \_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment._step()` provides some boilerplate code and internally
calls `Environment._state_step()` (which returns a transition outcome). The boilerplate code automatically stores
next state into the `_memory` attribute and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment._step()` to call the external environment and not
use the `Environment._state_step()` helper function.
:::

::: warning
Before calling `Environment._step()` the first time or when the end of an episode is
reached, `Initializable._reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

## DeterministicSchedulingDomain

This is the highest level scheduling domain class (inheriting top-level class for each mandatory
domain characteristic).

### add\_to\_current\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "add_to_current_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'state'}]}"></skdecide-signature>

Samples completion conditions for a given task and add these conditions to the list of conditions in the
given state. This function should be called when a task complete.

### all\_tasks\_possible <Badge text="MixedRenewable" type="warn"/>

<skdecide-signature name= "all_tasks_possible" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}], 'return': 'bool'}"></skdecide-signature>

Return a True is for each task there is at least one mode in which the task can be executed, given the
resource configuration in the state provided as argument. Returns False otherwise.
If this function returns False, the scheduling problem is unsolvable from this state.
This is to cope with the use of non-renable resources that may lead to state from which a
task will not be possible anymore.

### check\_if\_action\_can\_be\_started <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "check_if_action_can_be_started" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}], 'return': 'Tuple[bool, Dict[str, int]]'}"></skdecide-signature>

Check if a start or resume action can be applied. It returns a boolean and a dictionary of resources to use.
        

### check\_unique\_resource\_names <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "check_unique_resource_names" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return True if there are no duplicates in resource names across both resource types
and resource units name lists.

### check\_value <Badge text="Rewards" type="warn"/>

<skdecide-signature name= "check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### find\_one\_ressource\_to\_do\_one\_task <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "find_one_ressource_to_do_one_task" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'int'}], 'return': 'List[str]'}"></skdecide-signature>

For the common case when it is possible to do the task by one resource unit.
For general case, it might just return no possible ressource unit.

### get\_action\_space <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_action_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the (cached) domain action space (finite or infinite set).

By default, `Events.get_action_space()` internally calls `Events._get_action_space_()` the first time and
automatically caches its value to make future calls more efficient (since the action space is assumed to be
constant).

#### Returns
The action space.

### get\_all\_condition\_items <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_all_condition_items" :sig="{'params': [{'name': 'self'}], 'return': 'Enum'}"></skdecide-signature>

Return an Enum with all the elements that can be used to define a condition.

Example:
    return
        ConditionElementsExample(Enum):
            OK = 0
            NC_PART_1_OPERATION_1 = 1
            NC_PART_1_OPERATION_2 = 2
            NC_PART_2_OPERATION_1 = 3
            NC_PART_2_OPERATION_2 = 4
            HARDWARE_ISSUE_MACHINE_A = 5
            HARDWARE_ISSUE_MACHINE_B = 6
    

### get\_all\_resources\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_all_resources_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, Dict[str, Any]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a resource type or resource unit
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {unit: {skill: (detail of skill)}} 

### get\_all\_tasks\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_all_tasks_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, Dict[str, Any]]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a task
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {task: {skill: (detail of skill)}} 

### get\_all\_unconditional\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_all_unconditional_tasks" :sig="{'params': [{'name': 'self'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids for which there are no conditions. These tasks are to be considered at
the start of a project (i.e. in the initial state). 

### get\_applicable\_actions <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_applicable_actions" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
the internal one if omitted.

By default, `Events.get_applicable_actions()` provides some boilerplate code and internally
calls `Events._get_applicable_actions()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of applicable actions.

### get\_available\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_available_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids that can be considered under the conditions defined in the given state.
Note that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks
that are remaining, or that have been completed, paused or started / resumed.

### get\_enabled\_events <Badge text="Events" type="warn"/>

<skdecide-signature name= "get_enabled_events" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history), or in the internal one if omitted.

By default, `Events.get_enabled_events()` provides some boilerplate code and internally
calls `Events._get_enabled_events()`. The boilerplate code automatically passes the `_memory` attribute instead of
the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of enabled events.

### get\_goals <Badge text="Goals" type="warn"/>

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

### get\_initial\_state <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the (cached) initial state.

By default, `DeterministicInitialized.get_initial_state()` internally
calls `DeterministicInitialized._get_initial_state_()` the first time and automatically caches its value to make
future calls more efficient (since the initial state is assumed to be constant).

#### Returns
The initial state.

### get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized.get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### get\_max\_horizon <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "get_max_horizon" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Return the maximum time horizon (int)

### get\_mode\_costs <Badge text="WithModeCosts" type="warn"/>

<skdecide-signature name= "get_mode_costs" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, float]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode
and the value indicates the cost of execution the task in the mode.

### get\_next\_state <Badge text="DeterministicTransitions" type="warn"/>

<skdecide-signature name= "get_next_state" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the next state given a memory and action.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The deterministic next state.

### get\_next\_state\_distribution <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'DiscreteDistribution[D.T_state]'}"></skdecide-signature>

Get the discrete probability distribution of next state given a memory and action.

::: tip
In the Markovian case (memory only holds last state $s$), given an action $a$, this function can
be mathematically represented by $P(S'|s, a)$, where $S'$ is the next state random variable.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The discrete probability distribution of next state.

### get\_objectives <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "get_objectives" :sig="{'params': [{'name': 'self'}], 'return': 'List[SchedulingObjectiveEnum]'}"></skdecide-signature>

Return the objectives to consider as a list. The items should be of SchedulingObjectiveEnum type.

### get\_observation <Badge text="TransformedObservable" type="warn"/>

<skdecide-signature name= "get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable.get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### get\_preallocations <Badge text="WithPreallocations" type="warn"/>

<skdecide-signature name= "get_preallocations" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[str]]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value indicates the pre-allocated resources for this task (as a list of str)

### get\_predecessors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "get_predecessors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the predecessors of the task. Successors are given as a list for a task given as a key.

### get\_resource\_cost\_per\_time\_unit <Badge text="WithResourceCosts" type="warn"/>

<skdecide-signature name= "get_resource_cost_per_time_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, float]'}"></skdecide-signature>

Return a dictionary where the key is the name of a resource (str)
and the value indicates the cost of using this resource per time unit.

### get\_resource\_renewability <Badge text="MixedRenewable" type="warn"/>

<skdecide-signature name= "get_resource_renewability" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, bool]'}"></skdecide-signature>

Return a dictionary where the key is a resource name (string)
and the value whether this resource is renewable (True) or not (False).

### get\_resource\_type\_for\_unit <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "get_resource_type_for_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, str]'}"></skdecide-signature>

Return a dictionary where the key is a resource unit name and the value a resource type name.
An empty dictionary can be used if there are no resource unit matching a resource type.

### get\_resource\_types\_names <Badge text="WithResourceTypes" type="warn"/>

<skdecide-signature name= "get_resource_types_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource types as a list.

### get\_resource\_units\_names <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "get_resource_units_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource units as a list.

### get\_skills\_names <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_skills_names" :sig="{'params': [{'name': 'self'}], 'return': 'Set[str]'}"></skdecide-signature>

Return a list of all skill names as a list of str. Skill names are defined in the 2 dictionaries returned
by the get_all_resources_skills and get_all_tasks_skills functions.

### get\_skills\_of\_resource <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_skills_of_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}], 'return': 'Dict[str, Any]'}"></skdecide-signature>

Return the skills of a given resource

### get\_skills\_of\_task <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_skills_of_task" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'int'}], 'return': 'Dict[str, Any]'}"></skdecide-signature>

Return the skill requirements for a given task

### get\_successors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "get_successors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the successors of the tasks. Successors are given as a list for a task given as a key.

### get\_task\_existence\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_task_existence_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)
for the task to be part of the schedule. If a task has no entry in the dictionary,
there is no conditions for that task.

Example:
    return
         {
            20: [get_all_condition_items().NC_PART_1_OPERATION_1],
            21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]
            22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]
         }e

 

### get\_task\_on\_completion\_added\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_task_on_completion_added_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[Distribution]]'}"></skdecide-signature>

Return a dict of list. The key of the dict is the task id and each list is composed of a list of tuples.
Each tuple contains the probability (first item in tuple) that the conditionElement (second item in tuple)
is True. The probabilities in the inner list should sum up to 1. The dictionary should only contains the keys
of tasks that can create conditions.

Example:
     return
        {
            12:
                [
                DiscreteDistribution([(ConditionElementsExample.NC_PART_1_OPERATION_1, 0.1), (ConditionElementsExample.OK, 0.9)]),
                DiscreteDistribution([(ConditionElementsExample.HARDWARE_ISSUE_MACHINE_A, 0.05), ('paper', 0.1), (ConditionElementsExample.OK, 0.95)])
                ]
        }
    

### get\_task\_paused\_non\_renewable\_resource\_returned <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "get_task_paused_non_renewable_resource_returned" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type bool indicating
if the non-renewable resources are consumed when the task is paused (False) or made available again (True).
E.g. {
        2: False  # if paused, non-renewable resource will be consumed
        5: True  # if paused, the non-renewable resource will be available again
        }

### get\_task\_preemptivity <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "get_task_preemptivity" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a boolean indicating
if the task can be paused or stopped.
E.g. {
        1: False
        2: True
        3: False
        4: False
        5: True
        6: False
        }

### get\_task\_progress <Badge text="CustomTaskProgress" type="warn"/>

<skdecide-signature name= "get_task_progress" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 't_from', 'annotation': 'int'}, {'name': 't_to', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'Optional[int]'}, {'name': 'sampled_duration', 'default': 'None', 'annotation': 'Optional[int]'}], 'return': 'float'}"></skdecide-signature>

#### Returns
 The task progress (float) between t_from and t_to.
 

### get\_task\_resuming\_type <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "get_task_resuming_type" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, ResumeType]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type ResumeType indicating
if the task can be resumed (restarted from where it was paused with no time loss)
or restarted (restarted from the start).
E.g. {
        1: ResumeType.NA
        2: ResumeType.Resume
        3: ResumeType.NA
        4: ResumeType.NA
        5: ResumeType.Restart
        6: ResumeType.NA
        }

### get\_time\_lags <Badge text="WithTimeLag" type="warn"/>

<skdecide-signature name= "get_time_lags" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, TimeLag]]'}"></skdecide-signature>

Return nested dictionaries where the first key is the id of a task (int)
and the second key is the id of another task (int).
The value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end
of the first task to the start of the second task.

e.g.
    {
        12:{
            15: TimeLag(5, 10),
            16: TimeLag(5, 20),
            17: MinimumOnlyTimeLag(5),
            18: MaximumOnlyTimeLag(15),
        }
    }

#### Returns
A dictionary of TimeLag objects.

### get\_time\_window <Badge text="WithTimeWindow" type="warn"/>

<skdecide-signature name= "get_time_window" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, TimeWindow]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value is a TimeWindow object.
Note that the max time horizon needs to be provided to the TimeWindow constructors
e.g.
    {
        1: TimeWindow(10, 15, 20, 30, self.get_max_horizon())
        2: EmptyTimeWindow(self.get_max_horizon())
        3: EndTimeWindow(20, 25, self.get_max_horizon())
        4: EndBeforeOnlyTimeWindow(40, self.get_max_horizon())
    }

#### Returns
A dictionary of TimeWindow objects.

### get\_transition\_value <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "get_transition_value" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_agent[TransitionValue[D.T_value]]'}"></skdecide-signature>

Get the value (reward or cost) of a transition.

The transition to consider is defined by the function parameters.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `UncertainTransitions._is_transition_value_dependent_on_next_state_()` to return
False. This information can then be exploited by solvers to avoid computing next state to evaluate a
transition value (more efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
The transition value (reward or cost).

### get\_variable\_resource\_consumption <Badge text="VariableResourceConsumption" type="warn"/>

<skdecide-signature name= "get_variable_resource_consumption" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return true if the domain has variable resource consumption,
false if the consumption of resource does not vary in time for any of the tasks

### initialize\_domain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "initialize_domain" :sig="{'params': [{'name': 'self'}]}"></skdecide-signature>

Initialize a scheduling domain. This function needs to be called when instantiating a scheduling domain.

### is\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_action" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is an action (i.e. a controllable event for the agents).

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
action space provided by `Events.get_action_space()`, but it can be overridden for faster implementations.
:::

#### Parameters
- **event**: The event to consider.

#### Returns
True if the event is an action (False otherwise).

### is\_applicable\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_applicable_action" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
omitted.

By default, `Events.is_applicable_action()` provides some boilerplate code and internally
calls `Events._is_applicable_action()`. The boilerplate code automatically passes the `_memory` attribute instead
of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the action is applicable (False otherwise).

### is\_enabled\_event <Badge text="Events" type="warn"/>

<skdecide-signature name= "is_enabled_event" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an uncontrollable event is enabled in the given memory (state or history), or in the
internal one if omitted.

By default, `Events.is_enabled_event()` provides some boilerplate code and internally
calls `Events._is_enabled_event()`. The boilerplate code automatically passes the `_memory` attribute instead of
the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the event is enabled (False otherwise).

### is\_goal <Badge text="Goals" type="warn"/>

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

### is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable.get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### is\_terminal <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "is_terminal" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether a state is terminal.

A terminal state is a state with no outgoing transition (except to itself with value 0).

#### Parameters
- **state**: The state to consider.

#### Returns
True if the state is terminal (False otherwise).

### is\_transition\_value\_dependent\_on\_next\_state <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "is_transition_value_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether get_transition_value() requires the next_state parameter for its computation (cached).

By default, `UncertainTransitions.is_transition_value_dependent_on_next_state()` internally
calls `UncertainTransitions._is_transition_value_dependent_on_next_state_()` the first time and automatically
caches its value to make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable.reset()` provides some boilerplate code and internally calls `Initializable._reset()`
(which returns an initial state). The boilerplate code automatically stores the initial state into the `_memory`
attribute and samples a corresponding observation.

#### Returns
An initial observation.

### sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation.sample()` provides some boilerplate code and internally calls `Simulation._sample()`
(which returns a transition outcome). The boilerplate code automatically samples an observation corresponding to
the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation.sample()` to call the external simulator and not use
the `Simulation._sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### sample\_completion\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "sample_completion_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}], 'return': 'List[int]'}"></skdecide-signature>

Samples the condition distributions associated with the given task and return a list of sampled
conditions.

### sample\_quantity\_resource <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "sample_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Sample an amount of resource availability (int) for the given resource
(either resource type or resource unit) at the given time. This number should be the sum of the number of
resource available at time t and the number of resource of this type consumed so far).

### sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Sample, store and return task duration for the given task in the given mode.

### set\_inplace\_environment <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "set_inplace_environment" :sig="{'params': [{'name': 'self'}, {'name': 'inplace_environment', 'annotation': 'bool'}]}"></skdecide-signature>

Activate or not the fact that the simulator modifies the given state inplace or create a copy before.
The inplace version is several times faster but will lead to bugs in graph search solver.

### set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with successive `Environment.step()`
calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain.set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain.step(my_action)
```

### solve\_with <Badge text="Domain" type="warn"/>

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver', 'annotation': 'Solver'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver**: The solver.
- **domain_factory**: A callable with no argument returning the domain to solve (factory is the domain class if None).
- **load_path**: The path to restore the solver state from (if None, the solving process will be launched instead).

#### Returns
The new solver (auto-cast to the level of the domain).

### step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment.step()` provides some boilerplate code and internally calls `Environment._step()` (which
returns a transition outcome). The boilerplate code automatically stores next state into the `_memory` attribute
and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment.step()` to call the external environment and not
use the `Environment._step()` helper function.
:::

::: warning
Before calling `Environment.step()` the first time or when the end of an episode is
reached, `Initializable.reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

### update\_complete\_dummy\_tasks <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_complete_dummy_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the status of newly started tasks whose duration is 0 from ongoing to complete.

### update\_complete\_dummy\_tasks\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_complete_dummy_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update the status of newly started tasks whose duration is 0
from ongoing to complete.

### update\_complete\_dummy\_tasks\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_complete_dummy_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of newly started tasks whose duration is 0
from ongoing to complete.

### update\_complete\_tasks <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_complete_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}]}"></skdecide-signature>

Update the status of newly completed tasks in the state from ongoing to complete
and update resource availability. This function will also log in task_details the time it was complete

### update\_complete\_tasks\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_complete_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}]}"></skdecide-signature>

In a simulated scheduling environment, update the status of newly completed tasks in the state from ongoing to complete
and update resource availability. This function will also log in task_details the time it was complete

### update\_complete\_tasks\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_complete_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of newly completed tasks in the state from ongoing
to complete, update resource availability and update on-completion conditions.
This function will also log in task_details the time it was complete.

### update\_conditional\_tasks <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_conditional_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update remaining tasks by checking conditions and potentially adding conditional tasks.

### update\_conditional\_tasks\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_conditional_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update remaining tasks by checking conditions and potentially
adding conditional tasks.

### update\_conditional\_tasks\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_conditional_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update remaining tasks by checking conditions and potentially adding conditional tasks.

### update\_pause\_tasks <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_pause_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the status of a task from ongoing to paused if specified in the action
and update resource availability. This function will also log in task_details the time it was paused.

### update\_pause\_tasks\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_pause_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulation scheduling environment, update the status of a task from ongoing to paused if
specified in the action and update resource availability. This function will also log in task_details
the time it was paused.

### update\_pause\_tasks\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_pause_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of a task from ongoing to paused if
specified in the action and update resource availability. This function will also log in task_details
the time it was paused.

### update\_progress <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_progress" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}]}"></skdecide-signature>

Update the progress of all ongoing tasks in the state.

### update\_progress\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_progress_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}]}"></skdecide-signature>

In a simulation scheduling environment, update the progress of all ongoing tasks in the state.

### update\_progress\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_progress_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the progress of all ongoing tasks in the state.

### update\_resource\_availability <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_resource_availability" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update resource availability for next time step. This should be called after update_time().

### update\_resource\_availability\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_resource_availability_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update resource availability for next time step.
This should be called after update_time().

### update\_resource\_availability\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_resource_availability_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update resource availability for next time step. This should be called after update_time().

### update\_resume\_tasks <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_resume_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the status of a task from paused to ongoing if specified in the action
and update resource availability. This function will also log in task_details the time it was resumed

### update\_resume\_tasks\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_resume_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulationn scheduling environment, update the status of a task from paused to ongoing if specified
in the action and update resource availability. This function will also log in task_details the time it was
resumed.

### update\_resume\_tasks\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_resume_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of a task from paused to ongoing if specified
in the action and update resource availability. This function will also log in task_details the time it was
resumed.

### update\_start\_tasks <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_start_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the status of a task from remaining to ongoing if specified in the action
and update resource availability. This function will also log in task_details the time it was started.

### update\_start\_tasks\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_start_tasks_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update the status of a task from remaining to ongoing if
specified in the action and update resource availability. This function will also log in task_details the
time it was started.

### update\_start\_tasks\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_start_tasks_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In an uncertain scheduling environment, update the status of a task from remaining to ongoing
if specified in the action and update resource availability.
This function returns a DsicreteDistribution of State.
This function will also log in task_details the time it was started.

### update\_time <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_time" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the time of the state if the time_progress attribute of the given EnumerableAction is True.

### update\_time\_simulation <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_time_simulation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

In a simulated scheduling environment, update the time of the state if the time_progress attribute of the
given EnumerableAction is True.

### update\_time\_uncertain <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "update_time_uncertain" :sig="{'params': [{'name': 'self'}, {'name': 'states', 'annotation': 'DiscreteDistribution[State]'}, {'name': 'action', 'annotation': 'SchedulingAction'}]}"></skdecide-signature>

Update the time of the state if the time_progress attribute of the given EnumerableAction is True.

### \_add\_to\_current\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_add_to_current_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'state'}]}"></skdecide-signature>

Samples completion conditions for a given task and add these conditions to the list of conditions in the
given state. This function should be called when a task complete.

### \_check\_value <Badge text="Rewards" type="warn"/>

<skdecide-signature name= "_check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### \_get\_action\_space <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_action_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the (cached) domain action space (finite or infinite set).

By default, `Events._get_action_space()` internally calls `Events._get_action_space_()` the first time and
automatically caches its value to make future calls more efficient (since the action space is assumed to be
constant).

#### Returns
The action space.

### \_get\_action\_space\_ <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_action_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

To be implemented if needed one day.

### \_get\_all\_resources\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "_get_all_resources_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, Dict[str, Any]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a resource type or resource unit
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {unit: {skill: (detail of skill)}} 

### \_get\_all\_tasks\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "_get_all_tasks_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, Dict[str, Any]]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a task
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {task: {skill: (detail of skill)}} 

### \_get\_all\_unconditional\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_get_all_unconditional_tasks" :sig="{'params': [{'name': 'self'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids for which there are no conditions. These tasks are to be considered at
the start of a project (i.e. in the initial state). 

### \_get\_applicable\_actions <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_applicable_actions" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
the internal one if omitted.

By default, `Events._get_applicable_actions()` provides some boilerplate code and internally
calls `Events._get_applicable_actions_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of applicable actions.

### \_get\_applicable\_actions\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_applicable_actions_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Returns the action space from a state.
TODO : think about a way to avoid the instaceof usage.

### \_get\_available\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_get_available_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids that can be considered under the conditions defined in the given state.
Note that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks
that are remaining, or that have been completed, paused or started / resumed.

### \_get\_enabled\_events <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_enabled_events" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history), or in the internal one if omitted.

By default, `Events._get_enabled_events()` provides some boilerplate code and internally
calls `Events._get_enabled_events_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
The space of enabled events.

### \_get\_enabled\_events\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_get_enabled_events_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history).

This is a helper function called by default from `Events._get_enabled_events()`, the difference being that the
memory parameter is mandatory here.

#### Parameters
- **memory**: The memory to consider.

#### Returns
The space of enabled events.

### \_get\_goals <Badge text="Goals" type="warn"/>

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

### \_get\_goals\_ <Badge text="Goals" type="warn"/>

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

### \_get\_initial\_state <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the (cached) initial state.

By default, `DeterministicInitialized._get_initial_state()` internally
calls `DeterministicInitialized._get_initial_state_()` the first time and automatically caches its value to make
future calls more efficient (since the initial state is assumed to be constant).

#### Returns
The initial state.

### \_get\_initial\_state\_ <Badge text="DeterministicInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Create and return an empty initial state

### \_get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized._get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### \_get\_initial\_state\_distribution\_ <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_distribution_" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the probability distribution of initial states.

This is a helper function called by default from `UncertainInitialized._get_initial_state_distribution()`, the
difference being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The probability distribution of initial states.

### \_get\_max\_horizon <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "_get_max_horizon" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Return the maximum time horizon (int)

### \_get\_memory\_maxlen <Badge text="History" type="warn"/>

<skdecide-signature name= "_get_memory_maxlen" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Get the (cached) memory max length.

By default, `FiniteHistory._get_memory_maxlen()` internally calls `FiniteHistory._get_memory_maxlen_()` the first
time and automatically caches its value to make future calls more efficient (since the memory max length is
assumed to be constant).

#### Returns
The memory max length.

### \_get\_memory\_maxlen\_ <Badge text="FiniteHistory" type="warn"/>

<skdecide-signature name= "_get_memory_maxlen_" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Get the memory max length.

This is a helper function called by default from `FiniteHistory._get_memory_maxlen()`, the difference being that
the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The memory max length.

### \_get\_mode\_costs <Badge text="WithModeCosts" type="warn"/>

<skdecide-signature name= "_get_mode_costs" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, float]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode
and the value indicates the cost of execution the task in the mode.

### \_get\_next\_state <Badge text="DeterministicTransitions" type="warn"/>

<skdecide-signature name= "_get_next_state" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'D.T_state'}"></skdecide-signature>

This function will be used if the domain is defined with DeterministicTransitions. This function will be ignored
if the domain is defined as having UncertainTransitions or Simulation. 

### \_get\_next\_state\_distribution <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

This function will be used if the domain is defined with UncertainTransitions. This function will be ignored
if the domain is defined as a Simulation. This function may also be used by uncertainty-specialised solvers
 on deterministic domains.

### \_get\_objectives <Badge text="SchedulingDomain" type="warn"/>

<skdecide-signature name= "_get_objectives" :sig="{'params': [{'name': 'self'}], 'return': 'List[SchedulingObjectiveEnum]'}"></skdecide-signature>

Return the objectives to consider as a list. The items should be of SchedulingObjectiveEnum type.

### \_get\_observation <Badge text="TransformedObservable" type="warn"/>

<skdecide-signature name= "_get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable._get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### \_get\_observation\_space\_ <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

To be implemented if needed one day.

### \_get\_preallocations <Badge text="WithPreallocations" type="warn"/>

<skdecide-signature name= "_get_preallocations" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[str]]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value indicates the pre-allocated resources for this task (as a list of str)

### \_get\_predecessors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "_get_predecessors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the predecessors of the task. Successors are given as a list for a task given as a key.

### \_get\_resource\_cost\_per\_time\_unit <Badge text="WithResourceCosts" type="warn"/>

<skdecide-signature name= "_get_resource_cost_per_time_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, float]'}"></skdecide-signature>

Return a dictionary where the key is the name of a resource (str)
and the value indicates the cost of using this resource per time unit.

### \_get\_resource\_renewability <Badge text="MixedRenewable" type="warn"/>

<skdecide-signature name= "_get_resource_renewability" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, bool]'}"></skdecide-signature>

Return a dictionary where the key is a resource name (string)
and the value whether this resource is renewable (True) or not (False).

### \_get\_resource\_type\_for\_unit <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "_get_resource_type_for_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, str]'}"></skdecide-signature>

Return a dictionary where the key is a resource unit name and the value a resource type name.
An empty dictionary can be used if there are no resource unit matching a resource type.

### \_get\_resource\_types\_names <Badge text="WithResourceTypes" type="warn"/>

<skdecide-signature name= "_get_resource_types_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource types as a list.

### \_get\_resource\_units\_names <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "_get_resource_units_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource units as a list.

### \_get\_successors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "_get_successors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the successors of the tasks. Successors are given as a list for a task given as a key.

### \_get\_task\_existence\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_get_task_existence_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)
for the task to be part of the schedule. If a task has no entry in the dictionary,
there is no conditions for that task.

Example:
    return
         {
            20: [get_all_condition_items().NC_PART_1_OPERATION_1],
            21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]
            22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]
         }e

### \_get\_task\_paused\_non\_renewable\_resource\_returned <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "_get_task_paused_non_renewable_resource_returned" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type bool indicating
if the non-renewable resources are consumed when the task is paused (False) or made available again (True).
E.g. {
        2: False  # if paused, non-renewable resource will be consumed
        5: True  # if paused, the non-renewable resource will be available again
        }

### \_get\_task\_preemptivity <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "_get_task_preemptivity" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a boolean indicating
if the task can be paused or stopped.
E.g. {
        1: False
        2: True
        3: False
        4: False
        5: True
        6: False
        }

### \_get\_task\_progress <Badge text="CustomTaskProgress" type="warn"/>

<skdecide-signature name= "_get_task_progress" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 't_from', 'annotation': 'int'}, {'name': 't_to', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'Optional[int]'}, {'name': 'sampled_duration', 'default': 'None', 'annotation': 'Optional[int]'}], 'return': 'float'}"></skdecide-signature>

#### Returns
 The task progress (float) between t_from and t_to.
 

### \_get\_task\_resuming\_type <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "_get_task_resuming_type" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, ResumeType]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type ResumeType indicating
if the task can be resumed (restarted from where it was paused with no time loss)
or restarted (restarted from the start).
E.g. {
        1: ResumeType.NA
        2: ResumeType.Resume
        3: ResumeType.NA
        4: ResumeType.NA
        5: ResumeType.Restart
        6: ResumeType.NA
        }

### \_get\_tasks\_ids <Badge text="MultiMode" type="warn"/>

<skdecide-signature name= "_get_tasks_ids" :sig="{'params': [{'name': 'self'}], 'return': 'Union[Set[int], Dict[int, Any], List[int]]'}"></skdecide-signature>

Return a set or dict of int = id of tasks

### \_get\_tasks\_modes <Badge text="MultiMode" type="warn"/>

<skdecide-signature name= "_get_tasks_modes" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, ModeConsumption]]'}"></skdecide-signature>

Return a nested dictionary where the first key is a task id and the second key is a mode id.
 The value is a Mode object defining the resource consumption.
If the domain is an instance of VariableResourceConsumption, VaryingModeConsumption objects should be used.
If this is not the case (i.e. the domain is an instance of ConstantResourceConsumption),
then ConstantModeConsumption should be used.

E.g. with constant resource consumption
    {
        12: {
                1: ConstantModeConsumption({'rt_1': 2, 'rt_2': 0, 'ru_1': 1}),
                2: ConstantModeConsumption({'rt_1': 0, 'rt_2': 3, 'ru_1': 1}),
            }
    }

E.g. with time varying resource consumption
    {
    12: {
        1: VaryingModeConsumption({'rt_1': [2,2,2,2,3], 'rt_2': [0,0,0,0,0], 'ru_1': [1,1,1,1,1]}),
        2: VaryingModeConsumption({'rt_1': [1,1,1,1,2,2,2], 'rt_2': [0,0,0,0,0,0,0], 'ru_1': [1,1,1,1,1,1,1]}),
        }
    }

### \_get\_time\_lags <Badge text="WithTimeLag" type="warn"/>

<skdecide-signature name= "_get_time_lags" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, TimeLag]]'}"></skdecide-signature>

Return nested dictionaries where the first key is the id of a task (int)
and the second key is the id of another task (int).
The value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end
of the first task to the start of the second task.

e.g.
    {
        12:{
            15: TimeLag(5, 10),
            16: TimeLag(5, 20),
            17: MinimumOnlyTimeLag(5),
            18: MaximumOnlyTimeLag(15),
        }
    }

#### Returns
A dictionary of TimeLag objects.

### \_get\_time\_window <Badge text="WithTimeWindow" type="warn"/>

<skdecide-signature name= "_get_time_window" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, TimeWindow]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value is a TimeWindow object.
Note that the max time horizon needs to be provided to the TimeWindow constructors
e.g.
    {
        1: TimeWindow(10, 15, 20, 30, self.get_max_horizon())
        2: EmptyTimeWindow(self.get_max_horizon())
        3: EndTimeWindow(20, 25, self.get_max_horizon())
        4: EndBeforeOnlyTimeWindow(40, self.get_max_horizon())
    }

#### Returns
A dictionary of TimeWindow objects.

### \_get\_variable\_resource\_consumption <Badge text="VariableResourceConsumption" type="warn"/>

<skdecide-signature name= "_get_variable_resource_consumption" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return true if the domain has variable resource consumption,
false if the consumption of resource does not vary in time for any of the tasks

### \_init\_memory <Badge text="History" type="warn"/>

<skdecide-signature name= "_init_memory" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_memory[D.T_state]'}"></skdecide-signature>

Initialize memory (possibly with a state) according to its specification and return it.

This function is automatically called by `Initializable._reset()` to reinitialize the internal memory whenever
the domain is used as an environment.

#### Parameters
- **state**: An optional state to initialize the memory with (typically the initial state).

#### Returns
The new initialized memory.

### \_is\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_action" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is an action (i.e. a controllable event for the agents).

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
action space provided by `Events._get_action_space()`, but it can be overridden for faster implementations.
:::

#### Parameters
- **event**: The event to consider.

#### Returns
True if the event is an action (False otherwise).

### \_is\_applicable\_action <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_applicable_action" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
omitted.

By default, `Events._is_applicable_action()` provides some boilerplate code and internally
calls `Events._is_applicable_action_from()`. The boilerplate code automatically passes the `_memory` attribute
instead of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the action is applicable (False otherwise).

### \_is\_applicable\_action\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_applicable_action_from" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_event]'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an action is applicable in the given memory (state or history).

This is a helper function called by default from `Events._is_applicable_action()`, the difference being that the
memory parameter is mandatory here.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the space of
applicable actions provided by `Events._get_applicable_actions_from()`, but it can be overridden for faster
implementations.
:::

#### Parameters
- **memory**: The memory to consider.

#### Returns
True if the action is applicable (False otherwise).

### \_is\_enabled\_event <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_enabled_event" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an uncontrollable event is enabled in the given memory (state or history), or in the
internal one if omitted.

By default, `Events._is_enabled_event()` provides some boilerplate code and internally
calls `Events._is_enabled_event_from()`. The boilerplate code automatically passes the `_memory` attribute instead
of the memory parameter whenever the latter is None.

#### Parameters
- **memory**: The memory to consider (if None, the internal memory attribute `_memory` is used instead).

#### Returns
True if the event is enabled (False otherwise).

### \_is\_enabled\_event\_from <Badge text="Events" type="warn"/>

<skdecide-signature name= "_is_enabled_event_from" :sig="{'params': [{'name': 'self'}, {'name': 'event', 'annotation': 'D.T_event'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether an event is enabled in the given memory (state or history).

This is a helper function called by default from `Events._is_enabled_event()`, the difference being that the
memory parameter is mandatory here.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the space of
enabled events provided by `Events._get_enabled_events_from()`, but it can be overridden for faster
implementations.
:::

#### Parameters
- **memory**: The memory to consider.

#### Returns
True if the event is enabled (False otherwise).

### \_is\_goal <Badge text="Goals" type="warn"/>

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

### \_is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable._get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### \_is\_transition\_value\_dependent\_on\_next\_state <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_transition_value_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether _get_transition_value() requires the next_state parameter for its computation (cached).

By default, `UncertainTransitions._is_transition_value_dependent_on_next_state()` internally
calls `UncertainTransitions._is_transition_value_dependent_on_next_state_()` the first time and automatically
caches its value to make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### \_is\_transition\_value\_dependent\_on\_next\_state\_ <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_transition_value_dependent_on_next_state_" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether _get_transition_value() requires the next_state parameter for its computation.

This is a helper function called by default
from `UncertainTransitions._is_transition_value_dependent_on_next_state()`, the difference being that the result
is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### \_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable._reset()` provides some boilerplate code and internally
calls `Initializable._state_reset()` (which returns an initial state). The boilerplate code automatically stores
the initial state into the `_memory` attribute and samples a corresponding observation.

#### Returns
An initial observation.

### \_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation._sample()` provides some boilerplate code and internally
calls `Simulation._state_sample()` (which returns a transition outcome). The boilerplate code automatically
samples an observation corresponding to the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation._sample()` to call the external simulator and not use
the `Simulation._state_sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### \_sample\_completion\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_sample_completion_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}], 'return': 'List[int]'}"></skdecide-signature>

Samples the condition distributions associated with the given task and return a list of sampled
conditions.

### \_sample\_quantity\_resource <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "_sample_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Sample an amount of resource availability (int) for the given resource
(either resource type or resource unit) at the given time. This number should be the sum of the number of
resource available at time t and the number of resource of this type consumed so far).

### \_sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "_sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode.

### \_set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with
successive `Environment._step()` calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain._set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain._step(my_action)
```

### \_state\_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_state_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Reset the state of the environment and return an initial state.

This is a helper function called by default from `Initializable._reset()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Returns
An initial state.

### \_state\_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_state_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

This function will be used if the domain is defined as a Simulation (i.e. transitions are defined by call to
a simulation). This function may also be used by simulation-based solvers on non-Simulation domains.

### \_state\_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_state_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one step of the transition's dynamics.

This is a helper function called by default from `Environment._step()`. It focuses on the state level, as opposed
to the observation one for the latter.

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The transition outcome of this step.

### \_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment._step()` provides some boilerplate code and internally
calls `Environment._state_step()` (which returns a transition outcome). The boilerplate code automatically stores
next state into the `_memory` attribute and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment._step()` to call the external environment and not
use the `Environment._state_step()` helper function.
:::

::: warning
Before calling `Environment._step()` the first time or when the end of an episode is
reached, `Initializable._reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

