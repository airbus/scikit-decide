# domains

This module contains base classes for quickly building domains.

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## Domain

This is the highest level domain class (inheriting top-level class for each mandatory domain characteristic).

This helper class can be used as the main base class for domains.

Typical use:
```python
class D(Domain, ...)
```

with "..." replaced when needed by a number of classes from following domain characteristics (the ones in
parentheses are optional):

- **agent**: MultiAgent -> SingleAgent
- **concurrency**: Parallel -> Sequential
- **(constraints)**: Constrained
- **dynamics**: Environment -> Simulation -> UncertainTransitions -> EnumerableTransitions
  -> DeterministicTransitions
- **events**: Events -> Actions -> UnrestrictedActions
- **(goals)**: Goals
- **(initialization)**: Initializable -> UncertainInitialized -> DeterministicInitialized
- **memory**: History -> FiniteHistory -> Markovian -> Memoryless
- **observability**: PartiallyObservable -> TransformedObservable -> FullyObservable
- **(renderability)**: Renderable
- **value**: Rewards -> PositiveCosts

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

### solve\_with <Badge text="Domain" type="tip"/>

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver_factory', 'annotation': 'Callable[[], Solver]'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver_factory**: A callable with no argument returning the new solver (can be just a solver class).
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

### \_get\_memory\_maxlen <Badge text="History" type="warn"/>

<skdecide-signature name= "_get_memory_maxlen" :sig="{'params': [{'name': 'self'}], 'return': 'Optional[int]'}"></skdecide-signature>

Get the memory max length (or None if unbounded).

::: tip
This function returns always None by default because the memory length is unbounded at this level.
:::

#### Returns
The memory max length (or None if unbounded).

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

## RLDomain

This is a typical Reinforcement Learning domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- SingleAgent
- Sequential
- Environment
- Actions
- Initializable
- Markovian
- TransformedObservable
- Rewards

Typical use:
```python
class D(RLDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

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

### solve\_with <Badge text="Domain" type="warn"/>

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver_factory', 'annotation': 'Callable[[], Solver]'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver_factory**: A callable with no argument returning the new solver (can be just a solver class).
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

### \_state\_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_state_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Reset the state of the environment and return an initial state.

This is a helper function called by default from `Initializable._reset()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Returns
An initial state.

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

## MultiAgentRLDomain

This is a typical multi-agent Reinforcement Learning domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- MultiAgent
- Sequential
- Environment
- Actions
- Initializable
- Markovian
- TransformedObservable
- Rewards

Typical use:
```python
class D(RLDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

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

### solve\_with <Badge text="Domain" type="warn"/>

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver_factory', 'annotation': 'Callable[[], Solver]'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver_factory**: A callable with no argument returning the new solver (can be just a solver class).
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

### \_state\_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_state_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Reset the state of the environment and return an initial state.

This is a helper function called by default from `Initializable._reset()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Returns
An initial state.

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

## StatelessSimulatorDomain

This is a typical stateless simulator domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- SingleAgent
- Sequential
- Simulation
- Actions
- Markovian
- TransformedObservable
- Rewards

Typical use:
```python
class D(StatelessSimulatorDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

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

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver_factory', 'annotation': 'Callable[[], Solver]'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver_factory**: A callable with no argument returning the new solver (can be just a solver class).
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

## MDPDomain

This is a typical Markov Decision Process domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- SingleAgent
- Sequential
- EnumerableTransitions
- Actions
- DeterministicInitialized
- Markovian
- FullyObservable
- Rewards

Typical use:
```python
class D(MDPDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

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

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver_factory', 'annotation': 'Callable[[], Solver]'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver_factory**: A callable with no argument returning the new solver (can be just a solver class).
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

<skdecide-signature name= "_get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'DiscreteDistribution[D.T_state]'}"></skdecide-signature>

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

## POMDPDomain

This is a typical Partially Observable Markov Decision Process domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- SingleAgent
- Sequential
- EnumerableTransitions
- Actions
- UncertainInitialized
- Markovian
- PartiallyObservable
- Rewards

Typical use:
```python
class D(POMDPDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

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

### get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized.get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

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

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver_factory', 'annotation': 'Callable[[], Solver]'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver_factory**: A callable with no argument returning the new solver (can be just a solver class).
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

<skdecide-signature name= "_get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'DiscreteDistribution[D.T_state]'}"></skdecide-signature>

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

## GoalMDPDomain

This is a typical Goal Markov Decision Process domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- SingleAgent
- Sequential
- EnumerableTransitions
- Actions
- Goals
- DeterministicInitialized
- Markovian
- FullyObservable
- PositiveCosts

Typical use:
```python
class D(GoalMDPDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

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

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver_factory', 'annotation': 'Callable[[], Solver]'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver_factory**: A callable with no argument returning the new solver (can be just a solver class).
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

Check that a transition value is compliant with its cost specification (must be positive).

::: tip
This function calls `PositiveCost._is_positive()` to determine if a value is positive (can be overridden for
advanced value types).
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

<skdecide-signature name= "_get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'DiscreteDistribution[D.T_state]'}"></skdecide-signature>

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

### \_is\_positive <Badge text="PositiveCosts" type="warn"/>

<skdecide-signature name= "_is_positive" :sig="{'params': [{'name': 'self'}, {'name': 'cost', 'annotation': 'D.T_value'}], 'return': 'bool'}"></skdecide-signature>

Determine if a value is positive (can be overridden for advanced value types).

#### Parameters
- **cost**: The cost to evaluate.

#### Returns
True if the cost is positive (False otherwise).

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

## GoalPOMDPDomain

This is a typical Goal Partially Observable Markov Decision Process domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- SingleAgent
- Sequential
- EnumerableTransitions
- Actions
- Goals
- UncertainInitialized
- Markovian
- PartiallyObservable
- PositiveCosts

Typical use:
```python
class D(GoalPOMDPDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

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

### get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized.get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

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

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver_factory', 'annotation': 'Callable[[], Solver]'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver_factory**: A callable with no argument returning the new solver (can be just a solver class).
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

Check that a transition value is compliant with its cost specification (must be positive).

::: tip
This function calls `PositiveCost._is_positive()` to determine if a value is positive (can be overridden for
advanced value types).
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

<skdecide-signature name= "_get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'DiscreteDistribution[D.T_state]'}"></skdecide-signature>

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

### \_is\_positive <Badge text="PositiveCosts" type="warn"/>

<skdecide-signature name= "_is_positive" :sig="{'params': [{'name': 'self'}, {'name': 'cost', 'annotation': 'D.T_value'}], 'return': 'bool'}"></skdecide-signature>

Determine if a value is positive (can be overridden for advanced value types).

#### Parameters
- **cost**: The cost to evaluate.

#### Returns
True if the cost is positive (False otherwise).

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

## DeterministicPlanningDomain

This is a typical deterministic planning domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- SingleAgent
- Sequential
- DeterministicTransitions
- Actions
- Goals
- DeterministicInitialized
- Markovian
- FullyObservable
- PositiveCosts

Typical use:
```python
class D(DeterministicPlanningDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

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

<skdecide-signature name= "solve_with" :sig="{'params': [{'name': 'solver_factory', 'annotation': 'Callable[[], Solver]'}, {'name': 'domain_factory', 'default': 'None', 'annotation': 'Optional[Callable[[], Domain]]'}, {'name': 'load_path', 'default': 'None', 'annotation': 'Optional[str]'}], 'return': 'Solver'}"></skdecide-signature>

Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

By default, `Solver.check_domain()` provides some boilerplate code and internally
calls `Solver._check_domain_additional()` (which returns True by default but can be overridden  to define
specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
domain requirements are met.

#### Parameters
- **solver_factory**: A callable with no argument returning the new solver (can be just a solver class).
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

Check that a transition value is compliant with its cost specification (must be positive).

::: tip
This function calls `PositiveCost._is_positive()` to determine if a value is positive (can be overridden for
advanced value types).
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

### \_is\_positive <Badge text="PositiveCosts" type="warn"/>

<skdecide-signature name= "_is_positive" :sig="{'params': [{'name': 'self'}, {'name': 'cost', 'annotation': 'D.T_value'}], 'return': 'bool'}"></skdecide-signature>

Determine if a value is positive (can be overridden for advanced value types).

#### Parameters
- **cost**: The cost to evaluate.

#### Returns
True if the cost is positive (False otherwise).

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

