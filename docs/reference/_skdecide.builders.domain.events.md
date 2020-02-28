# builders.domain.events

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## Events

A domain must inherit this class if it handles events (controllable or not not by the agents).

### get\_action\_space <Badge text="Events" type="tip"/>

<skdecide-signature name= "get_action_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the (cached) domain action space (finite or infinite set).

By default, `Events.get_action_space()` internally calls `Events._get_action_space_()` the first time and
automatically caches its value to make future calls more efficient (since the action space is assumed to be
constant).

#### Returns
The action space.

### get\_applicable\_actions <Badge text="Events" type="tip"/>

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

### get\_enabled\_events <Badge text="Events" type="tip"/>

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

### is\_action <Badge text="Events" type="tip"/>

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

### is\_applicable\_action <Badge text="Events" type="tip"/>

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

### is\_enabled\_event <Badge text="Events" type="tip"/>

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

### \_get\_action\_space <Badge text="Events" type="tip"/>

<skdecide-signature name= "_get_action_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the (cached) domain action space (finite or infinite set).

By default, `Events._get_action_space()` internally calls `Events._get_action_space_()` the first time and
automatically caches its value to make future calls more efficient (since the action space is assumed to be
constant).

#### Returns
The action space.

### \_get\_action\_space\_ <Badge text="Events" type="tip"/>

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

### \_get\_applicable\_actions <Badge text="Events" type="tip"/>

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

### \_get\_applicable\_actions\_from <Badge text="Events" type="tip"/>

<skdecide-signature name= "_get_applicable_actions_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'D.T_agent[Space[D.T_event]]'}"></skdecide-signature>

Get the space (finite or infinite set) of applicable actions in the given memory (state or history).

This is a helper function called by default from `Events._get_applicable_actions()`, the difference being that
the memory parameter is mandatory here.

#### Parameters
- **memory**: The memory to consider.

#### Returns
The space of applicable actions.

### \_get\_enabled\_events <Badge text="Events" type="tip"/>

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

### \_get\_enabled\_events\_from <Badge text="Events" type="tip"/>

<skdecide-signature name= "_get_enabled_events_from" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'Space[D.T_event]'}"></skdecide-signature>

Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
history).

This is a helper function called by default from `Events._get_enabled_events()`, the difference being that the
memory parameter is mandatory here.

#### Parameters
- **memory**: The memory to consider.

#### Returns
The space of enabled events.

### \_is\_action <Badge text="Events" type="tip"/>

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

### \_is\_applicable\_action <Badge text="Events" type="tip"/>

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

### \_is\_applicable\_action\_from <Badge text="Events" type="tip"/>

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

### \_is\_enabled\_event <Badge text="Events" type="tip"/>

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

### \_is\_enabled\_event\_from <Badge text="Events" type="tip"/>

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

## Actions

A domain must inherit this class if it handles only actions (i.e. controllable events).

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

## UnrestrictedActions

A domain must inherit this class if it handles only actions (i.e. controllable events), which are always all
applicable.

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

