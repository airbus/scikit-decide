# builders.domain.memory

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## History

A domain must inherit this class if its full state history must be stored to compute its dynamics (non-Markovian
domain).

### \_get\_memory\_maxlen <Badge text="History" type="tip"/>

<skdecide-signature name= "_get_memory_maxlen" :sig="{'params': [{'name': 'self'}], 'return': 'Optional[int]'}"></skdecide-signature>

Get the memory max length (or None if unbounded).

::: tip
This function returns always None by default because the memory length is unbounded at this level.
:::

#### Returns
The memory max length (or None if unbounded).

### \_init\_memory <Badge text="History" type="tip"/>

<skdecide-signature name= "_init_memory" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_memory[D.T_state]'}"></skdecide-signature>

Initialize memory (possibly with a state) according to its specification and return it.

This function is automatically called by `Initializable._reset()` to reinitialize the internal memory whenever
the domain is used as an environment.

#### Parameters
- **state**: An optional state to initialize the memory with (typically the initial state).

#### Returns
The new initialized memory.

## FiniteHistory

A domain must inherit this class if the last N states must be stored to compute its dynamics (Markovian
domain of order N).

N is specified by the return value of the `FiniteHistory._get_memory_maxlen()` function.

### \_get\_memory\_maxlen <Badge text="History" type="warn"/>

<skdecide-signature name= "_get_memory_maxlen" :sig="{'params': [{'name': 'self'}], 'return': 'int'}"></skdecide-signature>

Get the (cached) memory max length.

By default, `FiniteHistory._get_memory_maxlen()` internally calls `FiniteHistory._get_memory_maxlen_()` the first
time and automatically caches its value to make future calls more efficient (since the memory max length is
assumed to be constant).

#### Returns
The memory max length.

### \_get\_memory\_maxlen\_ <Badge text="FiniteHistory" type="tip"/>

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

### \_init\_memory <Badge text="History" type="warn"/>

<skdecide-signature name= "_init_memory" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_memory[D.T_state]'}"></skdecide-signature>

Initialize memory (possibly with a state) according to its specification and return it.

This function is automatically called by `Initializable._reset()` to reinitialize the internal memory whenever
the domain is used as an environment.

#### Parameters
- **state**: An optional state to initialize the memory with (typically the initial state).

#### Returns
The new initialized memory.

## Markovian

A domain must inherit this class if only its last state must be stored to compute its dynamics (pure Markovian
domain).

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

### \_init\_memory <Badge text="History" type="warn"/>

<skdecide-signature name= "_init_memory" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_memory[D.T_state]'}"></skdecide-signature>

Initialize memory (possibly with a state) according to its specification and return it.

This function is automatically called by `Initializable._reset()` to reinitialize the internal memory whenever
the domain is used as an environment.

#### Parameters
- **state**: An optional state to initialize the memory with (typically the initial state).

#### Returns
The new initialized memory.

## Memoryless

A domain must inherit this class if it does not require any previous state(s) to be stored to compute its
dynamics.

A dice roll simulator is an example of memoryless domain (next states are independent of previous ones).

::: tip
Whenever an existing domain (environment, simulator...) needs to be wrapped instead of implemented fully in
scikit-decide (e.g. compiled ATARI games), Memoryless can be used because the domain memory (if any) would
be handled externally.
:::

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

### \_init\_memory <Badge text="History" type="warn"/>

<skdecide-signature name= "_init_memory" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_memory[D.T_state]'}"></skdecide-signature>

Initialize memory (possibly with a state) according to its specification and return it.

This function is automatically called by `Initializable._reset()` to reinitialize the internal memory whenever
the domain is used as an environment.

#### Parameters
- **state**: An optional state to initialize the memory with (typically the initial state).

#### Returns
The new initialized memory.

