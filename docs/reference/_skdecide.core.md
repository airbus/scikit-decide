# core

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## Space

A space representing a finite or infinite set.

This class (or any of its descendant) is typically used to specify action, observation or goal spaces.

### contains <Badge text="Space" type="tip"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

## ImplicitSpace

A space formalized implicitly, i.e. by a black-box contains() function.

### Constructor <Badge text="ImplicitSpace" type="tip"/>

<skdecide-signature name= "ImplicitSpace" :sig="{'params': [{'name': 'contains_function', 'annotation': 'Callable[[T], bool]'}], 'return': 'None'}"></skdecide-signature>

Initialize ImplicitSpace.

#### Parameters
- **contains_function**: The contains() function to use.

#### Example
```python
my_space = ImplicitSpace(lambda x: 10 > x['position'] > 5)
```

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

## EnumerableSpace

A space which elements can be enumerated.

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

### get\_elements <Badge text="EnumerableSpace" type="tip"/>

<skdecide-signature name= "get_elements" :sig="{'params': [{'name': 'self'}], 'return': 'Iterable[T]'}"></skdecide-signature>

Get the elements of this space.

#### Returns
The elements of this space.

## EmptySpace

A space which elements can be enumerated.

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

## SamplableSpace

A space which can be sampled (uniformly randomly).

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

### sample <Badge text="SamplableSpace" type="tip"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Uniformly randomly sample a random element of this space.

#### Returns
The sampled element.

## SerializableSpace

A space which can be serialized (to/from JSON).

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

### from\_jsonable <Badge text="SerializableSpace" type="tip"/>

<skdecide-signature name= "from_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Sequence'}], 'return': 'Iterable[T]'}"></skdecide-signature>

Convert a JSONable data type to a batch of samples from this space.

#### Parameters
- **sample_n**: The JSONable data type to convert.

#### Returns
The resulting batch of samples.

### to\_jsonable <Badge text="SerializableSpace" type="tip"/>

<skdecide-signature name= "to_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Iterable[T]'}], 'return': 'Sequence'}"></skdecide-signature>

Convert a batch of samples from this space to a JSONable data type.

#### Parameters
- **sample_n**: The batch of samples to convert.

#### Returns
The resulting JSONable data type.

## Distribution

A probability distribution.

### sample <Badge text="Distribution" type="tip"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Sample from this distribution.

#### Returns
The sampled element.

## ImplicitDistribution

A probability distribution formalized implicitly, i.e. by a black-box sample() function.

### Constructor <Badge text="ImplicitDistribution" type="tip"/>

<skdecide-signature name= "ImplicitDistribution" :sig="{'params': [{'name': 'sample_function', 'annotation': 'Callable[[], T]'}], 'return': 'None'}"></skdecide-signature>

Initialize ImplicitDistribution.

#### Parameters
- **sample_function**: The sample() function to use.

#### Example
```python
import random

dice = ImplicitDistribution(lambda: random.randint(1, 6))
roll = dice.sample()
```

### sample <Badge text="Distribution" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Sample from this distribution.

#### Returns
The sampled element.

## DiscreteDistribution

A discrete probability distribution.

### Constructor <Badge text="DiscreteDistribution" type="tip"/>

<skdecide-signature name= "DiscreteDistribution" :sig="{'params': [{'name': 'values', 'annotation': 'List[Tuple[T, float]]'}], 'return': 'None'}"></skdecide-signature>

Initialize DiscreteDistribution.

::: tip
If the given probabilities do not sum to 1, they are implicitly normalized as such for sampling.
:::

#### Parameters
- **values**: The list of (element, probability) pairs.

#### Example
```python
game_strategy = DiscreteDistribution([('rock', 0.7), ('paper', 0.1), ('scissors', 0.2)])
move = game_strategy.sample()
```

### get\_values <Badge text="DiscreteDistribution" type="tip"/>

<skdecide-signature name= "get_values" :sig="{'params': [{'name': 'self'}], 'return': 'List[Tuple[T, float]]'}"></skdecide-signature>

Get the list of (element, probability) pairs.

#### Returns
The (element, probability) pairs.

### sample <Badge text="Distribution" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Sample from this distribution.

#### Returns
The sampled element.

## SingleValueDistribution

A single value distribution (i.e. Dirac distribution).

### Constructor <Badge text="SingleValueDistribution" type="tip"/>

<skdecide-signature name= "SingleValueDistribution" :sig="{'params': [{'name': 'value', 'annotation': 'T'}], 'return': 'None'}"></skdecide-signature>

Initialize SingleValueDistribution.

#### Parameters
- **value**: The single value of this distribution.

### get\_value <Badge text="SingleValueDistribution" type="tip"/>

<skdecide-signature name= "get_value" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Get the single value of this distribution.

#### Returns
The single value of this distribution.

### get\_values <Badge text="DiscreteDistribution" type="warn"/>

<skdecide-signature name= "get_values" :sig="{'params': [{'name': 'self'}], 'return': 'List[Tuple[T, float]]'}"></skdecide-signature>

Get the list of (element, probability) pairs.

#### Returns
The (element, probability) pairs.

### sample <Badge text="Distribution" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Sample from this distribution.

#### Returns
The sampled element.

## TransitionValue

A transition value (reward or cost).

::: warning
It is recommended to use either the reward or the cost parameter. If no one is used, a reward/cost of 0 is
assumed. If both are used, reward will be considered and cost ignored. In any case, both reward and cost
attributes will be defined after initialization.
:::

#### Parameters
- **reward**: The optional reward.
- **cost**: The optional cost.

#### Example
```python
# These two lines are equivalent, use the one you prefer
value_1 = TransitionValue(reward=-5)
value_2 = TransitionValue(cost=5)

assert value_1.reward == value_2.reward == -5  # True
assert value_1.cost == value_2.cost == 5  # True
```

## EnvironmentOutcome

An environment outcome for an internal transition.

#### Parameters
- **observation**: The agent's observation of the current environment.
- **value**: The value (reward or cost) returned after previous action.
- **termination**: Whether the episode has ended, in which case further step() calls will return undefined results.
- **info**: Optional auxiliary diagnostic information (helpful for debugging, and sometimes learning).

### asdict <Badge text="ExtendedDataclass" type="warn"/>

<skdecide-signature name= "asdict" :sig="{'params': [{'name': 'self'}]}"></skdecide-signature>

Return the fields of the instance as a new dictionary mapping field names to field values.

### astuple <Badge text="ExtendedDataclass" type="warn"/>

<skdecide-signature name= "astuple" :sig="{'params': [{'name': 'self'}]}"></skdecide-signature>

Return the fields of the instance as a new tuple of field values.

### replace <Badge text="ExtendedDataclass" type="warn"/>

<skdecide-signature name= "replace" :sig="{'params': [{'name': 'self'}, {'name': 'changes'}]}"></skdecide-signature>

Return a new object replacing specified fields with new values.

## TransitionOutcome

A transition outcome.

#### Parameters
- **state**: The new state after the transition.
- **value**: The value (reward or cost) returned after previous action.
- **termination**: Whether the episode has ended, in which case further step() calls will return undefined results.
- **info**: Optional auxiliary diagnostic information (helpful for debugging, and sometimes learning).

### asdict <Badge text="ExtendedDataclass" type="warn"/>

<skdecide-signature name= "asdict" :sig="{'params': [{'name': 'self'}]}"></skdecide-signature>

Return the fields of the instance as a new dictionary mapping field names to field values.

### astuple <Badge text="ExtendedDataclass" type="warn"/>

<skdecide-signature name= "astuple" :sig="{'params': [{'name': 'self'}]}"></skdecide-signature>

Return the fields of the instance as a new tuple of field values.

### replace <Badge text="ExtendedDataclass" type="warn"/>

<skdecide-signature name= "replace" :sig="{'params': [{'name': 'self'}, {'name': 'changes'}]}"></skdecide-signature>

Return a new object replacing specified fields with new values.

## StrDict

A dictionary with String keys (e.g. agent names).

## Constraint

A constraint.

### check <Badge text="Constraint" type="tip"/>

<skdecide-signature name= "check" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory'}, {'name': 'action', 'annotation': 'D.T_event'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Check this constraint.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `Constraint._is_constraint_dependent_on_next_state_()` to return False. This
information can then be exploited by solvers to avoid computing next state to evaluate the constraint (more
efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
True if the constraint is checked (False otherwise).

### is\_constraint\_dependent\_on\_next\_state <Badge text="Constraint" type="tip"/>

<skdecide-signature name= "is_constraint_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether this constraint requires the next_state parameter for its computation (cached).

By default, `Constraint.is_constraint_dependent_on_next_state()` internally
calls `Constraint._is_constraint_dependent_on_next_state_()` the first time and automatically caches its value to
make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the constraint computation depends on next_state (False otherwise).

### \_is\_constraint\_dependent\_on\_next\_state\_ <Badge text="Constraint" type="tip"/>

<skdecide-signature name= "_is_constraint_dependent_on_next_state_" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether this constraint requires the next_state parameter for its computation.

This is a helper function called by default from `Constraint.is_constraint_dependent_on_next_state()`, the
difference being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
True if the constraint computation depends on next_state (False otherwise).

## ImplicitConstraint

A constraint formalized implicitly, i.e. by a black-box check() function.

### Constructor <Badge text="ImplicitConstraint" type="tip"/>

<skdecide-signature name= "ImplicitConstraint" :sig="{'params': [{'name': 'check_function', 'annotation': 'Callable[[D.T_memory, D.T_event, Optional[D.T_state]], bool]'}, {'name': 'depends_on_next_state', 'default': 'True', 'annotation': 'bool'}], 'return': 'None'}"></skdecide-signature>

Initialize ImplicitConstraint.

#### Parameters
- **check_function**: The check() function to use.
- **depends_on_next_state**: Whether the check() function requires the next_state parameter for its computation.

#### Example
```python
constraint = ImplicitConstraint(lambda memory, action, next_state: next_state.x % 2 == 0)
```

### check <Badge text="Constraint" type="warn"/>

<skdecide-signature name= "check" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory'}, {'name': 'action', 'annotation': 'D.T_event'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Check this constraint.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `Constraint._is_constraint_dependent_on_next_state_()` to return False. This
information can then be exploited by solvers to avoid computing next state to evaluate the constraint (more
efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
True if the constraint is checked (False otherwise).

### is\_constraint\_dependent\_on\_next\_state <Badge text="Constraint" type="warn"/>

<skdecide-signature name= "is_constraint_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether this constraint requires the next_state parameter for its computation (cached).

By default, `Constraint.is_constraint_dependent_on_next_state()` internally
calls `Constraint._is_constraint_dependent_on_next_state_()` the first time and automatically caches its value to
make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the constraint computation depends on next_state (False otherwise).

### \_is\_constraint\_dependent\_on\_next\_state\_ <Badge text="Constraint" type="warn"/>

<skdecide-signature name= "_is_constraint_dependent_on_next_state_" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether this constraint requires the next_state parameter for its computation.

This is a helper function called by default from `Constraint.is_constraint_dependent_on_next_state()`, the
difference being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
True if the constraint computation depends on next_state (False otherwise).

## BoundConstraint

A constraint characterized by an evaluation function, an inequality and a bound.

#### Example
A BoundConstraint with inequality '>=' is checked if (and only if) its `BoundConstraint.evaluate()` function returns
a float greater than or equal to its bound.

### Constructor <Badge text="BoundConstraint" type="tip"/>

<skdecide-signature name= "BoundConstraint" :sig="{'params': [{'name': 'evaluate_function', 'annotation': 'Callable[[D.T_memory, D.T_event, Optional[D.T_state]], float]'}, {'name': 'inequality', 'annotation': 'str'}, {'name': 'bound', 'annotation': 'float'}, {'name': 'depends_on_next_state', 'default': 'True', 'annotation': 'bool'}], 'return': 'None'}"></skdecide-signature>

Initialize BoundConstraint.

#### Parameters
- **evaluate_function**: The evaluate() function to use.
- **inequality**: A string ('\<', '\<=', '>' or '>=') describing the constraint inequality.
- **bound**: The bound of the constraint.
- **depends_on_next_state**: Whether the evaluate() function requires the next_state parameter for its computation.

#### Example
```python
constraint = BoundConstraint((lambda memory, action, next_state: next_state.x), '>', 5.)
```

### check <Badge text="Constraint" type="warn"/>

<skdecide-signature name= "check" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory'}, {'name': 'action', 'annotation': 'D.T_event'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'bool'}"></skdecide-signature>

Check this constraint.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `Constraint._is_constraint_dependent_on_next_state_()` to return False. This
information can then be exploited by solvers to avoid computing next state to evaluate the constraint (more
efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
True if the constraint is checked (False otherwise).

### evaluate <Badge text="BoundConstraint" type="tip"/>

<skdecide-signature name= "evaluate" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory'}, {'name': 'action', 'annotation': 'D.T_event'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'float'}"></skdecide-signature>

Evaluate the left side of this BoundConstraint.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `Constraint._is_constraint_dependent_on_next_state_()` to return False. This
information can then be exploited by solvers to avoid computing next state to evaluate the constraint (more
efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
The float value resulting from the evaluation.

### get\_bound <Badge text="BoundConstraint" type="tip"/>

<skdecide-signature name= "get_bound" :sig="{'params': [{'name': 'self'}], 'return': 'float'}"></skdecide-signature>

Get the bound of the constraint.

#### Returns
The constraint bound.

### get\_inequality <Badge text="BoundConstraint" type="tip"/>

<skdecide-signature name= "get_inequality" :sig="{'params': [{'name': 'self'}], 'return': 'str'}"></skdecide-signature>

Get the string ('\<', '\<=', '>' or '>=') describing the constraint inequality.

#### Returns
The string describing the inequality.

### is\_constraint\_dependent\_on\_next\_state <Badge text="Constraint" type="warn"/>

<skdecide-signature name= "is_constraint_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether this constraint requires the next_state parameter for its computation (cached).

By default, `Constraint.is_constraint_dependent_on_next_state()` internally
calls `Constraint._is_constraint_dependent_on_next_state_()` the first time and automatically caches its value to
make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the constraint computation depends on next_state (False otherwise).

### \_is\_constraint\_dependent\_on\_next\_state\_ <Badge text="Constraint" type="warn"/>

<skdecide-signature name= "_is_constraint_dependent_on_next_state_" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether this constraint requires the next_state parameter for its computation.

This is a helper function called by default from `Constraint.is_constraint_dependent_on_next_state()`, the
difference being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
True if the constraint computation depends on next_state (False otherwise).

