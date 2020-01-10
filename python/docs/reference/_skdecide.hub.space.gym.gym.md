# hub.space.gym.gym

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## GymSpace

This class wraps an OpenAI Gym space (gym.spaces) as a scikit-decide space.

::: warning
Using this class requires OpenAI Gym to be installed.
:::

### Constructor <Badge text="GymSpace" type="tip"/>

<skdecide-signature name= "GymSpace" :sig="{'params': [{'name': 'gym_space', 'annotation': 'gym.Space'}], 'return': 'None'}"></skdecide-signature>

Initialize GymSpace.

#### Parameters
- **gym_space**: The Gym space (gym.spaces) to wrap.

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

### from\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "from_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Sequence'}], 'return': 'Iterable[T]'}"></skdecide-signature>

Convert a JSONable data type to a batch of samples from this space.

#### Parameters
- **sample_n**: The JSONable data type to convert.

#### Returns
The resulting batch of samples.

### sample <Badge text="SamplableSpace" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Uniformly randomly sample a random element of this space.

#### Returns
The sampled element.

### to\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "to_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Iterable[T]'}], 'return': 'Sequence'}"></skdecide-signature>

Convert a batch of samples from this space to a JSONable data type.

#### Parameters
- **sample_n**: The batch of samples to convert.

#### Returns
The resulting JSONable data type.

### unwrapped <Badge text="GymSpace" type="tip"/>

<skdecide-signature name= "unwrapped" :sig="{'params': [{'name': 'self'}], 'return': 'gym.Space'}"></skdecide-signature>

Unwrap the Gym space (gym.spaces) and return it.

#### Returns
The original Gym space.

## BoxSpace

This class wraps an OpenAI Gym Box space (gym.spaces.Box) as a scikit-decide space.

::: warning
Using this class requires OpenAI Gym to be installed.
:::

### Constructor <Badge text="BoxSpace" type="tip"/>

<skdecide-signature name= "BoxSpace" :sig="{'params': [{'name': 'low'}, {'name': 'high'}, {'name': 'shape', 'default': 'None'}, {'name': 'dtype', 'default': '<class \'numpy.float32\'>'}]}"></skdecide-signature>

Initialize GymSpace.

#### Parameters
- **gym_space**: The Gym space (gym.spaces) to wrap.

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

### from\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "from_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Sequence'}], 'return': 'Iterable[T]'}"></skdecide-signature>

Convert a JSONable data type to a batch of samples from this space.

#### Parameters
- **sample_n**: The JSONable data type to convert.

#### Returns
The resulting batch of samples.

### sample <Badge text="SamplableSpace" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Uniformly randomly sample a random element of this space.

#### Returns
The sampled element.

### to\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "to_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Iterable[T]'}], 'return': 'Sequence'}"></skdecide-signature>

Convert a batch of samples from this space to a JSONable data type.

#### Parameters
- **sample_n**: The batch of samples to convert.

#### Returns
The resulting JSONable data type.

### unwrapped <Badge text="GymSpace" type="warn"/>

<skdecide-signature name= "unwrapped" :sig="{'params': [{'name': 'self'}], 'return': 'gym.Space'}"></skdecide-signature>

Unwrap the Gym space (gym.spaces) and return it.

#### Returns
The original Gym space.

## DiscreteSpace

This class wraps an OpenAI Gym Discrete space (gym.spaces.Discrete) as a scikit-decide space.

::: warning
Using this class requires OpenAI Gym to be installed.
:::

### Constructor <Badge text="DiscreteSpace" type="tip"/>

<skdecide-signature name= "DiscreteSpace" :sig="{'params': [{'name': 'n'}]}"></skdecide-signature>

Initialize GymSpace.

#### Parameters
- **gym_space**: The Gym space (gym.spaces) to wrap.

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

### from\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "from_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Sequence'}], 'return': 'Iterable[T]'}"></skdecide-signature>

Convert a JSONable data type to a batch of samples from this space.

#### Parameters
- **sample_n**: The JSONable data type to convert.

#### Returns
The resulting batch of samples.

### sample <Badge text="SamplableSpace" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Uniformly randomly sample a random element of this space.

#### Returns
The sampled element.

### to\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "to_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Iterable[T]'}], 'return': 'Sequence'}"></skdecide-signature>

Convert a batch of samples from this space to a JSONable data type.

#### Parameters
- **sample_n**: The batch of samples to convert.

#### Returns
The resulting JSONable data type.

### unwrapped <Badge text="GymSpace" type="warn"/>

<skdecide-signature name= "unwrapped" :sig="{'params': [{'name': 'self'}], 'return': 'gym.Space'}"></skdecide-signature>

Unwrap the Gym space (gym.spaces) and return it.

#### Returns
The original Gym space.

## MultiDiscreteSpace

This class wraps an OpenAI Gym MultiDiscrete space (gym.spaces.MultiDiscrete) as a scikit-decide space.

::: warning
Using this class requires OpenAI Gym to be installed.
:::

### Constructor <Badge text="MultiDiscreteSpace" type="tip"/>

<skdecide-signature name= "MultiDiscreteSpace" :sig="{'params': [{'name': 'nvec'}]}"></skdecide-signature>

Initialize GymSpace.

#### Parameters
- **gym_space**: The Gym space (gym.spaces) to wrap.

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

### from\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "from_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Sequence'}], 'return': 'Iterable[T]'}"></skdecide-signature>

Convert a JSONable data type to a batch of samples from this space.

#### Parameters
- **sample_n**: The JSONable data type to convert.

#### Returns
The resulting batch of samples.

### sample <Badge text="SamplableSpace" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Uniformly randomly sample a random element of this space.

#### Returns
The sampled element.

### to\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "to_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Iterable[T]'}], 'return': 'Sequence'}"></skdecide-signature>

Convert a batch of samples from this space to a JSONable data type.

#### Parameters
- **sample_n**: The batch of samples to convert.

#### Returns
The resulting JSONable data type.

### unwrapped <Badge text="GymSpace" type="warn"/>

<skdecide-signature name= "unwrapped" :sig="{'params': [{'name': 'self'}], 'return': 'gym.Space'}"></skdecide-signature>

Unwrap the Gym space (gym.spaces) and return it.

#### Returns
The original Gym space.

## MultiBinarySpace

This class wraps an OpenAI Gym MultiBinary space (gym.spaces.MultiBinary) as a scikit-decide space.

::: warning
Using this class requires OpenAI Gym to be installed.
:::

### Constructor <Badge text="MultiBinarySpace" type="tip"/>

<skdecide-signature name= "MultiBinarySpace" :sig="{'params': [{'name': 'n'}]}"></skdecide-signature>

Initialize GymSpace.

#### Parameters
- **gym_space**: The Gym space (gym.spaces) to wrap.

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

### from\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "from_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Sequence'}], 'return': 'Iterable[T]'}"></skdecide-signature>

Convert a JSONable data type to a batch of samples from this space.

#### Parameters
- **sample_n**: The JSONable data type to convert.

#### Returns
The resulting batch of samples.

### sample <Badge text="SamplableSpace" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Uniformly randomly sample a random element of this space.

#### Returns
The sampled element.

### to\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "to_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Iterable[T]'}], 'return': 'Sequence'}"></skdecide-signature>

Convert a batch of samples from this space to a JSONable data type.

#### Parameters
- **sample_n**: The batch of samples to convert.

#### Returns
The resulting JSONable data type.

### unwrapped <Badge text="GymSpace" type="warn"/>

<skdecide-signature name= "unwrapped" :sig="{'params': [{'name': 'self'}], 'return': 'gym.Space'}"></skdecide-signature>

Unwrap the Gym space (gym.spaces) and return it.

#### Returns
The original Gym space.

## TupleSpace

This class wraps an OpenAI Gym Tuple space (gym.spaces.Tuple) as a scikit-decide space.

::: warning
Using this class requires OpenAI Gym to be installed.
:::

### Constructor <Badge text="TupleSpace" type="tip"/>

<skdecide-signature name= "TupleSpace" :sig="{'params': [{'name': 'spaces'}]}"></skdecide-signature>

Initialize GymSpace.

#### Parameters
- **gym_space**: The Gym space (gym.spaces) to wrap.

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

### from\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "from_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Sequence'}], 'return': 'Iterable[T]'}"></skdecide-signature>

Convert a JSONable data type to a batch of samples from this space.

#### Parameters
- **sample_n**: The JSONable data type to convert.

#### Returns
The resulting batch of samples.

### sample <Badge text="SamplableSpace" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Uniformly randomly sample a random element of this space.

#### Returns
The sampled element.

### to\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "to_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Iterable[T]'}], 'return': 'Sequence'}"></skdecide-signature>

Convert a batch of samples from this space to a JSONable data type.

#### Parameters
- **sample_n**: The batch of samples to convert.

#### Returns
The resulting JSONable data type.

### unwrapped <Badge text="GymSpace" type="warn"/>

<skdecide-signature name= "unwrapped" :sig="{'params': [{'name': 'self'}], 'return': 'gym.Space'}"></skdecide-signature>

Unwrap the Gym space (gym.spaces) and return it.

#### Returns
The original Gym space.

## DictSpace

This class wraps an OpenAI Gym Dict space (gym.spaces.Dict) as a scikit-decide space.

::: warning
Using this class requires OpenAI Gym to be installed.
:::

### Constructor <Badge text="DictSpace" type="tip"/>

<skdecide-signature name= "DictSpace" :sig="{'params': [{'name': 'spaces', 'default': 'None'}, {'name': 'spaces_kwargs'}]}"></skdecide-signature>

Initialize GymSpace.

#### Parameters
- **gym_space**: The Gym space (gym.spaces) to wrap.

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

### from\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "from_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Sequence'}], 'return': 'Iterable[T]'}"></skdecide-signature>

Convert a JSONable data type to a batch of samples from this space.

#### Parameters
- **sample_n**: The JSONable data type to convert.

#### Returns
The resulting batch of samples.

### sample <Badge text="SamplableSpace" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Uniformly randomly sample a random element of this space.

#### Returns
The sampled element.

### to\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "to_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Iterable[T]'}], 'return': 'Sequence'}"></skdecide-signature>

Convert a batch of samples from this space to a JSONable data type.

#### Parameters
- **sample_n**: The batch of samples to convert.

#### Returns
The resulting JSONable data type.

### unwrapped <Badge text="GymSpace" type="warn"/>

<skdecide-signature name= "unwrapped" :sig="{'params': [{'name': 'self'}], 'return': 'gym.Space'}"></skdecide-signature>

Unwrap the Gym space (gym.spaces) and return it.

#### Returns
The original Gym space.

## EnumSpace

This class creates an OpenAI Gym Discrete space (gym.spaces.Discrete) from an enumeration and wraps it as a
scikit-decide enumerable space.

::: warning
Using this class requires OpenAI Gym to be installed.
:::

### Constructor <Badge text="EnumSpace" type="tip"/>

<skdecide-signature name= "EnumSpace" :sig="{'params': [{'name': 'enum_class', 'annotation': 'EnumMeta'}], 'return': 'None'}"></skdecide-signature>

Initialize EnumSpace.

#### Parameters
- **enum_class**: The enumeration class for creating the Gym Discrete space (gym.spaces.Discrete) to wrap.

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

### from\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "from_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Sequence'}], 'return': 'Iterable[T]'}"></skdecide-signature>

Convert a JSONable data type to a batch of samples from this space.

#### Parameters
- **sample_n**: The JSONable data type to convert.

#### Returns
The resulting batch of samples.

### get\_elements <Badge text="EnumerableSpace" type="warn"/>

<skdecide-signature name= "get_elements" :sig="{'params': [{'name': 'self'}], 'return': 'Iterable[T]'}"></skdecide-signature>

Get the elements of this space.

#### Returns
The elements of this space.

### sample <Badge text="SamplableSpace" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Uniformly randomly sample a random element of this space.

#### Returns
The sampled element.

### to\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "to_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Iterable[T]'}], 'return': 'Sequence'}"></skdecide-signature>

Convert a batch of samples from this space to a JSONable data type.

#### Parameters
- **sample_n**: The batch of samples to convert.

#### Returns
The resulting JSONable data type.

### unwrapped <Badge text="GymSpace" type="warn"/>

<skdecide-signature name= "unwrapped" :sig="{'params': [{'name': 'self'}], 'return': 'gym_spaces.Discrete'}"></skdecide-signature>

Unwrap the Gym Discrete space (gym.spaces.Discrete) and return it.

#### Returns
The original Gym Discrete space created from the enumeration.

## ListSpace

This class creates an OpenAI Gym Discrete space (gym.spaces.Discrete) from a list of elements and wraps it as a
scikit-decide enumerable space.

::: warning
Using this class requires OpenAI Gym to be installed.
:::

### Constructor <Badge text="ListSpace" type="tip"/>

<skdecide-signature name= "ListSpace" :sig="{'params': [{'name': 'elements', 'annotation': 'Iterable[T]'}], 'return': 'None'}"></skdecide-signature>

Initialize ListSpace.

#### Parameters
- **elements**: The list of elements for creating the Gym Discrete space (gym.spaces.Discrete) to wrap.

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

### from\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "from_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Sequence'}], 'return': 'Iterable[T]'}"></skdecide-signature>

Convert a JSONable data type to a batch of samples from this space.

#### Parameters
- **sample_n**: The JSONable data type to convert.

#### Returns
The resulting batch of samples.

### get\_elements <Badge text="EnumerableSpace" type="warn"/>

<skdecide-signature name= "get_elements" :sig="{'params': [{'name': 'self'}], 'return': 'Iterable[T]'}"></skdecide-signature>

Get the elements of this space.

#### Returns
The elements of this space.

### sample <Badge text="SamplableSpace" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Uniformly randomly sample a random element of this space.

#### Returns
The sampled element.

### to\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "to_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Iterable[T]'}], 'return': 'Sequence'}"></skdecide-signature>

Convert a batch of samples from this space to a JSONable data type.

#### Parameters
- **sample_n**: The batch of samples to convert.

#### Returns
The resulting JSONable data type.

### unwrapped <Badge text="GymSpace" type="warn"/>

<skdecide-signature name= "unwrapped" :sig="{'params': [{'name': 'self'}], 'return': 'gym_spaces.Discrete'}"></skdecide-signature>

Unwrap the Gym Discrete space (gym.spaces.Discrete) and return it.

#### Returns
The original Gym Discrete space created from the list.

## DataSpace

This class creates an OpenAI Gym Dict space (gym.spaces.Dict) from a dataclass and wraps it as a scikit-decide space.

::: warning
Using this class requires OpenAI Gym to be installed.
:::

### Constructor <Badge text="DataSpace" type="tip"/>

<skdecide-signature name= "DataSpace" :sig="{'params': [{'name': 'data_class', 'annotation': 'type'}, {'name': 'spaces', 'annotation': 'Union[Dict[str, gym.Space], List[Tuple[str, gym.Space]]]'}], 'return': 'None'}"></skdecide-signature>

Initialize DataSpace.

#### Parameters
- **data_class**: The dataclass for creating the Gym Dict space (gym.spaces.Dict) to wrap.
- **spaces**: The spaces description passed to the created Dict space (see gym.spaces.Dict constructor documentation).

#### Example
```python
from skdecide.wrappers.space import DataSpace

@dataclass(frozen=True)
class Action:
    position: int
    velocity: int

my_action_space = DataSpace(Action, {"position": gym.spaces.Discrete(2), "velocity": gym.spaces.Discrete(3)})
```

### contains <Badge text="Space" type="warn"/>

<skdecide-signature name= "contains" :sig="{'params': [{'name': 'self'}, {'name': 'x', 'annotation': 'T'}], 'return': 'bool'}"></skdecide-signature>

Check whether x is a valid member of this space.

#### Parameters
- **x**: The member to consider.

#### Returns
True if x is a valid member of this space (False otherwise).

### from\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "from_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Sequence'}], 'return': 'Iterable[T]'}"></skdecide-signature>

Convert a JSONable data type to a batch of samples from this space.

#### Parameters
- **sample_n**: The JSONable data type to convert.

#### Returns
The resulting batch of samples.

### sample <Badge text="SamplableSpace" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}], 'return': 'T'}"></skdecide-signature>

Uniformly randomly sample a random element of this space.

#### Returns
The sampled element.

### to\_jsonable <Badge text="SerializableSpace" type="warn"/>

<skdecide-signature name= "to_jsonable" :sig="{'params': [{'name': 'self'}, {'name': 'sample_n', 'annotation': 'Iterable[T]'}], 'return': 'Sequence'}"></skdecide-signature>

Convert a batch of samples from this space to a JSONable data type.

#### Parameters
- **sample_n**: The batch of samples to convert.

#### Returns
The resulting JSONable data type.

### unwrapped <Badge text="GymSpace" type="warn"/>

<skdecide-signature name= "unwrapped" :sig="{'params': [{'name': 'self'}], 'return': 'gym_spaces.Dict'}"></skdecide-signature>

Unwrap the Gym Dict space (gym.spaces.Dict) and return it.

#### Returns
The original Gym Dict space created from the dataclass.

