# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import EnumMeta
from typing import Union, Generic, Iterable, Sequence, Tuple, List, Dict

import gym
import gym.spaces as gym_spaces
import numpy as np

from skdecide import T, EnumerableSpace, SamplableSpace, SerializableSpace
from dataclasses import asdict


class GymSpace(Generic[T], SamplableSpace[T], SerializableSpace[T]):
    """This class wraps an OpenAI Gym space (gym.spaces) as a scikit-decide space.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_space: gym.Space) -> None:
        """Initialize GymSpace.

        # Parameters
        gym_space: The Gym space (gym.spaces) to wrap.
        """
        super().__init__()
        self._gym_space = gym_space
        self.shape = gym_space.shape  # TODO: remove if unnecessary?
        self.dtype = gym_space.dtype  # TODO: remove if unnecessary?

    def contains(self, x: T) -> bool:
        return self._gym_space.contains(x)

    def sample(self) -> T:
        return self._gym_space.sample()

    def to_jsonable(self, sample_n: Iterable[T]) -> Sequence:
        return self._gym_space.to_jsonable(sample_n)

    def from_jsonable(self, sample_n: Sequence) -> Iterable[T]:
        return self._gym_space.from_jsonable(sample_n)

    def unwrapped(self) -> gym.Space:
        """Unwrap the Gym space (gym.spaces) and return it.

        # Returns
        The original Gym space.
        """
        return self._gym_space

    def to_unwrapped(self, sample_n: Iterable[T]) -> Iterable:
        return sample_n

    def from_unwrapped(self, sample_n: Iterable) -> Iterable[T]:
        return sample_n


class BoxSpace(GymSpace[T]):
    """This class wraps an OpenAI Gym Box space (gym.spaces.Box) as a scikit-decide space.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, low, high, shape=None, dtype=np.float32):
        super().__init__(gym_space=gym_spaces.Box(low, high, shape, dtype))


class DiscreteSpace(GymSpace[T]):
    """This class wraps an OpenAI Gym Discrete space (gym.spaces.Discrete) as a scikit-decide space.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, n):
        super().__init__(gym_space=gym_spaces.Discrete(n))


class MultiDiscreteSpace(GymSpace[T]):
    """This class wraps an OpenAI Gym MultiDiscrete space (gym.spaces.MultiDiscrete) as a scikit-decide space.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, nvec):
        super().__init__(gym_space=gym_spaces.MultiDiscrete(nvec))


class MultiBinarySpace(GymSpace[T]):
    """This class wraps an OpenAI Gym MultiBinary space (gym.spaces.MultiBinary) as a scikit-decide space.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, n):
        super().__init__(gym_space=gym_spaces.MultiBinary(n))


class TupleSpace(GymSpace[T]):
    """This class wraps an OpenAI Gym Tuple space (gym.spaces.Tuple) as a scikit-decide space.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, spaces):
        super().__init__(gym_space=gym_spaces.Tuple(spaces))


class DictSpace(GymSpace[T]):
    """This class wraps an OpenAI Gym Dict space (gym.spaces.Dict) as a scikit-decide space.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, spaces=None, **spaces_kwargs):
        super().__init__(gym_space=gym_spaces.Dict(spaces, **spaces_kwargs))


class EnumSpace(Generic[T], GymSpace[T], EnumerableSpace[T]):
    """This class creates an OpenAI Gym Discrete space (gym.spaces.Discrete) from an enumeration and wraps it as a
    scikit-decide enumerable space.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, enum_class: EnumMeta) -> None:
        """Initialize EnumSpace.

        # Parameters
        enum_class: The enumeration class for creating the Gym Discrete space (gym.spaces.Discrete) to wrap.
        """
        self._enum_class = enum_class
        self._list_enum = list(enum_class)
        gym_space = gym_spaces.Discrete(len(enum_class))
        super().__init__(gym_space)

    def contains(self, x: T) -> bool:
        return isinstance(x, self._enum_class)

    def get_elements(self) -> Iterable[T]:
        return self._list_enum

    def sample(self) -> T:
        return self._list_enum[super().sample()]

    def to_jsonable(self, sample_n: Iterable[T]) -> Sequence:
        return [sample.name for sample in sample_n]

    def from_jsonable(self, sample_n: Sequence) -> Iterable[T]:
        return [self._enum_class[sample] for sample in sample_n]

    def unwrapped(self) -> gym_spaces.Discrete:
        """Unwrap the Gym Discrete space (gym.spaces.Discrete) and return it.

        # Returns
        The original Gym Discrete space created from the enumeration.
        """
        return super().unwrapped()

    def to_unwrapped(self, sample_n: Iterable[T]) -> Iterable[int]:
        return [self._list_enum.index(sample) for sample in sample_n]

    def from_unwrapped(self, sample_n: Iterable[int]) -> Iterable[T]:
        return [self._list_enum[sample] for sample in sample_n]


class ListSpace(Generic[T], GymSpace[T], EnumerableSpace[T]):
    """This class creates an OpenAI Gym Discrete space (gym.spaces.Discrete) from a list of elements and wraps it as a
    scikit-decide enumerable space.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, elements: Iterable[T]) -> None:
        """Initialize ListSpace.

        # Parameters
        elements: The list of elements for creating the Gym Discrete space (gym.spaces.Discrete) to wrap.
        """
        self._elements = list(elements)
        gym_space = gym_spaces.Discrete(len(self._elements))
        super().__init__(gym_space)

    def contains(self, x: T) -> bool:
        return x in self._elements

    def get_elements(self) -> Iterable[T]:
        return self._elements

    def sample(self) -> T:
        return self._elements[super().sample()]

    def to_jsonable(self, sample_n: Iterable[T]) -> Sequence:
        return sample_n

    def from_jsonable(self, sample_n: Sequence) -> Iterable[T]:
        return sample_n

    def unwrapped(self) -> gym_spaces.Discrete:
        """Unwrap the Gym Discrete space (gym.spaces.Discrete) and return it.

        # Returns
        The original Gym Discrete space created from the list.
        """
        return super().unwrapped()

    def to_unwrapped(self, sample_n: Iterable[T]) -> Iterable[int]:
        return [self._elements.index(sample) for sample in sample_n]

    def from_unwrapped(self, sample_n: Iterable[int]) -> Iterable[T]:
        return [self._elements[sample] for sample in sample_n]


class DataSpace(GymSpace[T]):
    """This class creates an OpenAI Gym Dict space (gym.spaces.Dict) from a dataclass and wraps it as a scikit-decide space.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, data_class: type, spaces: Union[Dict[str, gym.Space], List[Tuple[str, gym.Space]]]) -> None:
        """Initialize DataSpace.

        # Parameters
        data_class: The dataclass for creating the Gym Dict space (gym.spaces.Dict) to wrap.
        spaces: The spaces description passed to the created Dict space (see gym.spaces.Dict constructor documentation).

        # Example
        ```python
        from skdecide.wrappers.space import DataSpace

        @dataclass(frozen=True)
        class Action:
            position: int
            velocity: int

        my_action_space = DataSpace(Action, {"position": gym.spaces.Discrete(2), "velocity": gym.spaces.Discrete(3)})
        ```
        """
        self._data_class = data_class
        gym_space = gym_spaces.Dict(spaces)
        super().__init__(gym_space)

    def contains(self, x: T) -> bool:
        # works even when fields of the dataclass have been recast (e.g. numpy 0-dimensional array to scalar)
        return super().contains(super().from_jsonable(self.to_jsonable([x]))[0])
        # # bug: does not work when fields of the dataclass have been recast (e.g. numpy 0-dimensional array to scalar)
        # return super().contains(asdict(x))

    def sample(self) -> T:
        # TODO: convert to simple types (get rid of ndarray created by gym dict space...)?
        return self._data_class(**super().sample())

    def to_jsonable(self, sample_n: Iterable[T]) -> Sequence:
        dict_sample_n = self.to_unwrapped(sample_n)
        return super().to_jsonable(dict_sample_n)

    def from_jsonable(self, sample_n: Sequence) -> Iterable[T]:
        dict_sample_n = super().from_jsonable(sample_n)
        return self.from_unwrapped(dict_sample_n)

    def unwrapped(self) -> gym_spaces.Dict:
        """Unwrap the Gym Dict space (gym.spaces.Dict) and return it.

        # Returns
        The original Gym Dict space created from the dataclass.
        """
        return super().unwrapped()

    def to_unwrapped(self, sample_n: Iterable[T]) -> Iterable[Dict]:
        return [asdict(sample) for sample in sample_n]

    def from_unwrapped(self, sample_n: Iterable[Dict]) -> Iterable[T]:
        # TODO: convert to simple types (get rid of ndarray created by gym dict space...)?
        return [self._data_class(**sample) for sample in sample_n]
