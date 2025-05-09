# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from enum import EnumMeta
from typing import Any, Generic, Union

import gymnasium as gym
import gymnasium.spaces as gym_spaces
import numpy as np
from gymnasium.spaces.space import T_cov

from skdecide import EnumerableSpace, SamplableSpace, SerializableSpace, T


class GymSpace(Generic[T], SamplableSpace[T], SerializableSpace[T]):
    """This class wraps a gymnasium space (gym.spaces) as a scikit-decide space.

    !!! warning
        Using this class requires gymnasium to be installed.
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
    """This class wraps a gymnasium Box space (gym.spaces.Box) as a scikit-decide space.

    !!! warning
        Using this class requires gymnasium to be installed.
    """

    def __init__(self, low, high, shape=None, dtype=np.float32):
        super().__init__(gym_space=gym_spaces.Box(low, high, shape, dtype))


class DiscreteSpace(GymSpace[T], EnumerableSpace[T]):
    """This class wraps a gymnasium Discrete space (gym.spaces.Discrete) as a scikit-decide space.

    !!! warning
        Using this class requires gymnasium to be installed.
    """

    def __init__(self, n, element_class=int):
        super().__init__(gym_space=gym_spaces.Discrete(n))
        self._element_class = element_class

    def get_elements(self) -> Sequence[T]:
        """Get the elements of this space.

        # Returns
        The elements of this space.
        """
        return range(self._gym_space.n)

    def to_unwrapped(self, sample_n: Iterable[T]) -> Iterable:
        return (
            sample_n
            if self._element_class is int
            else [int(sample) for sample in sample_n]
        )

    def from_unwrapped(self, sample_n: Iterable) -> Iterable[T]:
        return (
            sample_n
            if self._element_class is int
            else [self._element_class(sample) for sample in sample_n]
        )


class MultiDiscreteSpace(GymSpace[T], EnumerableSpace[T]):
    """This class wraps a gymnasium MultiDiscrete space (gym.spaces.MultiDiscrete) as a scikit-decide space.

    !!! warning
        Using this class requires gymnasium to be installed.
    """

    def __init__(self, nvec, element_class=np.ndarray):
        super().__init__(gym_space=gym_spaces.MultiDiscrete(nvec))
        self._element_class = element_class

    def get_elements(self) -> Sequence[T]:
        """Get the elements of this space.

        # Returns
        The elements of this space.
        """
        return tuple(itertools.product(*(range(n) for n in self._gym_space.nvec)))

    def to_unwrapped(self, sample_n: Iterable[T]) -> Iterable:
        return (
            sample_n
            if self._element_class is np.ndarray
            else [np.asarray(sample, dtype=np.int64) for sample in sample_n]
        )

    def from_unwrapped(self, sample_n: Iterable) -> Iterable[T]:
        return (
            sample_n
            if self._element_class is np.ndarray
            else [self._element_class(sample) for sample in sample_n]
        )


class MaskableMultiDiscreteSpace(MultiDiscreteSpace[T]):
    """Maskable version of MultiDiscreteSpace.

    Elements can also contain -1 components which means that this component will be ignored.
    It is used to model variable length multidiscrete space, e.g. for parametric actions whose arity depend
    on the first component (the action type).

    So the i-th component of an element of the space will be an integer between -1 and nvec[i]-1  (both included).

    When unwrapping, we still return a MultiDiscrete(nvec) space as we still want to mean that the i-th component
    is a categorical variable with nvec[i] choices (although if not needed, it will use -1 as another choice)


    """

    def get_elements(self) -> Sequence[T]:
        return tuple(itertools.product(*(range(n) for n in self._gym_space.nvec)))

    def contains(self, x: T) -> bool:
        return gym_spaces.MultiDiscrete(self._gym_space.nvec + 1).contains(
            np.asarray(x) + 1
        )

    def sample(self) -> T:
        return gym_spaces.MultiDiscrete(self._gym_space.nvec + 1).sample() - 1


class MultiBinarySpace(GymSpace[T], EnumerableSpace[T]):
    """This class wraps a gymnasium MultiBinary space (gym.spaces.MultiBinary) as a scikit-decide space.

    !!! warning
        Using this class requires gymnasium to be installed.
    """

    def __init__(self, n, element_class=np.ndarray):
        super().__init__(gym_space=gym_spaces.MultiBinary(n))
        self._element_class = element_class

    def get_elements(self) -> Sequence[T]:
        """Get the elements of this space.

        # Returns
        The elements of this space.
        """
        return tuple(itertools.product(*((1, 0) for _ in range(self._gym_space.n))))

    def to_unwrapped(self, sample_n: Iterable[T]) -> Iterable:
        return (
            sample_n
            if self._element_class is np.ndarray
            else [np.asarray(sample, dtype=np.bool_).astype(int) for sample in sample_n]
        )

    def from_unwrapped(self, sample_n: Iterable) -> Iterable[T]:
        return (
            sample_n
            if self._element_class is np.ndarray
            else [self._element_class(sample) for sample in sample_n]
        )


class TupleSpace(GymSpace[T]):
    """This class wraps a gymnasium Tuple space (gym.spaces.Tuple) as a scikit-decide space.

    !!! warning
        Using this class requires gymnasium to be installed.
    """

    def __init__(
        self, spaces: tuple[Union[GymSpace[T], gym.Space]], element_class=tuple
    ):
        super().__init__(
            gym_space=gym_spaces.Tuple(
                [sp if isinstance(sp, gym.Space) else sp.unwrapped() for sp in spaces]
            )
        )
        self._spaces = spaces
        self._element_class = element_class
        assert element_class is tuple or all(
            m in dir(element_class) for m in ["to_tuple", "from_tuple"]
        ), "Tuple space's element class must be of type tuple or it must provide the to_tuple and from_tuple methods"
        self._to_tuple = (
            (lambda e: e) if element_class is tuple else (lambda e: e.to_tuple())
        )
        self._from_tuple = (
            (lambda e: e)
            if element_class is tuple
            else (lambda e: self._element_class.from_tuple(e))
        )

    def to_unwrapped(self, sample_n: Iterable[T]) -> Iterable:
        return [
            tuple(
                next(iter(self._spaces[i].to_unwrapped([e])))
                if isinstance(self._spaces[i], GymSpace)
                else e
                for i, e in enumerate(self._to_tuple(sample))
            )
            for sample in sample_n
        ]

    def from_unwrapped(self, sample_n: Iterable) -> Iterable[T]:
        return [
            self._from_tuple(
                tuple(
                    next(iter(self._spaces[i].from_unwrapped([e])))
                    if isinstance(self._spaces[i], GymSpace)
                    else e
                    for i, e in enumerate(sample)
                )
            )
            for sample in sample_n
        ]


class DictSpace(GymSpace[T]):
    """This class wraps a gymnasium Dict space (gym.spaces.Dict) as a scikit-decide space.

    !!! warning
        Using this class requires gymnasium to be installed.
    """

    def __init__(
        self,
        spaces: dict[Any, Union[GymSpace[T], gym.Space]] = None,
        element_class=dict,
        **spaces_kwargs,
    ):
        super().__init__(
            gym_space=gym_spaces.Dict(
                {
                    k: sp if isinstance(sp, gym.Space) else sp.unwrapped()
                    for k, sp in spaces.items()
                },
                **spaces_kwargs,
            )
        )
        self._spaces = spaces
        self._element_class = element_class
        assert element_class is dict or all(
            m in dir(element_class) for m in ["to_dict", "from_dict"]
        ), "Dict space's element class must be of type dict or it must provide the to_dict and from_dict methods"
        self._to_dict = (
            (lambda e: e) if element_class is dict else (lambda e: e.to_dict())
        )
        self._from_dict = (
            (lambda e: e)
            if element_class is dict
            else (lambda e: self._element_class.from_dict(e))
        )

    def to_unwrapped(self, sample_n: Iterable[T]) -> Iterable:
        return [
            {
                k: next(iter(self._spaces[k].to_unwrapped([e])))
                if isinstance(self._spaces[k], GymSpace)
                else e
                for k, e in self._to_dict(sample).items()
            }
            for sample in sample_n
        ]

    def from_unwrapped(self, sample_n: Iterable) -> Iterable[T]:
        return [
            self._from_dict(
                {
                    k: next(iter(self._spaces[k].from_unwrapped([e])))
                    if isinstance(self._spaces[k], GymSpace)
                    else e
                    for k, e in sample.items()
                }
            )
            for sample in sample_n
        ]


class EnumSpace(Generic[T], GymSpace[T], EnumerableSpace[T]):
    """This class creates a gymnasium Discrete space (gym.spaces.Discrete) from an enumeration and wraps it as a
    scikit-decide enumerable space.

    !!! warning
        Using this class requires gymnasium to be installed.
    """

    def __init__(self, enum_class: EnumMeta) -> None:
        """Initialize EnumSpace.

        # Parameters
        enum_class: The enumeration class for creating the Gym Discrete space (gym.spaces.Discrete) to wrap.
        """
        self._enum_class = enum_class
        self._list_enum = tuple(enum_class)
        gym_space = gym_spaces.Discrete(len(enum_class))
        super().__init__(gym_space)

    def contains(self, x: T) -> bool:
        return isinstance(x, self._enum_class)

    def get_elements(self) -> Sequence[T]:
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
    """This class creates a gymnasium Discrete space (gym.spaces.Discrete) from a list of elements and wraps it as a
    scikit-decide enumerable space. If ordering is not important contrary to the 'contains' test, it is advised to
    use the 'SetSpace' class instead.

    !!! warning
        Using this class requires gymnasium to be installed.
    """

    def __init__(self, elements: Iterable[T]) -> None:
        """Initialize ListSpace.

        # Parameters
        elements: The list of elements for creating the Gym Discrete space (gym.spaces.Discrete) to wrap.
        """
        self._elements = list(elements)
        if len(self._elements) > 0:
            gym_space = gym_spaces.Discrete(len(self._elements))
        else:
            gym_space = Empty()

        super().__init__(gym_space)

    def contains(self, x: T) -> bool:
        return x in self._elements

    def get_elements(self) -> Sequence[T]:
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


class SetSpace(Generic[T], GymSpace[T], EnumerableSpace[T]):
    """This class creates a gymnasium Discrete space (gym.spaces.Discrete) from a set of elements and wraps it as a
    scikit-decide enumerable space.

    !!! warning
        Using this class requires gymnasium to be installed.
    """

    def __init__(self, elements: Iterable[T]) -> None:
        """Initialize SetSpace.

        # Parameters
        elements: The set of elements for creating the Gym Discrete space (gym.spaces.Discrete) to wrap.
        """
        self._elements = set(elements)
        self._to_indexes = {e: i for i, e in enumerate(self._elements)}
        self._to_elements = [e for e in self._elements]
        gym_space = gym_spaces.Discrete(len(self._elements))
        super().__init__(gym_space)

    def contains(self, x: T) -> bool:
        return x in self._elements

    def get_elements(self) -> Sequence[T]:
        return self._elements

    def sample(self) -> T:
        return self._to_elements[super().sample()]

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
        return [self._to_indexes[sample] for sample in sample_n]

    def from_unwrapped(self, sample_n: Iterable[int]) -> Iterable[T]:
        return [self._to_elements[sample] for sample in sample_n]


class DataSpace(GymSpace[T]):
    """This class creates a gymnasium Dict space (gym.spaces.Dict) from a dataclass and wraps it as a scikit-decide space.

    !!! warning
        Using this class requires gymnasium to be installed.
    """

    def __init__(
        self,
        data_class: type,
        spaces: Union[dict[str, gym.Space], list[tuple[str, gym.Space]]],
    ) -> None:
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

    def to_unwrapped(self, sample_n: Iterable[T]) -> Iterable[dict]:
        return [asdict(sample) for sample in sample_n]

    def from_unwrapped(self, sample_n: Iterable[dict]) -> Iterable[T]:
        # TODO: convert to simple types (get rid of ndarray created by gym dict space...)?
        return [self._data_class(**sample) for sample in sample_n]


class VariableSpace(GymSpace[T]):

    """This class wraps a gymnasium Space (gym.spaces.Space) to allow dynamic length of elements."""

    def __init__(
        self,
        space: gym.Space,
        max_len: int,
        **kwargs,
    ):
        self._gym_space = space
        self.max_len = max_len
        self.size = (self.max_len, self._gym_space._shape[0])

    def sample(self):
        length = self.max_len
        return list(np.array(self._gym_space.sample()) for _ in range(length))

    def unwrapped(self):
        return gym.spaces.Box(
            low=self._gym_space.low.min(),
            high=self._gym_space.high.max(),
            shape=self.size,
        )

    def to_unwrapped(self, sample_n: Iterable[T]) -> Iterable:
        return [
            np.pad(
                np.array(v),
                ((0, self.max_len - len(v)), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            for v in sample_n
        ]

    def from_unwrapped(self, sample_n: Iterable) -> Iterable[T]:
        return [
            np.array(ligne)
            for ligne in [row for row in sample_n if not np.all(row == 0)]
        ]

    def __repr__(self):
        return f"RepeatedSpace({self._gym_space}, max_len={self.max_len})"


class Empty(gym.spaces.Space):
    @property
    def is_np_flattenable(self) -> bool:
        return False

    def sample(self, mask: Any | None = None) -> T_cov:
        raise RuntimeError("Cannot sample an empty space.")

    def contains(self, x: Any) -> bool:
        return False
