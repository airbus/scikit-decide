# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import inspect
import random
import re
from dataclasses import asdict, astuple, dataclass, replace
from typing import (
    Callable,
    Deque,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

__all__ = [
    "T",
    "D",
    "Space",
    "ImplicitSpace",
    "EnumerableSpace",
    "EmptySpace",
    "SamplableSpace",
    "SerializableSpace",
    "Distribution",
    "ImplicitDistribution",
    "DiscreteDistribution",
    "SingleValueDistribution",
    "Value",
    "EnvironmentOutcome",
    "TransitionOutcome",
    "Memory",
    "StrDict",
    "Constraint",
    "ImplicitConstraint",
    "BoundConstraint",
    "autocast_all",
    "autocastable",
    "nocopy",
]

T = TypeVar("T")  # Any type


class D:
    T_state = TypeVar("T_state")  # Type of states
    T_observation = TypeVar("T_observation")  # Type of observations
    T_event = TypeVar("T_event")  # Type of events
    T_value = TypeVar("T_value")  # Type of transition values (rewards or costs)
    T_predicate = TypeVar("T_predicate")  # Type of logical checks
    T_info = TypeVar(
        "T_info"
    )  # Type of additional information given as part of an environment outcome
    T_memory = TypeVar("T_memory")
    T_agent = TypeVar("T_agent")
    T_concurrency = TypeVar("T_concurrency")


# Tree (utility class)
class Tree:
    def __init__(self, type_: object, sub: List[Tree] = []):
        self.type = type_
        self.sub = sub


# Castable (utility class)
class Castable:
    def _cast(self, src_sub: List[Tree], dst_sub: List[Tree]):
        raise NotImplementedError


# Space
class Space(Generic[T]):
    """A space representing a finite or infinite set.

    This class (or any of its descendant) is typically used to specify action, observation or goal spaces.
    """

    def contains(self, x: T) -> bool:
        """Check whether x is a valid member of this space.

        # Parameters
        x: The member to consider.

        # Returns
        True if x is a valid member of this space (False otherwise).
        """
        raise NotImplementedError

    def __contains__(self, item: T) -> bool:
        return self.contains(item)


class ImplicitSpace(Space[T]):
    """A space formalized implicitly, i.e. by a black-box contains() function."""

    def __init__(self, contains_function: Callable[[T], bool]) -> None:
        """Initialize ImplicitSpace.

        # Parameters
        contains_function: The contains() function to use.

        # Example
        ```python
        my_space = ImplicitSpace(lambda x: 10 > x['position'] > 5)
        ```
        """
        self.contains_function = contains_function

    def contains(self, x: T) -> bool:
        return self.contains_function(x)


class EnumerableSpace(Space[T]):
    """A space which elements can be enumerated."""

    def get_elements(self) -> Iterable[T]:
        """Get the elements of this space.

        # Returns
        The elements of this space.
        """
        raise NotImplementedError

    def contains(self, x: T) -> bool:
        return x in self.get_elements()


class EmptySpace(EnumerableSpace[T]):
    """An (enumerable) empty space."""

    def get_elements(self) -> Iterable[T]:
        return ()


class SamplableSpace(Space[T]):
    """A space which can be sampled (uniformly randomly)."""

    def sample(self) -> T:
        """Uniformly randomly sample a random element of this space.

        # Returns
        The sampled element.
        """
        raise NotImplementedError


class SerializableSpace(Space[T]):
    """A space which can be serialized (to/from JSON)."""

    def to_jsonable(self, sample_n: Iterable[T]) -> Sequence:
        """Convert a batch of samples from this space to a JSONable data type.

        # Parameters
        sample_n: The batch of samples to convert.

        # Returns
        The resulting JSONable data type.
        """
        # By default, assume identity is JSONable
        return sample_n

    def from_jsonable(self, sample_n: Sequence) -> Iterable[T]:
        """Convert a JSONable data type to a batch of samples from this space.

        # Parameters
        sample_n: The JSONable data type to convert.

        # Returns
        The resulting batch of samples.
        """
        # By default, assume identity is JSONable
        return sample_n


# Distribution
class Distribution(Generic[T], Castable):
    """A probability distribution."""

    def sample(self) -> T:
        """Sample from this distribution.

        # Returns
        The sampled element.
        """
        raise NotImplementedError


class ImplicitDistribution(Distribution[T]):
    """A probability distribution formalized implicitly, i.e. by a black-box sample() function."""

    def __init__(self, sample_function: Callable[[], T]) -> None:
        """Initialize ImplicitDistribution.

        # Parameters
        sample_function: The sample() function to use.

        # Example
        ```python
        import random

        dice = ImplicitDistribution(lambda: random.randint(1, 6))
        roll = dice.sample()
        ```
        """
        self._sample_function = sample_function

    def sample(self) -> T:
        return self._sample_function()

    def _cast(self, src_sub: List[Tree], dst_sub: List[Tree]):
        def cast_sample_function():
            result = self._sample_function()
            return cast(result, src_sub[0], dst_sub[0])

        return ImplicitDistribution(cast_sample_function)


class DiscreteDistribution(Distribution[T]):
    """A discrete probability distribution."""

    def __init__(self, values: List[Tuple[T, float]]) -> None:
        """Initialize DiscreteDistribution.

        !!! tip
            If the given probabilities do not sum to 1, they are implicitly normalized as such for sampling.

        # Parameters
        values: The list of (element, probability) pairs.

        # Example
        ```python
        game_strategy = DiscreteDistribution([('rock', 0.7), ('paper', 0.1), ('scissors', 0.2)])
        move = game_strategy.sample()
        ```
        """
        self._values = values
        unzip_values = list(zip(*values))
        # TODO: make sure every population member is unique (and if not, aggregate same ones by summing their weights)?
        self._population = unzip_values[0]
        self._weights = unzip_values[1]

    def sample(self) -> T:
        return random.choices(self._population, self._weights)[0]

    def get_values(self) -> List[Tuple[T, float]]:
        """Get the list of (element, probability) pairs.

        # Returns
        The (element, probability) pairs.
        """
        return self._values

    def _cast(self, src_sub: List[Tree], dst_sub: List[Tree]):
        return DiscreteDistribution(
            [(cast(e, src_sub[0], dst_sub[0]), p) for e, p in self._values]
        )


class SingleValueDistribution(DiscreteDistribution[T]):
    """A single value distribution (i.e. Dirac distribution)."""

    def __init__(self, value: T) -> None:
        """Initialize SingleValueDistribution.

        # Parameters
        value: The single value of this distribution.
        """
        self._value = value

    def sample(self) -> T:
        return self._value

    def get_values(self) -> List[Tuple[T, float]]:
        return [(self._value, 1.0)]

    def get_value(self) -> T:
        """Get the single value of this distribution.

        # Returns
        The single value of this distribution.
        """
        return self._value

    def _cast(self, src_sub: List[Tree], dst_sub: List[Tree]):
        return SingleValueDistribution(cast(self._value, src_sub[0], dst_sub[0]))


# ExtendedDataclass (dataclasses can inherit from it to extended their methods with useful utilities)
class ExtendedDataclass:
    def asdict(self):
        """Return the fields of the instance as a new dictionary mapping field names to field values."""
        return asdict(self)

    def astuple(self):
        """Return the fields of the instance as a new tuple of field values."""
        return astuple(self)

    def replace(self, **changes):
        """Return a new object replacing specified fields with new values."""
        return replace(self, **changes)


# Value
@dataclass
class Value(Generic[D.T_value]):
    """A value (reward or cost).

    !!! warning
        It is recommended to use either the reward or the cost parameter. If no one is used, a reward/cost of 0 is
        assumed. If both are used, reward will be considered and cost ignored. In any case, both reward and cost
        attributes will be defined after initialization.

    # Parameters
    reward: The optional reward.
    cost: The optional cost.

    # Example
    ```python
    # These two lines are equivalent, use the one you prefer
    value_1 = Value(reward=-5)
    value_2 = Value(cost=5)

    assert value_1.reward == value_2.reward == -5  # True
    assert value_1.cost == value_2.cost == 5  # True
    ```
    """

    reward: Optional[D.T_value] = None
    cost: Optional[D.T_value] = None

    def __post_init__(self) -> None:
        if self.reward is not None:
            self.cost = self._reward_to_cost(self.reward)
        elif self.cost is not None:
            self.reward = self._cost_to_reward(self.cost)
        else:
            self.reward = 0
            self.cost = 0

    def _cost_to_reward(self, cost: D.T_value) -> D.T_value:
        return -cost

    def _reward_to_cost(self, reward: D.T_value) -> D.T_value:
        return -reward


# EnvironmentOutcome
@dataclass
class EnvironmentOutcome(
    Generic[D.T_observation, D.T_value, D.T_predicate, D.T_info],
    Castable,
    ExtendedDataclass,
):
    """An environment outcome for an internal transition.

    # Parameters
    observation: The agent's observation of the current environment.
    value: The value (reward or cost) returned after previous action.
    termination: Whether the episode has ended, in which case further step() calls will return undefined results.
    info: Optional auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    """

    observation: D.T_observation
    value: Optional[D.T_value] = None
    termination: Optional[D.T_predicate] = None
    info: Optional[D.T_info] = None

    def __post_init__(self) -> None:
        if self.value is None:
            self.value = (
                {k: Value() for k in self.observation}
                if isinstance(self.observation, dict)
                else Value()
            )
        if self.termination is None:
            self.termination = (
                {k: False for k in self.observation}
                if isinstance(self.observation, dict)
                else False
            )
        if self.info is None:
            self.info = (
                {k: None for k in self.observation}
                if isinstance(self.observation, dict)
                else None
            )

    def _cast(self, src_sub: List[Tree], dst_sub: List[Tree]):
        return EnvironmentOutcome(
            cast(self.observation, src_sub[0], dst_sub[0]),
            cast(self.value, src_sub[1], dst_sub[1]),
            cast(self.termination, src_sub[2], dst_sub[2]),
            cast(self.info, src_sub[3], dst_sub[3]),
        )


# TransitionOutcome
@dataclass
class TransitionOutcome(
    Generic[D.T_state, D.T_value, D.T_predicate, D.T_info], Castable, ExtendedDataclass
):
    """A transition outcome.

    # Parameters
    state: The new state after the transition.
    value: The value (reward or cost) returned after previous action.
    termination: Whether the episode has ended, in which case further step() calls will return undefined results.
    info: Optional auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    """

    state: D.T_state
    value: Optional[D.T_value] = None
    termination: Optional[D.T_predicate] = None
    info: Optional[D.T_info] = None

    def __post_init__(self) -> None:
        if self.value is None:
            self.value = (
                {k: Value() for k in self.state}
                if isinstance(self.state, dict)
                else Value()
            )
        if self.termination is None:
            self.termination = (
                {k: False for k in self.observation}
                if isinstance(self.observation, dict)
                else False
            )

        if self.info is None:
            self.info = (
                {k: None for k in self.state} if isinstance(self.state, dict) else None
            )

    def _cast(self, src_sub: List[Tree], dst_sub: List[Tree]):
        return TransitionOutcome(
            cast(self.state, src_sub[0], dst_sub[0]),
            cast(self.value, src_sub[1], dst_sub[1]),
            cast(self.termination, src_sub[2], dst_sub[2]),
            cast(self.info, src_sub[3], dst_sub[3]),
        )


# Memory
class Memory(Deque[T]):
    pass


# StrDict
class StrDict(Generic[T], Dict[str, T]):
    """A dictionary with String keys (e.g. agent names)."""

    pass


# Constraint
class Constraint(Generic[D.T_memory, D.T_event, D.T_state], Castable):
    """A constraint."""

    def check(
        self,
        memory: D.T_memory,
        action: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> bool:
        """Check this constraint.

        !!! tip
            If this function never depends on the next_state parameter for its computation, it is recommended to
            indicate it by overriding #Constraint._is_constraint_dependent_on_next_state_() to return False. This
            information can then be exploited by solvers to avoid computing next state to evaluate the constraint (more
            efficient).

        # Parameters
        memory: The source memory (state or history) of the transition.
        action: The action taken in the given memory (state or history) triggering the transition.
        next_state: The next state in which the transition ends (if needed for the computation).

        # Returns
        True if the constraint is checked (False otherwise).
        """
        raise NotImplementedError

    @functools.lru_cache()
    def is_constraint_dependent_on_next_state(self) -> bool:
        """Indicate whether this constraint requires the next_state parameter for its computation (cached).

        By default, #Constraint.is_constraint_dependent_on_next_state() internally
        calls #Constraint._is_constraint_dependent_on_next_state_() the first time and automatically caches its value to
        make future calls more efficient (since the returned value is assumed to be constant).

        # Returns
        True if the constraint computation depends on next_state (False otherwise).
        """
        return self._is_constraint_dependent_on_next_state_()

    def _is_constraint_dependent_on_next_state_(self) -> bool:
        """Indicate whether this constraint requires the next_state parameter for its computation.

        This is a helper function called by default from #Constraint.is_constraint_dependent_on_next_state(), the
        difference being that the result is not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        True if the constraint computation depends on next_state (False otherwise).
        """
        raise NotImplementedError


class ImplicitConstraint(Constraint[D.T_memory, D.T_event, D.T_state]):
    """A constraint formalized implicitly, i.e. by a black-box check() function."""

    def __init__(
        self,
        check_function: Callable[[D.T_memory, D.T_event, Optional[D.T_state]], bool],
        depends_on_next_state: bool = True,
    ) -> None:
        """Initialize ImplicitConstraint.

        # Parameters
        check_function: The check() function to use.
        depends_on_next_state: Whether the check() function requires the next_state parameter for its computation.

        # Example
        ```python
        constraint = ImplicitConstraint(lambda memory, action, next_state: next_state.x % 2 == 0)
        ```
        """
        self._check_function = check_function
        self._depends_on_next_state = depends_on_next_state

    def check(
        self,
        memory: D.T_memory,
        action: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> bool:
        return self._check_function(memory, action, next_state)

    def _is_constraint_dependent_on_next_state_(self) -> bool:
        return self._depends_on_next_state

    def _cast(self, src_sub: List[Tree], dst_sub: List[Tree]):
        def cast_check_function(memory, action, next_state):
            cast_memory = cast(memory, dst_sub[0], src_sub[0])
            cast_action = cast(action, dst_sub[1], src_sub[1])
            cast_next_state = cast(next_state, dst_sub[2], src_sub[2])
            return self._check_function(cast_memory, cast_action, cast_next_state)

        return ImplicitConstraint(cast_check_function, self._depends_on_next_state)


class BoundConstraint(Constraint[D.T_memory, D.T_event, D.T_state]):
    """A constraint characterized by an evaluation function, an inequality and a bound.

    # Example
    A BoundConstraint with inequality '>=' is checked if (and only if) its #BoundConstraint.evaluate() function returns
    a float greater than or equal to its bound.
    """

    def __init__(
        self,
        evaluate_function: Callable[
            [D.T_memory, D.T_event, Optional[D.T_state]], float
        ],
        inequality: str,
        bound: float,
        depends_on_next_state: bool = True,
    ) -> None:
        """Initialize BoundConstraint.

        # Parameters
        evaluate_function: The evaluate() function to use.
        inequality: A string ('<', '<=', '>' or '>=') describing the constraint inequality.
        bound: The bound of the constraint.
        depends_on_next_state: Whether the evaluate() function requires the next_state parameter for its computation.

        # Example
        ```python
        constraint = BoundConstraint((lambda memory, action, next_state: next_state.x), '>', 5.)
        ```
        """
        self._evaluate_function = evaluate_function
        self._inequality = inequality
        self._bound = bound
        self._depends_on_next_state = depends_on_next_state

        assert inequality in ["<", "<=", ">", ">="]
        inequality_functions = {
            "<": (lambda val, bnd: val < bnd),
            "<=": (lambda val, bnd: val <= bnd),
            ">": (lambda val, bnd: val > bnd),
            ">=": (lambda val, bnd: val >= bnd),
        }
        self._check_function = inequality_functions[inequality]

    def check(
        self,
        memory: D.T_memory,
        action: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> bool:
        return self._check_function(
            self.evaluate(memory, action, next_state), self._bound
        )

    def _is_constraint_dependent_on_next_state_(self) -> bool:
        return self._depends_on_next_state

    def evaluate(
        self,
        memory: D.T_memory,
        action: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> float:
        """Evaluate the left side of this BoundConstraint.

        !!! tip
            If this function never depends on the next_state parameter for its computation, it is recommended to
            indicate it by overriding #Constraint._is_constraint_dependent_on_next_state_() to return False. This
            information can then be exploited by solvers to avoid computing next state to evaluate the constraint (more
            efficient).

        # Parameters
        memory: The source memory (state or history) of the transition.
        action: The action taken in the given memory (state or history) triggering the transition.
        next_state: The next state in which the transition ends (if needed for the computation).

        # Returns
        The float value resulting from the evaluation.
        """
        return self._evaluate_function(memory, action, next_state)

    def get_inequality(self) -> str:
        """Get the string ('<', '<=', '>' or '>=') describing the constraint inequality.

        # Returns
        The string describing the inequality.
        """
        return self._inequality

    def get_bound(self) -> float:
        """Get the bound of the constraint.

        # Returns
        The constraint bound.
        """
        return self._bound

    def _cast(self, src_sub: List[Tree], dst_sub: List[Tree]):
        def cast_evaluate_function(memory, action, next_state):
            cast_memory = cast(memory, dst_sub[0], src_sub[0])
            cast_action = cast(action, dst_sub[1], src_sub[1])
            cast_next_state = cast(next_state, dst_sub[2], src_sub[2])
            return self._evaluate_function(cast_memory, cast_action, cast_next_state)

        return BoundConstraint(
            cast_evaluate_function,
            self._inequality,
            self._bound,
            self._depends_on_next_state,
        )


SINGLE_AGENT_ID = "agent"

# (auto)cast-related objects/functions
cast_dict = {
    (Memory, Union): lambda obj, src, dst: cast(obj[0], src[0], dst[0]),
    (Union, Memory): lambda obj, src, dst: Memory([cast(obj, src[0], dst[0])]),
    (Memory, Memory): lambda obj, src, dst: Memory(
        [cast(x, src[0], dst[0]) for x in obj]
    ),
    (StrDict, Union): lambda obj, src, dst: cast(
        next(iter(obj.values())), src[0], dst[0]
    ),
    (Union, StrDict): lambda obj, src, dst: {
        SINGLE_AGENT_ID: cast(obj, src[0], dst[0])
    },
    (StrDict, StrDict): lambda obj, src, dst: {
        k: cast(v, src[0], dst[0]) for k, v in obj.items()
    },
    # (Set, Union): lambda obj, src, dst: cast(next(iter(obj)), src[0], dst[0]),
    # (Union, Set): lambda obj, src, dst: {cast(obj, src[0], dst[0])},
    # (Set, Set): lambda obj, src, dst: {cast(x, src[0], dst[0]) for x in obj},
    (List, Union): lambda obj, src, dst: cast(obj[0], src[0], dst[0]),
    (Union, List): lambda obj, src, dst: [cast(obj, src[0], dst[0])],
    (List, List): lambda obj, src, dst: [cast(x, src[0], dst[0]) for x in obj],
    (Union, Union): lambda obj, src, dst: cast(obj, src[0], dst[0]),
    (Optional, Optional): lambda obj, src, dst: cast(obj, src[0], dst[0])
    if obj is not None
    else None,
}  # (src_type, dst_type): (obj: src_type, src_sub_hintrees: List[Tree], dst_sub_hintrees: List[Tree]) -> dst_type

default_cast = (
    lambda obj, src, dst: obj._cast(src, dst) if isinstance(obj, Castable) else obj
)


def parse_hint(obj: object, hint: str, hint_obj: str) -> Tree:
    # note: hint is assumed to contain no whitespace (which will be true by construction)
    match = re.match(
        rf"^(?P<hint_obj>{hint_obj}\.)?(?P<type>\w+)(?:\[(?P<generics>.+)\])?$", hint
    )
    groups = match.groupdict()
    type_ = getattr(obj, groups["type"]) if groups["hint_obj"] else eval(groups["type"])
    if groups["generics"]:
        generics = groups["generics"].split(",")
        if all(
            g.count("[") == g.count("]") for g in generics
        ):  # check that the generics split is not at a sub-level
            sub = [parse_hint(obj, h, hint_obj) for h in generics]
        else:
            sub = [parse_hint(obj, groups["generics"], hint_obj)]
    else:
        sub = []
    return Tree(type_, sub)


def get_args_dict(func: Callable, args: Tuple, kwargs: Dict) -> Dict:
    while hasattr(
        func, "__wrapped__"
    ):  # get to the core function even if it was decorated
        func = func.__wrapped__
    args_names = func.__code__.co_varnames[
        inspect.ismethod(func) : func.__code__.co_argcount
    ]
    return {**dict(zip(args_names, args)), **kwargs}


@functools.lru_cache(maxsize=1000)
def cast_needed(src_hintree: Tree, dst_hintree: Tree) -> bool:
    # note: src_hintree and dst_hintree are assumed to have the same tree structure (which will be true by construction)
    src_type = src_hintree.type
    dst_type = dst_hintree.type
    if src_type != dst_type and (src_type, dst_type) in cast_dict:
        return True
    else:
        src_sub_hintrees = src_hintree.sub
        dst_sub_hintrees = dst_hintree.sub
        return any(
            cast_needed(src_sub_hintrees[i], dst_sub_hintrees[i])
            for i in range(len(src_sub_hintrees))
        )


def cast(obj: object, src_hintree: Tree, dst_hintree: Tree):
    # print('> Cast', obj, src_hintree.type, dst_hintree.type)
    if cast_needed(src_hintree, dst_hintree):
        cast_pair = (src_hintree.type, dst_hintree.type)
        return cast_dict.get(cast_pair, default_cast)(
            obj, src_hintree.sub, dst_hintree.sub
        )
    else:
        return obj


def autocast(func: Callable, src: object, dst: object, hint_obj: str = "D") -> Callable:
    hints = func.__annotations__
    cast_args = []
    src_hintrees = {}
    dst_hintrees = {}
    for arg, hint in hints.items():
        # if hint depends on a hint_obj attribute at least once (e.g. 'D.T_state')
        if re.search(rf"(?<=\b{hint_obj}\.)\w+", hint):
            formatted_hint = re.sub(r"\s+", "", hint)
            src_hintrees[arg] = parse_hint(src, formatted_hint, hint_obj)
            dst_hintrees[arg] = parse_hint(dst, formatted_hint, hint_obj)
            if cast_needed(src_hintrees[arg], dst_hintrees[arg]):
                cast_args.append(arg)

    @functools.wraps(func)
    def wrapper_autocast(
        *args, **kwargs
    ):  # TODO: raise autocast exception when problem inside wrapper?
        # print('Wrapper used for:', func, src_hintrees, dst_hintrees)
        args_dict = get_args_dict(func, args, kwargs)
        cast_args_dict = {
            k: (cast(v, dst_hintrees[k], src_hintrees[k]) if k in cast_args else v)
            for k, v in args_dict.items()
        }
        result = func(**cast_args_dict)
        return (
            cast(result, src_hintrees["return"], dst_hintrees["return"])
            if "return" in cast_args
            else result
        )

    return wrapper_autocast if len(cast_args) > 0 else func


def autocast_all(obj: object, src: object, dst: object):
    for name, f in inspect.getmembers(
        obj, lambda x: inspect.isfunction(x) or inspect.ismethod(x)
    ):
        if getattr(f, "_autocastable", None):
            setattr(obj, name, autocast(f, src, dst))


# autocastable (function decorator)
def autocastable(func: Callable):
    func._autocastable = True
    return func


# nocopy (class decorator)
def nocopy(cls):
    cls.__copy__ = lambda self: self
    cls.__deepcopy__ = lambda self, memodict={}: self
    return cls
