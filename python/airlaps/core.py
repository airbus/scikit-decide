import random
import functools
from collections import deque
from typing import TypeVar, Generic, Optional, Tuple, Iterable, Sequence, MutableSequence, Deque, Callable

from airlaps.dataclasses import dataclass, \
    field  # TODO: replace 'airlaps.dataclasses' by 'dataclasses' once transitioned to Python 3.7

__all__ = ['T', 'T_state', 'T_observation', 'T_event', 'T_value', 'T_info', 'Space', 'ImplicitSpace', 'EnumerableSpace',
           'SamplableSpace', 'SerializableSpace', 'Distribution', 'ImplicitDistribution', 'DiscreteDistribution',
           'SingleValueDistribution', 'TransitionValue', 'EnvironmentOutcome', 'TransitionOutcome', 'Memory',
           'Constraint', 'BoundConstraint', 'nocopy']

T = TypeVar('T')  # Any type
T_state = TypeVar('T_state')  # Type of states
T_observation = TypeVar('T_observation')  # Type of observations
T_event = TypeVar('T_event')  # Type of events
T_value = TypeVar('T_value')  # Type of transition values (rewards or costs)
T_info = TypeVar('T_info')  # Type of additional information given as part of an environment outcome


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
        """Initialize ImplicitSpace

        # Parameters
        contains_function: The contains() function to use.

        # Example
        ```python
        my_space = ImplicitSpace(lambda x: 5 < x['position'] < 10)
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


class Distribution(Generic[T]):
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


class DiscreteDistribution(Distribution[T]):
    """A discrete probability distribution."""

    def __init__(self, values: Iterable[Tuple[T, float]]) -> None:
        """Initialize DiscreteDistribution.

        !!! note
            If the given probabilities do not sum to 1, they are implicitly normalized as such for sampling.

        # Parameters
        values: The iterable (e.g. list) of (element, probability) pairs.

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

    def get_values(self) -> Iterable[Tuple[T, float]]:
        """Get the iterable (e.g. list) of (element, probability) pairs.

        # Returns
        The (element, probability) pairs.
        """
        return self._values


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

    def get_values(self) -> Iterable[Tuple[T, float]]:
        return [(self._value, 1.0)]

    def get_value(self) -> T:
        """Get the single value of this distribution.

        # Returns
        The single value of this distribution.
        """
        return self._value


@dataclass
class TransitionValue(Generic[T_value]):
    """A transition value (reward or cost).

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
    value_1 = TransitionValue(reward=-5)
    value_2 = TransitionValue(cost=5)

    assert value_1.reward == value_2.reward == -5  # True
    assert value_1.cost == value_2.cost == 5  # True
    ```
    """
    reward: Optional[T_value] = None
    cost: Optional[T_value] = None

    def __post_init__(self) -> None:
        if self.reward is not None:
            self.cost = self._reward_to_cost(self.reward)
        elif self.cost is not None:
            self.reward = self._cost_to_reward(self.cost)
        else:
            self.reward = 0
            self.cost = 0

    def _cost_to_reward(self, cost: T_value) -> T_value:
        return -cost

    def _reward_to_cost(self, reward: T_value) -> T_value:
        return -reward


@dataclass
class EnvironmentOutcome(Generic[T_observation, T_value, T_info]):
    """An environment outcome for an internal transition.

    # Parameters
    observation: The agent's observation of the current environment.
    value: The value (reward or cost) returned after previous action.
    termination: Whether the episode has ended, in which case further step() calls will return undefined results.
    info: Optional auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    """
    observation: T_observation
    value: TransitionValue[T_value] = field(default_factory=TransitionValue)
    termination: bool = False
    info: Optional[T_info] = None


@dataclass
class TransitionOutcome(Generic[T_state, T_value, T_info]):
    """A transition outcome.

    # Parameters
    state: The new state after the transition.
    value: The value (reward or cost) returned after previous action.
    termination: Whether the episode has ended, in which case further step() calls will return undefined results.
    info: Optional auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    """
    state: T_state
    value: TransitionValue[T_value] = field(default_factory=TransitionValue)
    termination: bool = False
    info: Optional[T_info] = None


class Memory(deque, Generic[T]):
    """A memory that acts as a finite or infinite buffer.

    !!! tip
        Set maxlen in the constructor to a fixed value to define a finite memory (if maxlen is omitted or None, an
        infinite memory is created).

    # Example
    ```python
    # Create a memory of last 3 elements
    memory = Memory(maxlen=3)


    for i in range(10):
        memory.append(i)

    print(memory)  # prints: Memory([7, 8, 9], maxlen=3)
    print(memory[-1])  # prints last element: 9
    ```
    """

    def __repr__(self) -> str:
        return super().__repr__().replace('deque', self.__class__.__name__)

    def __init__(self, iterable: Iterable[T] = (), maxlen: int = None) -> None:
        super().__init__(iterable, maxlen)

    def __setitem__(self, i: int, x: T) -> None:
        super().__setitem__(i, x)

    def __add__(self, other: Deque[T]) -> Deque[T]:
        return super().__add__(other)

    def __iadd__(self, x: Iterable[T]) -> MutableSequence[T]:
        return super().__iadd__(x)

    def append(self, x: T) -> None:
        super().append(x)

    def appendleft(self, x: T) -> None:
        super().appendleft(x)

    def count(self, x: T) -> int:
        return super().count(x)

    def extend(self, iterable: Iterable[T]) -> None:
        super().extend(iterable)

    def extendleft(self, iterable: Iterable[T]) -> None:
        super().extendleft(iterable)

    def insert(self, i: int, x: T) -> None:
        super().insert(i, x)

    def index(self, x: T, start: int = None, stop: int = None) -> int:
        return super().index(x, start, stop)

    def remove(self, value: T) -> None:
        super().remove(value)


class Constraint(Generic[T_state, T_event]):
    """A constraint."""

    def __init__(self, check_function: Callable[[Memory[T_state], T_event, Optional[T_state]], bool],
                 depends_on_next_state: bool = True) -> None:
        """Initialize Constraint.

        # Parameters
        check_function: The check() function to use.
        depends_on_next_state: Whether the check() function requires the next_state parameter for its computation.

        # Example
        ```python
        constraint = Constraint(lambda memory, event, next_state: next_state.x % 2 == 0)
        ```
        """
        self._check_function = check_function
        self._depends_on_next_state = depends_on_next_state

    def check(self, memory: Memory[T_state], event: T_event, next_state: Optional[T_state] = None) -> bool:
        """Check this constraint.

        !!! tip
            If this function never depends on the next_state parameter for its computation, it is recommended to
            indicate it by overriding #Constraint._is_constraint_dependent_on_next_state_() to return False. This
            information can then be exploited by solvers to avoid computing next state to evaluate the constraint (more
            efficient).

        # Parameters
        memory: The source memory (state or history) of the transition.
        event: The event taken in the given memory (state or history) triggering the transition.
        next_state: The next state in which the transition ends (if needed for the computation).

        # Returns
        True if the constraint is checked (False otherwise).
        """
        return self._check_function(memory, event, next_state)

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
        return self._depends_on_next_state


class BoundConstraint(Constraint[T_state, T_event]):
    """A constraint characterized by an evaluation function, an inequality and a bound.

    # Example
    A BoundConstraint with inequality '>=' is checked if (and only if) its #BoundConstraint.evaluate() function returns
    a float greater than or equal to its bound.
    """

    def __init__(self, evaluate_function: Callable[[Memory[T_state], T_event, Optional[T_state]], float],
                 inequality: str, bound: float, depends_on_next_state: bool = True) -> None:
        """Initialize BoundConstraint.

        # Parameters
        evaluate_function: The evaluate() function to use.
        inequality: A string ('<', '<=', '>' or '>=') describing the constraint inequality.
        bound: The bound of the constraint.
        depends_on_next_state: Whether the evaluate() function requires the next_state parameter for its computation.

        # Example
        ```python
        constraint = BoundConstraint((lambda memory, event, next_state: next_state.x), '<', 5.)
        ```
        """
        self._evaluate_function = evaluate_function
        self._inequality = inequality
        self._bound = bound
        self._depends_on_next_state = depends_on_next_state

        assert inequality in ['<', '<=', '>', '>=']
        inequality_functions = {'<': (lambda val, bnd: val < bnd), '<=': (lambda val, bnd: val <= bnd),
                                '>': (lambda val, bnd: val > bnd), '>=': (lambda val, bnd: val >= bnd)}
        self._check_function = inequality_functions[inequality]

    def check(self, memory: Memory[T_state], event: T_event, next_state: Optional[T_state] = None) -> bool:
        return self._check_function(self.evaluate(memory, event, next_state), self._bound)

    def evaluate(self, memory: Memory[T_state], event: T_event, next_state: Optional[T_state] = None) -> float:
        """Evaluate the left side of this BoundConstraint.

        !!! tip
            If this function never depends on the next_state parameter for its computation, it is recommended to
            indicate it by overriding #Constraint._is_constraint_dependent_on_next_state_() to return False. This
            information can then be exploited by solvers to avoid computing next state to evaluate the constraint (more
            efficient).

        # Parameters
        memory: The source memory (state or history) of the transition.
        event: The event taken in the given memory (state or history) triggering the transition.
        next_state: The next state in which the transition ends (if needed for the computation).

        # Returns
        The float value resulting from the evaluation.
        """
        return self._evaluate_function(memory, event, next_state)

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


def nocopy(cls):
    cls.__copy__ = lambda self: self
    cls.__deepcopy__ = lambda self, memodict={}: self
    return cls
