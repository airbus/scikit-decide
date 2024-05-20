# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module contains base classes for quickly building domains."""
from __future__ import annotations

import logging
from typing import Callable, NewType, Optional

from skdecide.builders.domain.agent import MultiAgent, SingleAgent
from skdecide.builders.domain.concurrency import Parallel, Sequential
from skdecide.builders.domain.dynamics import (
    DeterministicTransitions,
    EnumerableTransitions,
    Environment,
    Simulation,
)
from skdecide.builders.domain.events import Actions, Events
from skdecide.builders.domain.goals import Goals
from skdecide.builders.domain.initialization import (
    DeterministicInitialized,
    Initializable,
    UncertainInitialized,
)
from skdecide.builders.domain.memory import History, Markovian
from skdecide.builders.domain.observability import (
    FullyObservable,
    PartiallyObservable,
    TransformedObservable,
)
from skdecide.builders.domain.value import PositiveCosts, Rewards
from skdecide.builders.solver.fromanystatesolvability import FromAnyState
from skdecide.core import D, autocast_all

if (
    False
):  # trick to avoid circular import & IDE error ("Unresolved reference 'Solver'")
    from skdecide.solvers import Solver

__all__ = [
    "Domain",
    "RLDomain",
    "MultiAgentRLDomain",
    "StatelessSimulatorDomain",
    "MDPDomain",
    "POMDPDomain",
    "GoalMDPDomain",
    "GoalPOMDPDomain",
    "DeterministicPlanningDomain",
]

logger = logging.getLogger("skdecide.domains")

logger.setLevel(logging.INFO)

if not len(logger.handlers):
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
    logger.propagate = False


# MAIN BASE CLASS

T_state = NewType("T_state", object)
T_observation = NewType("T_observation", object)
T_event = NewType("T_event", object)
T_value = NewType("T_value", object)
T_predicate = NewType("T_predicate", object)
T_info = NewType("T_info", object)


class Domain(
    MultiAgent, Parallel, Environment, Events, History, PartiallyObservable, Rewards
):
    """This is the highest level domain class (inheriting top-level class for each mandatory domain characteristic).

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
    """

    T_state = T_state
    T_observation = T_observation
    T_event = T_event
    T_value = T_value
    T_predicate = T_predicate
    T_info = T_info


# ALTERNATE BASE CLASSES (for typical combinations)


class RLDomain(
    Domain,
    SingleAgent,
    Sequential,
    Environment,
    Actions,
    Initializable,
    Markovian,
    TransformedObservable,
    Rewards,
):
    """This is a typical Reinforcement Learning domain class.

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

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """

    pass


class MultiAgentRLDomain(
    Domain,
    MultiAgent,
    Sequential,
    Environment,
    Actions,
    Initializable,
    Markovian,
    TransformedObservable,
    Rewards,
):
    """This is a typical multi-agent Reinforcement Learning domain class.

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

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """

    pass


class StatelessSimulatorDomain(
    Domain,
    SingleAgent,
    Sequential,
    Simulation,
    Actions,
    Markovian,
    TransformedObservable,
    Rewards,
):
    """This is a typical stateless simulator domain class.

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

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """

    pass


class MDPDomain(
    Domain,
    SingleAgent,
    Sequential,
    EnumerableTransitions,
    Actions,
    DeterministicInitialized,
    Markovian,
    FullyObservable,
    Rewards,
):
    """This is a typical Markov Decision Process domain class.

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

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """

    pass


class POMDPDomain(
    Domain,
    SingleAgent,
    Sequential,
    EnumerableTransitions,
    Actions,
    UncertainInitialized,
    Markovian,
    PartiallyObservable,
    Rewards,
):
    """This is a typical Partially Observable Markov Decision Process domain class.

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

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """

    pass


class GoalMDPDomain(
    Domain,
    SingleAgent,
    Sequential,
    EnumerableTransitions,
    Actions,
    Goals,
    DeterministicInitialized,
    Markovian,
    FullyObservable,
    PositiveCosts,
):
    """This is a typical Goal Markov Decision Process domain class.

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

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """

    pass


class GoalPOMDPDomain(
    Domain,
    SingleAgent,
    Sequential,
    EnumerableTransitions,
    Actions,
    Goals,
    UncertainInitialized,
    Markovian,
    PartiallyObservable,
    PositiveCosts,
):
    """This is a typical Goal Partially Observable Markov Decision Process domain class.

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

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """

    pass


class DeterministicPlanningDomain(
    Domain,
    SingleAgent,
    Sequential,
    DeterministicTransitions,
    Actions,
    Goals,
    DeterministicInitialized,
    Markovian,
    FullyObservable,
    PositiveCosts,
):
    """This is a typical deterministic planning domain class.

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

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """

    pass
