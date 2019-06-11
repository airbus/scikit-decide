"""This module contains template classes for quickly building domains."""

from airlaps.builders.domain.dynamics import EnvironmentDomain, SimulationDomain, EnumerableTransitionDomain, \
    DeterministicTransitionDomain
from airlaps.builders.domain.events import EventDomain, ActionDomain
from airlaps.builders.domain.goals import GoalDomain
from airlaps.builders.domain.initialization import InitializableDomain, UncertainInitializedDomain, \
    DeterministicInitializedDomain
from airlaps.builders.domain.memory import HistoryDomain, MarkovianDomain
from airlaps.builders.domain.observability import PartiallyObservableDomain, TransformedObservableDomain, \
    FullyObservableDomain
from airlaps.builders.domain.value import RewardDomain, PositiveCostDomain

__all__ = ['Domain', 'ResettableDomain', 'RLDomain', 'StatelessSimulatorDomain', 'MDP', 'POMDP', 'GoalMDP', 'GoalPOMDP',
           'DeterministicPlanningDomain']


# MAIN TEMPLATE CLASS

class Domain(EnvironmentDomain, EventDomain, HistoryDomain, PartiallyObservableDomain, RewardDomain):
    """This is the highest level domain class (inheriting top-level class for each mandatory domain characteristic).

    This helper class can be used as the main template class for domains.

    Typical use:
    ```python
    class MyDomain(Domain, ...)
    ```

    with "..." replaced when needed by a number of classes from following domain characteristics (the ones in
    parentheses are optional):

    - **(constraints)**: ConstrainedDomain
    - **dynamics**: EnvironmentDomain -> SimulationDomain -> UncertainTransitionDomain -> EnumerableTransitionDomain
      -> DeterministicTransitionDomain
    - **events**: EventDomain -> ActionDomain -> UnrestrictedActionDomain
    - **(goals)**: GoalDomain
    - **(initialization)**: InitializableDomain -> UncertainInitializedDomain -> DeterministicInitializedDomain
    - **memory**: HistoryDomain -> FiniteHistoryDomain -> MarkovianDomain -> MemorylessDomain
    - **observability**: PartiallyObservableDomain -> TransformedObservableDomain -> FullyObservableDomain
    - **(renderability)**: RenderableDomain
    - **value**: RewardDomain -> PositiveCostDomain
    """
    pass


# ALTERNATE TEMPLATE CLASSES (for typical combinations)

class ResettableDomain(Domain, EnvironmentDomain, EventDomain, InitializableDomain, HistoryDomain,
                       PartiallyObservableDomain, RewardDomain):
    """This is a typical resettable domain class.

    This helper class can be used as an alternate template class for domains, inheriting the following:

    - Domain
    - EnvironmentDomain
    - EventDomain
    - InitializableDomain
    - HistoryDomain
    - PartiallyObservableDomain
    - RewardDomain

    Typical use:
    ```python
    class MyDomain(ResettableDomain)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class MyDomain(RLDomain, FullyObservableDomain)
        ```
    """
    pass


class RLDomain(Domain, EnvironmentDomain, ActionDomain, InitializableDomain, MarkovianDomain,
               TransformedObservableDomain, RewardDomain):
    """This is a typical Reinforcement Learning domain class.

    This helper class can be used as an alternate template class for domains, inheriting the following:

    - Domain
    - EnvironmentDomain
    - ActionDomain
    - InitializableDomain
    - MarkovianDomain
    - TransformedObservableDomain
    - RewardDomain

    Typical use:
    ```python
    class MyDomain(RLDomain)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class MyDomain(RLDomain, FullyObservableDomain)
        ```
    """
    pass


class StatelessSimulatorDomain(Domain, SimulationDomain, EventDomain, MarkovianDomain, TransformedObservableDomain,
                               RewardDomain):
    """This is a typical stateless simulator domain class.

    This helper class can be used as an alternate template class for domains, inheriting the following:

    - Domain
    - SimulationDomain
    - EventDomain
    - MarkovianDomain
    - TransformedObservableDomain
    - RewardDomain

    Typical use:
    ```python
    class MyDomain(StatelessSimulatorDomain)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class MyDomain(RLDomain, FullyObservableDomain)
        ```
    """
    pass


class MDP(Domain, EnumerableTransitionDomain, ActionDomain, DeterministicInitializedDomain, MarkovianDomain,
          FullyObservableDomain, RewardDomain):
    """This is a typical Markov Decision Process domain class.

    This helper class can be used as an alternate template class for domains, inheriting the following:

    - Domain
    - EnumerableTransitionDomain
    - ActionDomain
    - DeterministicInitializedDomain
    - MarkovianDomain
    - FullyObservableDomain
    - RewardDomain

    Typical use:
    ```python
    class MyDomain(MDP)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class MyDomain(RLDomain, FullyObservableDomain)
        ```
    """
    pass


class POMDP(Domain, EnumerableTransitionDomain, ActionDomain, UncertainInitializedDomain, MarkovianDomain,
            PartiallyObservableDomain, RewardDomain):
    """This is a typical Partially Observable Markov Decision Process domain class.

    This helper class can be used as an alternate template class for domains, inheriting the following:

    - Domain
    - EnumerableTransitionDomain
    - ActionDomain
    - UncertainInitializedDomain
    - MarkovianDomain
    - PartiallyObservableDomain
    - RewardDomain

    Typical use:
    ```python
    class MyDomain(POMDP)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class MyDomain(RLDomain, FullyObservableDomain)
        ```
    """
    pass


class GoalMDP(Domain, EnumerableTransitionDomain, ActionDomain, GoalDomain, DeterministicInitializedDomain,
              MarkovianDomain, FullyObservableDomain, PositiveCostDomain):
    """This is a typical Goal Markov Decision Process domain class.

    This helper class can be used as an alternate template class for domains, inheriting the following:

    - Domain
    - EnumerableTransitionDomain
    - ActionDomain
    - GoalDomain
    - DeterministicInitializedDomain
    - MarkovianDomain
    - FullyObservableDomain
    - PositiveCostDomain

    Typical use:
    ```python
    class MyDomain(GoalMDP)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class MyDomain(RLDomain, FullyObservableDomain)
        ```
    """
    pass


class GoalPOMDP(Domain, EnumerableTransitionDomain, ActionDomain, GoalDomain, UncertainInitializedDomain,
                MarkovianDomain, PartiallyObservableDomain, PositiveCostDomain):
    """This is a typical Goal Partially Observable Markov Decision Process domain class.

    This helper class can be used as an alternate template class for domains, inheriting the following:

    - Domain
    - EnumerableTransitionDomain
    - ActionDomain
    - GoalDomain
    - UncertainInitializedDomain
    - MarkovianDomain
    - PartiallyObservableDomain
    - PositiveCostDomain

    Typical use:
    ```python
    class MyDomain(GoalPOMDP)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class MyDomain(RLDomain, FullyObservableDomain)
        ```
    """
    pass


class DeterministicPlanningDomain(Domain, DeterministicTransitionDomain, ActionDomain, GoalDomain,
                                  DeterministicInitializedDomain, MarkovianDomain, FullyObservableDomain,
                                  PositiveCostDomain):
    """This is a typical deterministic planning domain class.

    This helper class can be used as an alternate template class for domains, inheriting the following:

    - Domain
    - DeterministicTransitionDomain
    - ActionDomain
    - GoalDomain
    - DeterministicInitializedDomain
    - MarkovianDomain
    - FullyObservableDomain
    - PositiveCostDomain

    Typical use:
    ```python
    class MyDomain(DeterministicPlanningDomain)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class MyDomain(RLDomain, FullyObservableDomain)
        ```
    """
    pass
