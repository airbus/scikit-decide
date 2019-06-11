from typing import Optional, Callable, Any, Iterable

from airlaps import T_observation, T_event, Distribution, ImplicitDistribution, Memory, Domain
from airlaps.builders.domain import EnvironmentDomain, EventDomain, HistoryDomain, PartiallyObservableDomain, \
    RewardDomain
from airlaps.builders.solver import DomainSolver, UncertainPolicySolver, SolutionSolver


class RandomWalk(DomainSolver, UncertainPolicySolver, SolutionSolver):

    def _reset(self) -> None:
        self._domain = self._new_domain()

    def get_domain_requirements(self) -> Iterable[type]:
        return [EnvironmentDomain, EventDomain, HistoryDomain, PartiallyObservableDomain, RewardDomain]

    def _check_domain(self, domain: Domain) -> bool:
        return True

    def get_next_action_distribution(self, memory: Memory[T_observation]) -> Distribution[T_event]:
        return ImplicitDistribution(lambda: self._domain.get_applicable_actions(memory).sample())

    def is_policy_defined_for(self, memory: Memory[T_observation]) -> bool:
        return True

    def solve(self, from_observation: Optional[Memory[T_observation]] = None,
              on_update: Optional[Callable[..., bool]] = None, max_time: Optional[float] = None, **kwargs: Any) -> None:
        pass
