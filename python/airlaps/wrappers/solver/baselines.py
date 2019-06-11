from typing import Optional, Callable, Any, Iterable

from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

from airlaps import Memory, T_observation, T_event, Domain
from airlaps.builders.domain.dynamics import EnvironmentDomain
from airlaps.builders.domain.events import UnrestrictedActionDomain
from airlaps.builders.domain.initialization import InitializableDomain
from airlaps.builders.solver.domain import DomainSolver
from airlaps.builders.solver.policy import PolicySolver
from airlaps.builders.solver.restorability import RestorableSolver
from airlaps.builders.solver.temporality import SolutionSolver
from airlaps.wrappers.domain.gym import AsGymEnv
from airlaps.wrappers.space.gym import GymSpace


class BaselinesSolver(DomainSolver, PolicySolver, RestorableSolver, SolutionSolver):
    """This class wraps a stable OpenAI Baselines solver (stable_baselines) as an AIRLAPS domain.

    !!! warning
        Using this class requires Stable Baselines to be installed.
    """

    def __init__(self, algo_class: type, baselines_policy: Any, **kwargs: Any) -> None:
        """Initialize BaselinesSolver.

        # Parameters
        algo_class: The class of Baselines solver (stable_baselines) to wrap.
        baselines_policy: The class of Baselines policy network (stable_baselines.common.policies) to use.
        """
        self._algo_class = algo_class
        self._baselines_policy = baselines_policy
        self._algo_kwargs = kwargs

    def _reset(self) -> None:
        # TODO: improve code for parallelism
        #  (https://stable-baselines.readthedocs.io/en/master/guide/examples.html
        #  #multiprocessing-unleashing-the-power-of-vectorized-environments)?
        domain = self._new_domain()
        env = Monitor(AsGymEnv(domain), filename=None, allow_early_resets=True)
        env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
        self._algo = self._algo_class(self._baselines_policy, env, **self._algo_kwargs)
        self._wrap_action = lambda a: next(iter(domain.get_action_space().from_unwrapped([a])))

    def get_domain_requirements(self) -> Iterable[type]:
        return [EnvironmentDomain, UnrestrictedActionDomain, InitializableDomain]

    def _check_domain(self, domain: Domain) -> bool:
        return isinstance(domain.get_action_space(), GymSpace) and isinstance(domain.get_observation_space(), GymSpace)

    def sample_action(self, memory: Memory[T_observation]) -> T_event:
        action, _ = self._algo.predict(memory[-1])
        return self._wrap_action(action)

    def is_policy_defined_for(self, memory: Memory[T_observation]) -> bool:
        return True

    def save(self, path: str) -> None:
        self._algo.save(path)

    def restore(self, path: str) -> None:
        self._algo = self._algo_class.load(path)

    def solve(self, from_observation: Optional[Memory[T_observation]] = None,
              on_update: Optional[Callable[[], bool]] = None, max_time: Optional[float] = None, **kwargs: Any) -> None:
        assert from_observation is None  # TODO: should from_observation really be in (every) solve signature?
        assert max_time is None  # TODO: should max_time really be in (every) solve signature?
        assert 'total_timesteps' in kwargs  # This argument is required by the algortihms
        self._algo.learn(callback=on_update, **kwargs)
