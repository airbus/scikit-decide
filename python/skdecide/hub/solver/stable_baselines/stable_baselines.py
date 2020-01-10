# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable, Any, Dict

from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

from skdecide import Domain, Solver
from skdecide.builders.domain import SingleAgent, Sequential, UnrestrictedActions, Initializable
from skdecide.builders.solver import Policies, Restorable
from skdecide.hub.domain.gym import AsGymEnv
from skdecide.hub.space.gym import GymSpace


class D(Domain, SingleAgent, Sequential, UnrestrictedActions, Initializable):
    pass


class StableBaseline(Solver, Policies, Restorable):
    """This class wraps a stable OpenAI Baselines solver (stable_baselines) as a scikit-decide solver.

    !!! warning
        Using this class requires Stable Baselines to be installed.
    """
    T_domain = D

    def __init__(self, algo_class: type, baselines_policy: Any, learn_config: Dict = None, **kwargs: Any) -> None:
        """Initialize StableBaselines.

        # Parameters
        algo_class: The class of Baselines solver (stable_baselines) to wrap.
        baselines_policy: The class of Baselines policy network (stable_baselines.common.policies) to use.
        """
        self._algo_class = algo_class
        self._baselines_policy = baselines_policy
        self._learn_config = learn_config if learn_config is not None else {}
        self._algo_kwargs = kwargs

    @classmethod
    def _check_domain_additional(cls, domain: Domain) -> bool:
        return isinstance(domain.get_action_space(), GymSpace) and isinstance(domain.get_observation_space(), GymSpace)

    def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
        # TODO: improve code for parallelism
        #  (https://stable-baselines.readthedocs.io/en/master/guide/examples.html
        #  #multiprocessing-unleashing-the-power-of-vectorized-environments)?
        if not hasattr(self, '_algo'):  # reuse algo if possible (enables further learning)
            domain = domain_factory()
            env = Monitor(AsGymEnv(domain), filename=None, allow_early_resets=True)
            env = DummyVecEnv([lambda: env])  # the algorithms require a vectorized environment to run
            self._algo = self._algo_class(self._baselines_policy, env, **self._algo_kwargs)
            self._init_algo(domain)
        self._algo.learn(**self._learn_config)

    def _sample_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        action, _ = self._algo.predict(observation)
        return self._wrap_action(action)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _save(self, path: str) -> None:
        self._algo.save(path)

    def _load(self, path: str, domain_factory: Callable[[], D]):
        self._algo = self._algo_class.load(path)
        self._init_algo(domain_factory())

    def _init_algo(self, domain: D):
        self._wrap_action = lambda a: next(iter(domain.get_action_space().from_unwrapped([a])))
