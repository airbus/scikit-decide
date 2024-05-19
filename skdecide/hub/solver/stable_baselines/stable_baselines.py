# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, Union

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, ConvertCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from skdecide import Domain, Solver
from skdecide.builders.domain import (
    Initializable,
    Sequential,
    SingleAgent,
    UnrestrictedActions,
)
from skdecide.builders.solver import Policies, Restorable
from skdecide.hub.domain.gym import AsGymnasiumEnv
from skdecide.hub.space.gym import GymSpace


class D(Domain, SingleAgent, Sequential, UnrestrictedActions, Initializable):
    pass


class StableBaseline(Solver, Policies, Restorable):
    """This class wraps a stable OpenAI Baselines solver (stable_baselines3) as a scikit-decide solver.

    !!! warning
        Using this class requires Stable Baselines 3 to be installed.
    """

    T_domain = D

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        algo_class: Type[BaseAlgorithm],
        baselines_policy: Union[str, Type[BasePolicy]],
        learn_config: Optional[Dict[str, Any]] = None,
        callback: Callable[[StableBaseline], bool] = lambda solver: False,
        **kwargs: Any,
    ) -> None:
        """Initialize StableBaselines.

        # Parameters
        domain_factory: A callable with no argument returning the domain to solve (can be a mere domain class).
            The resulting domain will be auto-cast to the level expected by the solver.
        algo_class: The class of Baselines solver (stable_baselines3) to wrap.
        baselines_policy: The class of Baselines policy network (stable_baselines3.common.policies or str) to use.
        learn_config: the kwargs passed to sb3 algo's `learn()` method
        callback: function called at each solver iteration. If returning true, the solve process stops.

        """
        Solver.__init__(self, domain_factory=domain_factory)
        self._algo_class = algo_class
        self._baselines_policy = baselines_policy
        self._learn_config = learn_config if learn_config is not None else {}
        self._algo_kwargs = kwargs
        self.callback = callback

    @classmethod
    def _check_domain_additional(cls, domain: Domain) -> bool:
        return isinstance(domain.get_action_space(), GymSpace) and isinstance(
            domain.get_observation_space(), GymSpace
        )

    def _solve(self) -> None:
        # TODO: improve code for parallelism
        #  (https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
        #  #multiprocessing-unleashing-the-power-of-vectorized-environments)?
        if not hasattr(
            self, "_algo"
        ):  # reuse algo if possible (enables further learning)
            domain = self._domain_factory()
            env = DummyVecEnv(
                [lambda: AsGymnasiumEnv(domain)]
            )  # the algorithms require a vectorized environment to run
            self._algo = self._algo_class(
                self._baselines_policy, env, **self._algo_kwargs
            )
            self._init_algo(domain)

        # Add user callback to list of callbacks in learn_config
        learn_config = dict(self._learn_config)
        callbacks_list: MaybeCallback = learn_config.get("callback", [])
        if callbacks_list is None:
            callbacks_list = []
        if isinstance(callbacks_list, BaseCallback):
            callbacks_list = [callbacks_list]
        elif not isinstance(callbacks_list, list):
            callbacks_list = [ConvertCallback(callbacks_list)]
        callbacks_list.append(Sb3Callback(callback=self.callback, solver=self))
        learn_config["callback"] = callbacks_list

        self._algo.learn(**learn_config)

    def _sample_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        action, _ = self._algo.predict(self._unwrap_obs(observation))
        return self._wrap_action(action)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _save(self, path: str) -> None:
        self._algo.save(path)

    def _load(self, path: str):
        domain = self._domain_factory()
        env = DummyVecEnv([lambda: AsGymnasiumEnv(domain)])
        self._algo = self._algo_class.load(path, env=env)
        self._init_algo(domain)

    def _init_algo(self, domain: D):
        self._wrap_action = lambda a: next(
            iter(domain.get_action_space().from_unwrapped([a]))
        )
        self._unwrap_obs = lambda o: next(
            iter(domain.get_observation_space().to_unwrapped([o]))
        )

    def get_policy(self) -> BasePolicy:
        """Return the computed policy."""
        return self._algo.policy


class Sb3Callback(BaseCallback):
    def __init__(
        self, callback: Callable[[StableBaseline], bool], solver: StableBaseline
    ):
        super().__init__()
        self.solver = solver
        self.callback = callback

    def _on_step(self) -> bool:
        return not self.callback(self.solver)
