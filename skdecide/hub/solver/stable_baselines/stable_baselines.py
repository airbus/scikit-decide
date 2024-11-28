# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional, Union

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, ConvertCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback

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

    hyperparameters = [
        CategoricalHyperparameter(
            name="algo_class",
            choices={
                "A2C": A2C,
                "DDPG": DDPG,
                "DQN": DQN,
                "PPO": PPO,
                "SAC": SAC,
                "TD3": TD3,
            },
        ),
        FloatHyperparameter(
            name="learning_rate_log",  # learning_rate = 10 ** learning_rate_log
            low=-5,
            high=-1,
            step=1,
        ),
        IntegerHyperparameter(
            name="batch_size_log2",  # batch_size = 2 ** batch_size_log2
            low=4,
            high=8,
            depends_on=("algo_class", [DDPG, DQN, PPO, SAC, TD3]),
        ),
        FloatHyperparameter(
            name="gamma_complement_log",  # gamma = 1 - 10 ** gamma_complement_log
            low=-3,
            high=-1,
            step=1,
        ),
        FloatHyperparameter(
            name="ent_coef_log",  # ent_coef = 10 ** ent_coef_log
            low=-3,
            high=-1,
            step=1,
            depends_on=("algo_class", [A2C, PPO]),
        ),
    ]

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        algo_class: type[BaseAlgorithm],
        baselines_policy: Union[str, type[BasePolicy]],
        learn_config: Optional[dict[str, Any]] = None,
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
        kwargs: keyword arguments passed to the algo_class constructor.

        """
        Solver.__init__(self, domain_factory=domain_factory)
        self._algo_class = algo_class
        self._baselines_policy = baselines_policy
        self._learn_config = learn_config if learn_config is not None else {}
        self._algo_kwargs = kwargs
        self.callback = callback

        # Handle kwargs (potentially generated by optuna)
        if "total_timesteps" in kwargs:
            # total_timesteps is not for algo __init__() but algo learn()
            self._learn_config["total_timesteps"] = kwargs.pop("total_timesteps")
        if "batch_size_log2" in kwargs:
            # batch_size
            batch_size_log2 = kwargs.pop("batch_size_log2")
            kwargs["batch_size"] = 2**batch_size_log2
        if "gamma_complement_log" in kwargs:
            # gamma
            gamma_complement_log = kwargs.pop("gamma_complement_log")
            kwargs["gamma"] = 1 - 10**gamma_complement_log
        if "learning_rate_log" in kwargs:
            # learning_rate
            learning_rate_log = kwargs.pop("learning_rate_log")
            kwargs["learning_rate"] = 10**learning_rate_log
        if "ent_coef_log" in kwargs:
            # ent_coef
            ent_coef_log = kwargs.pop("ent_coef_log")
            kwargs["ent_coef"] = 10**ent_coef_log

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
            env = AsGymnasiumEnv(domain)  # we let the algo wrap it in a vectorized env
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
        self, observation: D.T_agent[D.T_observation], **kwargs: Any
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        action, _ = self._algo.predict(self._unwrap_obs(observation))
        return self._wrap_action(action)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _save(self, path: str) -> None:
        self._algo.save(path)

    def _load(self, path: str):
        domain = self._domain_factory()
        self._algo = self._algo_class.load(path, env=AsGymnasiumEnv(domain))
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
