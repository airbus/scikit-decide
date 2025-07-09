# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)
from sb3_contrib.common.wrappers import ActionMasker
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
from skdecide.builders.solver import Maskable, Policies, Restorable
from skdecide.hub.domain.gym import AsGymnasiumEnv
from skdecide.hub.space.gym import GymSpace, MultiDiscreteSpace

logger = logging.getLogger(__name__)


class D(Domain, SingleAgent, Sequential, UnrestrictedActions, Initializable):
    pass


class StableBaseline(Solver, Policies, Restorable, Maskable):
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
        use_action_masking: bool = False,
        autoregressive_action: bool = False,
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
        use_action_masking: if True,
          - the domain will be wrapped in a gymnasium environment exposing `action_masks()`,
          - `self.sample_action()` will pass action masks to underlying sb3 algo's `predict()`
             (e.g. MaskablePPO or MaskableGraphPPO),
          - `self.using_applicable_actions()` will return True so that rollout knows to retrieve action masks
             before sampling actions.
        autoregressive_action: if True,
          - the domain will be wrapped in a gymnasium environment exposing a `action_masks()` method
            returning applicable actions as a numpy array (needed by AutoregressivePPO algorithm),
          - `self.sample_action()` will pass applicable action to underlying sb3 algo's `predict()`,
          - `self.using_applicable_actions()` will return True so that rollout knows to retrieve applicable action
             before sampling actions.
          - hypotheses: action space should be a MultiDiscreteSpace
            and get_applicable_actions return an enumerable space.
        kwargs: keyword arguments passed to the algo_class constructor.

        """
        Solver.__init__(self, domain_factory=domain_factory)
        self._algo_class = algo_class
        self._baselines_policy = baselines_policy
        self._learn_config = learn_config if learn_config is not None else {}
        self._algo_kwargs = kwargs
        self.callback = callback
        self.use_action_masking = use_action_masking
        self.autoregressive_action = autoregressive_action
        self._applicable_actions: Optional[npt.NDArray[int]] = None
        domain = self._domain_factory()
        self._learning_env = self._as_gymnasium_env(domain)
        self._wrap_action = lambda a: next(
            iter(domain.get_action_space().from_unwrapped([a]))
        )
        self._unwrap_obs = lambda o: next(
            iter(domain.get_observation_space().to_unwrapped([o]))
        )

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

    def using_applicable_actions(self):
        """Tell if the solver is able to use applicable actions information."""
        return self.use_action_masking or self.autoregressive_action

    def retrieve_applicable_actions(self, domain: Domain) -> None:
        if self.autoregressive_action:
            self._applicable_actions = np.array(
                domain.get_applicable_actions().get_elements()
            )
        else:
            super().retrieve_applicable_actions(domain)

    def get_applicable_actions(self) -> Optional[npt.NDArray[int]]:
        return self._applicable_actions

    def _as_gymnasium_env(self, domain: Domain) -> gym.Env:
        if self.autoregressive_action:
            return as_masked_autoregressive_gymnasium_env(domain)
        elif self.use_action_masking:
            return as_masked_gymnasium_env(domain)
        else:
            return as_gymnasium_env(domain)

    @classmethod
    def _check_domain_additional(cls, domain: Domain) -> bool:
        return isinstance(domain.get_action_space(), GymSpace) and isinstance(
            domain.get_observation_space(), GymSpace
        )

    def _init_algo(self):
        self._algo = self._algo_class(
            self._baselines_policy, self._learning_env, **self._algo_kwargs
        )

    def _solve(self) -> None:
        # TODO: improve code for parallelism
        #  (https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
        #  #multiprocessing-unleashing-the-power-of-vectorized-environments)?
        if not hasattr(
            self, "_algo"
        ):  # reuse algo if possible (enables further learning)
            self._init_algo()

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
        if self.autoregressive_action:
            # e.g. algo = AutoregressivePPO
            action, _ = self._algo.predict(
                self._unwrap_obs(observation),
                action_masks=self.get_applicable_actions(),
            )
        elif self.use_action_masking:
            # e.g. algo = MaskablePPO or MaskableGraphPPO
            action, _ = self._algo.predict(
                self._unwrap_obs(observation), action_masks=self.get_action_mask()
            )
        else:
            action, _ = self._algo.predict(self._unwrap_obs(observation))
        return self._wrap_action(action)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _save(self, path: str) -> None:
        self._algo.save(path)

    def _load(self, path: str):
        load_algo_kwargs = dict(self._algo_kwargs)
        if "policy_kwargs" in load_algo_kwargs:
            logger.warning("load(): ignoring 'policy_kwargs'.")
            load_algo_kwargs.pop("policy_kwargs")
        self._algo = self._algo_class.load(
            path, env=self._learning_env, **load_algo_kwargs
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


def as_gymnasium_env(domain: Domain) -> gym.Env:
    """Wraps the domain into a gymnasium env.

    To be fed to sb3 algorithms.

    """
    return AsGymnasiumEnv(domain=domain)


def as_masked_gymnasium_env(domain: Domain) -> gym.Env:
    """Wraps the domain into an action-masked gymnasium env.

    This means that it exposes a method `self.action_masks()` as expected by algorithms like
    `sb3_contrib.MaskablePPO`.

    """
    return ActionMasker(
        env=AsGymnasiumEnv(domain=domain),
        action_mask_fn=lambda env: env.domain.get_action_mask(),
    )


def as_masked_autoregressive_gymnasium_env(domain: Domain) -> gym.Env:
    """Wraps the domain into an action-masked gymnasium env but for autoregressive action prediction.

    This means that it exposes a method `self.action_masks()` as expected by algorithms like
    `sb3_contrib.MaskablePPO`.
    In the autoregressive case, we need actually that`self.action_masks()` return the list of applicable actions
    in multidiscrete mode as a numpy array, with variable number of rows (= number of applicable actions),
    and a fixed number of columns (len(action_space.nvec)), and we let the possibility of -1 values which mean that
    the corresponding components are irrelevant (in case of variable components size actions)

    Hypotheses
    - domain.get_action_space() returns a MultiDiscreteSpace
    - domain.get_applicable_actions() returns an enumerable space

    """
    assert isinstance(domain.get_action_space(), MultiDiscreteSpace)
    return ActionMasker(
        env=AsGymnasiumEnv(domain=domain),
        action_mask_fn=lambda env: np.array(
            env.domain.get_applicable_actions().get_elements()
        ),
    )
