# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Type, Union

import gymnasium as gym
import numpy as np
import ray
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
    SubBrickKwargsHyperparameter,
)
from packaging.version import Version
from ray.rllib.algorithms import DQN, PPO, SAC
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.multi_agent_env_compatibility import (
    MultiAgentEnvCompatibility,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.from_config import NotProvided
from ray.tune.registry import register_env

from skdecide import Domain, Solver
from skdecide.builders.domain import SingleAgent, UnrestrictedActions
from skdecide.builders.domain.observability import FullyObservable
from skdecide.builders.solver import Policies, Restorable
from skdecide.core import EnumerableSpace
from skdecide.domains import MultiAgentRLDomain
from skdecide.hub.domain.gym import AsLegacyGymV21Env
from skdecide.hub.space.gym import GymSpace

from .custom_models import TFParametricActionsModel, TorchParametricActionsModel

if TYPE_CHECKING:
    # imports useful only in annotations, may change according to releases
    from ray.rllib import RolloutWorker


class D(MultiAgentRLDomain):
    pass


class RayRLlib(Solver, Policies, Restorable):
    """This class wraps a Ray RLlib solver (ray[rllib]) as a scikit-decide solver.

    !!! warning
        Using this class requires Ray RLlib to be installed.
    """

    T_domain = D

    hyperparameters = [
        IntegerHyperparameter(name="train_iterations", low=1, high=3),
        CategoricalHyperparameter(
            name="algo_class",
            choices={
                "PPO": PPO,
                "DQN": DQN,
                "SAC": SAC,
            },
        ),
        FloatHyperparameter(
            name="gamma_complement_log",  # gamma = 1 - 10 ** gamma_complement_log
            low=-3,
            high=-1,
            step=1,
        ),
        FloatHyperparameter(
            name="lr_log",  # lr = 10 ** lr_log
            low=-5,
            high=-1,
            step=1,
        ),
        IntegerHyperparameter(
            name="train_batch_size_log2",  # train_batch_size = 2 ** train_batch_size_log2
            low=4,
            high=8,
            depends_on=("algo_class", [DQN, SAC]),
        ),
        IntegerHyperparameter(
            name="sgd_minibatch_size_log2",  # sgd_minibatch_size = 2 ** sgd_minibatch_size_log2
            low=4,
            high=8,
            depends_on=("algo_class", [PPO]),
        ),
        FloatHyperparameter(
            name="entropy_coeff_log",  # entropy_coeff = 10 ** entropy_coeff_log
            low=-3,
            high=-1,
            step=1,
            depends_on=("algo_class", [PPO]),
        ),
    ]

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        algo_class: Type[Algorithm],
        train_iterations: int,
        config: Optional[AlgorithmConfig] = None,
        policy_configs: Optional[Dict[str, Dict]] = None,
        policy_mapping_fn: Optional[
            Callable[[str, Optional["EpisodeV2"], Optional["RolloutWorker"]], str]
        ] = None,
        action_embed_sizes: Optional[Dict[str, int]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        callback: Callable[[RayRLlib], bool] = lambda solver: False,
        **kwargs,
    ) -> None:
        """Initialize Ray RLlib.

        # Parameters
        domain_factory: A callable with no argument returning the domain to solve (can be a mere domain class).
            The resulting domain will be auto-cast to the level expected by the solver.
        algo_class: The class of Ray RLlib trainer/agent to wrap.
        train_iterations: The number of iterations to call the trainer's train() method.
        config: The configuration dictionary for the trainer.
        policy_configs: The mapping from policy id (str) to additional config (dict) (leave default for single policy).
        policy_mapping_fn: The function mapping agent ids to policy ids (leave default for single policy).
        action_embed_sizes: The mapping from policy id (str) to action embedding size (only used with domains filtering allowed actions per state, default to 2)
        config_kwargs: keyword arguments for the `AlgorithmConfigKwargs` class used to update programmatically the algorithm config.
            Will be used by hyerparameters tuners like optuna. Should probably not be used directly by the user,
            who could rather directly specify the correct `config`.
        callback: function called at each solver iteration.
            If returning true, the solve process stops and exit the current train iteration.
            However, if train_iterations > 1, another train loop will be entered after that.
            (One can code its callback in such a way that further training loop are stopped directly after that.)

        """
        Solver.__init__(self, domain_factory=domain_factory)
        self.callback = callback
        self._algo_class = algo_class
        self._train_iterations = train_iterations
        self._config = config or algo_class.get_default_config()
        if policy_configs is None:
            self._policy_configs = {"policy": {}}
        else:
            self._policy_configs = policy_configs
        if policy_mapping_fn is None:
            self._policy_mapping_fn = lambda agent_id, episode, worker: "policy"
        else:
            self._policy_mapping_fn = policy_mapping_fn
        self._action_embed_sizes = (
            action_embed_sizes
            if action_embed_sizes is not None
            else {k: 2 for k in self._policy_configs.keys()}
        )
        if self._action_embed_sizes.keys() != self._policy_configs.keys():
            raise RuntimeError(
                "Action embed size keys must be the same as policy config keys"
            )

        ray.init(ignore_reinit_error=True)
        self._algo_callbacks: Optional[DefaultCallbacks] = None
        self._algo_worker_callbacks: Optional[DefaultCallbacks] = None
        self._algo_evaluation_worker_callbacks: Optional[DefaultCallbacks] = None

        # Handle kwargs (potentially generated by optuna)
        if "train_batch_size_log2" in kwargs:
            # train_batch_size
            train_batch_size_log2 = kwargs.pop("train_batch_size_log2")
            kwargs["train_batch_size"] = 2**train_batch_size_log2
        if "sgd_minibatch_size_log2" in kwargs:
            # sgd_minibatch_size
            sgd_minibatch_size_log2 = kwargs.pop("sgd_minibatch_size_log2")
            kwargs["sgd_minibatch_size"] = 2**sgd_minibatch_size_log2
        if "gamma_complement_log" in kwargs:
            # gamma
            gamma_complement_log = kwargs.pop("gamma_complement_log")
            kwargs["gamma"] = 1 - 10**gamma_complement_log
        if "lr_log" in kwargs:
            # lr
            lr_log = kwargs.pop("lr_log")
            kwargs["lr"] = 10**lr_log
        if "entropy_coeff_log" in kwargs:
            # entropy_coeff
            entropy_coeff_log = kwargs.pop("entropy_coeff_log")
            kwargs["entropy_coeff"] = 10**entropy_coeff_log

        # Update algorithm config with hyperparameters found in kwargs
        if kwargs:
            self._config.update_from_dict(kwargs)

    def get_policy(self) -> Dict[str, Policy]:
        """Return the computed policy."""
        return {
            policy_id: self._algo.get_policy(policy_id=policy_id)
            for policy_id in self._policy_configs
        }

    @classmethod
    def _check_domain_additional(cls, domain: Domain) -> bool:
        if isinstance(domain, SingleAgent):
            return isinstance(domain.get_action_space(), GymSpace) and isinstance(
                domain.get_observation_space(), GymSpace
            )
        else:
            return all(
                isinstance(a, GymSpace) for a in domain.get_action_space().values()
            ) and all(
                isinstance(o, GymSpace) for o in domain.get_observation_space().values()
            )

    def _solve(self) -> None:
        # Reuse algo if possible (enables further learning)
        if not hasattr(self, "_algo"):
            self._init_algo()

        # Training loop
        for _ in range(self._train_iterations):
            try:
                self._algo.train()
            except SolveEarlyStop as e:
                # if stopping exception raise, we choose to stop this train iteration
                pass

    def _sample_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        action = {
            k: self._algo.compute_single_action(
                self._unwrap_obs(observation, k),
                policy_id=self._policy_mapping_fn(k, None, None),
            )
            for k in observation.keys()
        }
        return self._wrap_action(action)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _save(self, path: str) -> None:
        self.forget_callback()  # avoid serializing issues
        self._algo.save(path)
        self.set_callback()  # put it back in case of further solve

    def _load(self, path: str):
        self._init_algo()
        self._algo.restore(path)
        self.set_callback()  # ensure putting back actual callback

    def _init_algo(self) -> None:
        domain = self._domain_factory()
        wrapped_action_space = domain.get_action_space()
        wrapped_observation_space = domain.get_observation_space()
        # Test if the domain is using restricted actions or not
        self._action_masking = (
            (not isinstance(domain, UnrestrictedActions))
            and isinstance(domain, FullyObservable)
            and all(
                isinstance(agent_action_space, EnumerableSpace)
                for agent_action_space in wrapped_action_space.values()
            )
            and self._algo_class.__name__
            # Only the following algos handle action masking in ray[rllib]==2.9.0
            in ["APPO", "BC", "DQN", "Rainbow", "IMPALA", "MARWIL", "PPO"]
        )
        if self._action_masking:
            if self._config.get("framework") not in ["tf", "tf2", "torch"]:
                raise RuntimeError(
                    "Action masking (invalid action filtering) for RLlib requires TensorFlow or PyTorch to be installed"
                )
            ModelCatalog.register_custom_model(
                "skdecide_rllib_custom_model",
                TFParametricActionsModel
                if self._config.get("framework") in ["tf", "tf2"]
                else TorchParametricActionsModel
                if self._config.get("framework") == "torch"
                else NotProvided,
            )
            if self._algo_class.__name__ == "DQN":
                self._config.training(
                    hiddens=[],
                    dueling=False,
                )
            elif self._algo_class.__name__ == "PPO":
                self._config.training(
                    model={"vf_share_layers": True},
                )
            # States are not automatically autocasted when the domain is actually single agent
            # because the type hints of states in functions taking them as argument are not in
            # the form of D.T_agent[...] (since they are global to all the agents in multi-agent settings)
            self._state_access = (
                (lambda s: next(iter(s.values())))
                if isinstance(domain, SingleAgent)
                else (lambda s: s)
            )
        else:
            self._state_access = None
        self._wrap_action = lambda a, wrapped_action_space=wrapped_action_space: {
            # Trick to assign v's wrapped value to self._wrap_action
            # (no wrapping method for single unwrapped values in enumerable spaces)
            k: next(iter(wrapped_action_space[k].from_unwrapped([v])))
            for k, v in a.items()
        }
        # Trick to assign o's unwrapped value to self._unwrap_obs
        # (no unwrapping method for single elements in enumerable spaces)
        self._unwrap_obs = (
            lambda obs, agent, domain=domain, wrapped_action_space=wrapped_action_space, wrapped_observation_space=wrapped_observation_space: next(
                iter(wrapped_observation_space[agent].to_unwrapped([obs[agent]]))
            )
            if not self._action_masking
            else {
                "valid_avail_actions_mask": np.array(
                    [
                        1
                        if domain.get_applicable_actions(self._state_access(obs))[
                            agent
                        ].contains(a)
                        else 0
                        for a in wrapped_action_space[agent].get_elements()
                    ],
                    dtype=np.int64,
                ),
                "true_obs": next(
                    iter(wrapped_observation_space[agent].to_unwrapped([obs[agent]]))
                ),
            }
        )
        # Overwrite multi-agent config
        pol_obs_spaces = (
            {
                self._policy_mapping_fn(k, None, None): v.unwrapped()
                for k, v in wrapped_observation_space.items()
            }
            if not self._action_masking
            else {
                self._policy_mapping_fn(k, None, None): gym.spaces.Dict(
                    {
                        "valid_avail_actions_mask": gym.spaces.Box(
                            0,
                            1,
                            shape=(len(wrapped_action_space[k].get_elements()),),
                            dtype=np.int64,
                        ),
                        "true_obs": v.unwrapped(),
                    }
                )
                for k, v in wrapped_observation_space.items()
            }
        )
        pol_act_spaces = {
            self._policy_mapping_fn(k, None, None): v.unwrapped()
            for k, v in wrapped_action_space.items()
        }

        policies = (
            {
                k: (None, pol_obs_spaces[k], pol_act_spaces[k], v or {})
                for k, v in self._policy_configs.items()
            }
            if not self._action_masking
            else {
                self._policy_mapping_fn(k, None, None): (
                    None,
                    pol_obs_spaces[k],
                    pol_act_spaces[k],
                    {
                        **(self._policy_configs[k] or {}),
                        **{
                            "model": {
                                "custom_model": "skdecide_rllib_custom_model",
                                "custom_model_config": {
                                    "true_obs_space": pol_obs_spaces[k].spaces[
                                        "true_obs"
                                    ],
                                    "action_embed_size": action_embed_size,
                                },
                            },
                        },
                    },
                )
                for k, action_embed_size in self._action_embed_sizes.items()
            }
        )
        self._config.multi_agent(
            policies=policies,
            policy_mapping_fn=self._policy_mapping_fn,
        )

        register_env(
            "skdecide_env",
            lambda _, domain_factory=self._domain_factory, rayrllib=self: AsRLlibMultiAgentEnv(
                domain=domain_factory(),
                action_masking=rayrllib._action_masking,
                state_access=rayrllib._state_access,
            ),
        )
        if Version(ray.__version__) >= Version("2.20.0"):
            # starting from ray 2.20, no more checks on environment are made,
            # and `disable_env_checking` use raises an error
            self._config.environment(env="skdecide_env")
        else:
            # Disable env checking in case of action masking otherwise RLlib will try to simulate
            # next state transition with invalid actions, which might make some domains crash if
            # they require action masking
            self._config.environment(
                env="skdecide_env", disable_env_checking=self._action_masking
            )

        # set callback class for algo config
        self.set_callback()

        # Instantiate algo
        self._algo = self._algo_class(config=self._config)

    def set_callback(self):
        """Set back callback.

        Useful to do it after serializing/deserializing because of potential issues with
        - lambda functions
        - dynamic classes

        """
        # generate specific callback class
        callbacks_class = generate_rllibcallback_class(
            callback=self.callback, solver=self
        )
        # use it in all algo config, and callbacks attributes
        self._set_callbackclass(callbacks_class=callbacks_class)

    def forget_callback(self):
        """Forget about actual callback to avoid serializing issues."""
        # use default callback class
        callbacks_class = DefaultCallbacks
        # use it in algo config & evaluation_config, worker config, and for algo.callbacks, worker.callbacks
        self._set_callbackclass(callbacks_class=callbacks_class)

    def _set_callbackclass(self, callbacks_class: Type[DefaultCallbacks]):
        _set_callbackclass_in_config(
            callbacks_class=callbacks_class, config=self._config
        )
        if hasattr(self, "_algo"):
            tmp = self._algo.callbacks
            if (
                self._algo_callbacks
                and self._algo_callbacks.__class__ is callbacks_class
            ):
                self._algo.callbacks = self._algo_callbacks
            else:
                self._algo.callbacks = callbacks_class()
            self._algo_callbacks = tmp
            if self._algo.evaluation_config:
                _set_callbackclass_in_config(
                    callbacks_class=callbacks_class, config=self._algo.evaluation_config
                )
            if self._algo.workers:
                if Version(ray.__version__) < Version("2.34.0"):
                    # starting from 2.34, algo.workers becomes algo.env_runner_group
                    local_worker: RolloutWorker = self._algo.workers.local_worker()
                else:
                    local_worker: RolloutWorker = (
                        self._algo.env_runner_group.local_worker()
                    )
                if local_worker:
                    _set_callbackclass_in_config(
                        callbacks_class=callbacks_class, config=local_worker.config
                    )
                    self._algo_worker_callbacks = _swap_callbacks(
                        algo_or_worker=local_worker,
                        previous_callbacks=self._algo_worker_callbacks,
                        callbacks_class=callbacks_class,
                    )
                    for pid, policy in local_worker.policy_map.items():
                        policy.config["callbacks"] = callbacks_class


def _set_callbackclass_in_config(
    callbacks_class: Type[DefaultCallbacks], config: AlgorithmConfig
) -> None:
    is_frozen = config._is_frozen
    if is_frozen:
        # allow callbacks update
        config._is_frozen = False
    config.callbacks(callbacks_class=callbacks_class)
    config._is_frozen = is_frozen


def _swap_callbacks(
    algo_or_worker: Union[Algorithm, RolloutWorker, Policy],
    previous_callbacks,
    callbacks_class,
):
    tmp = algo_or_worker.callbacks
    if previous_callbacks and previous_callbacks.__class__ is callbacks_class:
        algo_or_worker.callbacks = previous_callbacks
    else:
        algo_or_worker.callbacks = callbacks_class()
    previous_callbacks = tmp
    return previous_callbacks


class AsRLlibMultiAgentEnv(MultiAgentEnvCompatibility):
    def __init__(
        self,
        domain: D,
        action_masking: bool = False,
        state_access: Callable[[D.T_agent[D.T_observation]], D.T_state] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        old_env = AsLegacyRLlibMultiAgentEnv(
            domain=domain, action_masking=action_masking, state_access=state_access
        )
        self._domain = domain
        super().__init__(old_env=old_env, render_mode=render_mode)

    def get_agent_ids(self) -> Set[str]:
        return self._domain.get_agents()


class AsLegacyRLlibMultiAgentEnv(AsLegacyGymV21Env):
    def __init__(
        self,
        domain: D,
        action_masking: bool,
        state_access: Callable[D.T_agent[D.T_observation], D.T_state],
        unwrap_spaces: bool = True,
    ) -> None:
        """Initialize AsLegacyRLlibMultiAgentEnv.

        # Parameters
        domain: The scikit-decide domain to wrap as a RLlib multi-agent environment.
        action_masking: Boolean specifying whether action masking is used
        state_access: Lambda function auto-casting fully observable observations in single or multi agent states accordingly
        unwrap_spaces: Boolean specifying whether the action & observation spaces should be unwrapped.
        """
        self._domain = domain
        self._action_masking = action_masking
        self._state_access = state_access
        self._unwrap_spaces = unwrap_spaces
        self._wrapped_observation_space = domain.get_observation_space()
        self._wrapped_action_space = domain.get_action_space()
        if unwrap_spaces:
            if not self._action_masking:
                self.observation_space = gym.spaces.Dict(
                    {
                        k: agent_observation_space.unwrapped()
                        for k, agent_observation_space in self._wrapped_observation_space.items()
                    }
                )
            else:
                self.observation_space = gym.spaces.Dict(
                    {
                        k: gym.spaces.Dict(
                            {
                                "valid_avail_actions_mask": gym.spaces.Box(
                                    0,
                                    1,
                                    shape=(
                                        len(
                                            self._wrapped_action_space[k].get_elements()
                                        ),
                                    ),
                                    dtype=np.int64,
                                ),
                                "true_obs": agent_observation_space.unwrapped(),
                            }
                        )
                        for k, agent_observation_space in self._wrapped_observation_space.items()
                    }
                )
            self.action_space = gym.spaces.Dict(
                {
                    k: agent_action_space.unwrapped()
                    for k, agent_action_space in self._wrapped_action_space.items()
                }
            )
        else:
            if not self._action_masking:
                self.observation_space = self._wrapped_observation_space
            else:
                self.observation_space = gym.spaces.Dict(
                    {
                        k: gym.spaces.Dict(
                            {
                                "valid_avail_actions_mask": gym.spaces.Box(
                                    0,
                                    1,
                                    shape=(
                                        len(
                                            self._wrapped_action_space[k].get_elements()
                                        ),
                                    ),
                                    dtype=np.int64,
                                ),
                                "true_obs": agent_observation_space,
                            }
                        )
                        for k, agent_observation_space in self._wrapped_observation_space.items()
                    }
                )
            self.action_space = self._wrapped_action_space

    def reset(self):
        """Resets the env and returns observations from ready agents.

        # Returns
        obs (dict): New observations for each ready agent.
        """
        raw_observation = self._domain.reset()
        if not self._action_masking:
            observation = {
                # Trick to assign v's unwrapped value to k
                # (no unwrapping method for single elements in enumerable spaces)
                k: next(iter(self._wrapped_observation_space[k].to_unwrapped([v])))
                for k, v in raw_observation.items()
            }
        else:
            applicable_actions = self._domain.get_applicable_actions(
                self._state_access(raw_observation)
            )
            observation = {
                # Trick to assign v's unwrapped value to k
                # (no unwrapping method for single elements in enumerable spaces)
                k: {
                    "valid_avail_actions_mask": np.array(
                        [
                            1 if applicable_actions[k].contains(a) else 0
                            for a in self._wrapped_action_space[k].get_elements()
                        ],
                        dtype=np.int64,
                    ),
                    "true_obs": next(
                        iter(self._wrapped_observation_space[k].to_unwrapped([v]))
                    ),
                }
                for k, v in raw_observation.items()
            }
        return observation

    def step(self, action_dict):
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        # Returns
        obs (dict): New observations for each ready agent.
        rewards (dict): Reward values for each ready agent. If the episode is just started, the value will be None.
        dones (dict): Done values for each ready agent. The special key "__all__" (required) is used to indicate env
            termination.
        infos (dict): Optional info values for each agent id.
        """
        action = {
            # Trick to assign v's wrapped value to k
            # (no wrapping method from single unwrapped values in enumerable spaces)
            k: next(iter(self._wrapped_action_space[k].from_unwrapped([v])))
            for k, v in action_dict.items()
        }
        outcome = self._domain.step(action)
        if not self._action_masking:
            observations = {
                # Trick to assign v's unwrapped value to k
                # (no unwrapping method for single elements in enumerable spaces)
                k: next(iter(self._wrapped_observation_space[k].to_unwrapped([v])))
                for k, v in outcome.observation.items()
            }
        else:
            applicable_actions = self._domain.get_applicable_actions(
                self._state_access(outcome.observation)
            )
            observations = {
                # Trick to assign v's unwrapped value to k
                # (no unwrapping method for single elements in enumerable spaces)
                k: {
                    "valid_avail_actions_mask": np.array(
                        [
                            1 if applicable_actions[k].contains(a) else 0
                            for a in self._wrapped_action_space[k].get_elements()
                        ],
                        dtype=np.int64,
                    ),
                    "true_obs": next(
                        iter(self._wrapped_observation_space[k].to_unwrapped([v]))
                    ),
                }
                for k, v in outcome.observation.items()
            }
        rewards = {k: v.reward for k, v in outcome.value.items()}
        done = outcome.termination
        done.update({"__all__": all(outcome.termination.values())})
        infos = {k: (v or {}) for k, v in outcome.info.items()}
        return observations, rewards, done, infos


class BaseRLlibCallback(DefaultCallbacks):
    callback: _CallbackWrapper
    solver: RayRLlib

    def on_episode_step(
        self,
        *args,
        **kwargs,
    ) -> None:
        stopping = self.callback(self.solver)
        if stopping:
            raise SolveEarlyStop("Solve process stopped by user callback")


class _CallbackWrapper:
    """Wrapper to avoid surprises with lambda functions"""

    def __init__(self, callback: Callable[[RayRLlib], bool]):
        self.callback = callback

    def __call__(self, solver) -> bool:
        return self.callback(solver)


class SolveEarlyStop(Exception):
    """Exception raised if a callback tells to stop the solve process."""


def generate_rllibcallback_class(
    callback: _CallbackWrapper, solver: RayRLlib, classname=None
) -> Type[BaseRLlibCallback]:
    if classname is None:
        classname = f"MyCallbackClass{id(solver)}"
    return type(
        classname,
        (BaseRLlibCallback,),
        dict(solver=solver, callback=_CallbackWrapper(callback=callback)),
    )
