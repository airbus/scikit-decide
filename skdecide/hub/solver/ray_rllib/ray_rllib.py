# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable, Dict, Optional, Set, Type

import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.env.wrappers.multi_agent_env_compatibility import (
    MultiAgentEnvCompatibility,
)
from ray.rllib.models import ModelCatalog
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


class D(MultiAgentRLDomain):
    pass


class RayRLlib(Solver, Policies, Restorable):
    """This class wraps a Ray RLlib solver (ray[rllib]) as a scikit-decide solver.

    !!! warning
        Using this class requires Ray RLlib to be installed.
    """

    T_domain = D

    def __init__(
        self,
        algo_class: Type[Algorithm],
        train_iterations: int,
        config: Optional[AlgorithmConfig] = None,
        policy_configs: Optional[Dict[str, Dict]] = None,
        policy_mapping_fn: Optional[
            Callable[[str, Optional["EpisodeV2"], Optional["RolloutWorker"]], str]
        ] = None,
        action_embed_sizes: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initialize Ray RLlib.

        # Parameters
        algo_class: The class of Ray RLlib trainer/agent to wrap.
        train_iterations: The number of iterations to call the trainer's train() method.
        config: The configuration dictionary for the trainer.
        policy_configs: The mapping from policy id (str) to additional config (dict) (leave default for single policy).
        policy_mapping_fn: The function mapping agent ids to policy ids (leave default for single policy).
        action_embed_sizes: The mapping from policy id (str) to action embedding size (only used with domains filtering allowed actions per state, default to 2)
        """
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

    def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
        # Reuse algo if possible (enables further learning)
        if not hasattr(self, "_algo"):
            self._init_algo(domain_factory)

        # Training loop
        for _ in range(self._train_iterations):
            self._algo.train()

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
        self._algo.save(path)

    def _load(self, path: str, domain_factory: Callable[[], D]):
        self._init_algo(domain_factory)
        self._algo.restore(path)

    def _init_algo(self, domain_factory: Callable[[], D]):
        domain = domain_factory()
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

        # Instantiate algo
        register_env(
            "skdecide_env",
            lambda _, domain_factory=domain_factory, rayrllib=self: AsRLlibMultiAgentEnv(
                domain=domain_factory(),
                action_masking=rayrllib._action_masking,
                state_access=rayrllib._state_access,
            ),
        )
        # Disable env checking in case of action masking otherwise RLlib will try to simulate
        # next state transition with invalid actions, which might make some domains crash if
        # they require action masking
        self._config.environment(
            env="skdecide_env", disable_env_checking=self._action_masking
        )
        self._algo = self._algo_class(config=self._config)


class AsRLlibMultiAgentEnv(MultiAgentEnvCompatibility):
    def __init__(
        self,
        domain: D,
        action_masking: bool = False,
        state_access: Callable[D.T_agent[D.T_observation], D.T_state] = None,
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
