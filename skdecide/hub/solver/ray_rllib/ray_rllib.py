# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import glob
import os
from typing import Callable, Dict, Optional, Set, Type

import gymnasium as gym
import ray
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.env.wrappers.multi_agent_env_compatibility import (
    MultiAgentEnvCompatibility,
)
from ray.tune.registry import register_env

from skdecide import Domain, Solver
from skdecide.builders.domain import (
    Initializable,
    Sequential,
    SingleAgent,
    UnrestrictedActions,
)
from skdecide.builders.solver import Policies, Restorable
from skdecide.hub.domain.gym import AsLegacyGymV21Env
from skdecide.hub.space.gym import GymSpace


# TODO: remove UnrestrictedActions?
class D(Domain, Sequential, UnrestrictedActions, Initializable):
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
    ) -> None:
        """Initialize Ray RLlib.

        # Parameters
        algo_class: The class of Ray RLlib trainer/agent to wrap.
        train_iterations: The number of iterations to call the trainer's train() method.
        config: The configuration dictionary for the trainer.
        policy_configs: The mapping from policy id (str) to additional config (dict) (leave default for single policy).
        policy_mapping_fn: The function mapping agent ids to policy ids (leave default for single policy).
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
                self._unwrap_obs(v, k), policy_id=self._policy_mapping_fn(k, None, None)
            )
            for k, v in observation.items()
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
        self._wrap_action = lambda a: {
            k: next(iter(domain.get_action_space()[k].from_unwrapped([v])))
            for k, v in a.items()
        }
        self._unwrap_obs = lambda o, agent: next(
            iter(domain.get_observation_space()[agent].to_unwrapped([o]))
        )
        # Overwrite multi-agent config
        pol_obs_spaces = {
            self._policy_mapping_fn(k, None, None): v.unwrapped()
            for k, v in domain.get_observation_space().items()
        }
        pol_act_spaces = {
            self._policy_mapping_fn(k, None, None): v.unwrapped()
            for k, v in domain.get_action_space().items()
        }
        policies = {
            k: (None, pol_obs_spaces[k], pol_act_spaces[k], v or {})
            for k, v in self._policy_configs.items()
        }
        self._config.multi_agent(
            policies=policies,
            policy_mapping_fn=self._policy_mapping_fn,
        )
        # Instanciate algo
        register_env("skdecide_env", lambda _: AsRLlibMultiAgentEnv(domain_factory()))
        self._config.environment(env="skdecide_env")
        self._algo = self._algo_class(config=self._config)


class AsRLlibMultiAgentEnv(MultiAgentEnvCompatibility):
    def __init__(self, domain: D, render_mode: Optional[str] = None) -> None:
        old_env = AsLegacyRLlibMultiAgentEnv(domain=domain)
        self._domain = domain
        super().__init__(old_env=old_env, render_mode=render_mode)

    def get_agent_ids(self) -> Set[str]:
        return self._domain.get_agents()


class AsLegacyRLlibMultiAgentEnv(AsLegacyGymV21Env):
    def __init__(self, domain: D, unwrap_spaces: bool = True) -> None:
        """Initialize AsRLlibMultiAgentEnv.

        # Parameters
        domain: The scikit-decide domain to wrap as a RLlib multi-agent environment.
        unwrap_spaces: Boolean specifying whether the action & observation spaces should be unwrapped.
        """
        self._domain = domain
        self._unwrap_spaces = unwrap_spaces
        if unwrap_spaces:
            self.observation_space = gym.spaces.Dict(
                {
                    k: agent_observation_space.unwrapped()
                    for k, agent_observation_space in domain.get_observation_space().items()
                }
            )
            self.action_space = gym.spaces.Dict(
                {
                    k: agent_action_space.unwrapped()
                    for k, agent_action_space in domain.get_action_space().items()
                }
            )
        else:
            self.observation_space = domain.get_observation_space()
            self.action_space = (
                domain.get_action_space()
            )  # assumes all actions are always applicable

    def reset(self):
        """Resets the env and returns observations from ready agents.

        # Returns
        obs (dict): New observations for each ready agent.
        """
        raw_observation = super().reset()
        observation = {
            k: next(iter(self._domain.get_observation_space()[k].to_unwrapped([v])))
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
            k: next(iter(self._domain.get_action_space()[k].from_unwrapped([v])))
            for k, v in action_dict.items()
        }
        outcome = self._domain.step(action)
        observations = {
            k: next(iter(self._domain.get_observation_space()[k].to_unwrapped([v])))
            for k, v in outcome.observation.items()
        }
        rewards = {k: v.reward for k, v in outcome.value.items()}
        done = outcome.termination
        done.update({"__all__": all(outcome.termination.values())})
        infos = {k: (v or {}) for k, v in outcome.info.items()}
        return observations, rewards, done, infos
