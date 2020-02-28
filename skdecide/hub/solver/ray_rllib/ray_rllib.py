# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import glob
import os
from typing import Optional, Callable, Type, Dict

import ray
from ray.rllib.agents.trainer import Trainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from skdecide import Domain, Solver
from skdecide.builders.domain import SingleAgent, Sequential, UnrestrictedActions, Initializable
from skdecide.builders.solver import Policies, Restorable
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

    def __init__(self, algo_class: Type[Trainer], train_iterations: int, config: Optional[Dict] = None,
                 policy_configs: Dict[str, Dict] = {'policy': {}},
                 policy_mapping_fn: Callable[[str], str] = lambda agent_id: 'policy') -> None:
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
        self._config = config or {}
        self._policy_configs = policy_configs
        self._policy_mapping_fn = policy_mapping_fn
        ray.init(ignore_reinit_error=True)

    @classmethod
    def _check_domain_additional(cls, domain: Domain) -> bool:
        if isinstance(domain, SingleAgent):
            return isinstance(domain.get_action_space(), GymSpace) and \
                   isinstance(domain.get_observation_space(), GymSpace)
        else:
            return all(isinstance(a, GymSpace) for a in domain.get_action_space().values()) \
                   and all(isinstance(o, GymSpace) for o in domain.get_observation_space().values())

    def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
        # Reuse algo if possible (enables further learning)
        if not hasattr(self, '_algo'):
            self._init_algo(domain_factory)

        # Training loop
        for _ in range(self._train_iterations):
            self._algo.train()

    def _sample_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        action = {k: self._algo.compute_action(self._unwrap_obs(v, k), policy_id=self._policy_mapping_fn(k)) for
                  k, v in observation.items()}
        return self._wrap_action(action)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _save(self, path: str) -> None:
        self._algo.save(path)

    def _load(self, path: str, domain_factory: Callable[[], D]):
        if not os.path.isfile(path):
            # Find latest checkpoint
            metadata_files = glob.glob(f'{path}/**/*.tune_metadata')
            latest_metadata_file = max(metadata_files, key=os.path.getctime)
            path = latest_metadata_file[:-len('.tune_metadata')]
        self._init_algo(domain_factory)
        self._algo.restore(path)

    def _init_algo(self, domain_factory: Callable[[], D]):
        domain = domain_factory()
        self._wrap_action = lambda a: {k: next(iter(domain.get_action_space()[k].from_unwrapped([v]))) for k, v in
                                       a.items()}
        self._unwrap_obs = lambda o, agent: next(iter(domain.get_observation_space()[agent].to_unwrapped([o])))
        # Overwrite multi-agent config
        pol_obs_spaces = {self._policy_mapping_fn(k): v.unwrapped() for k, v in domain.get_observation_space().items()}
        pol_act_spaces = {self._policy_mapping_fn(k): v.unwrapped() for k, v in domain.get_action_space().items()}
        policies = {k: (None, pol_obs_spaces[k], pol_act_spaces[k], v or {}) for k, v in self._policy_configs.items()}
        self._config['multiagent'] = {'policies': policies, 'policy_mapping_fn': self._policy_mapping_fn}
        # Instanciate algo
        register_env('skdecide_env', lambda _: AsRLlibMultiAgentEnv(domain_factory()))
        self._algo = self._algo_class(env='skdecide_env', config=self._config)


class AsRLlibMultiAgentEnv(MultiAgentEnv):

    def __init__(self, domain: D) -> None:
        """Initialize AsRLlibMultiAgentEnv.

        # Parameters
        domain: The scikit-decide domain to wrap as a RLlib multi-agent environment.
        """
        self._domain = domain

    def reset(self):
        """Resets the env and returns observations from ready agents.

        # Returns
        obs (dict): New observations for each ready agent.
        """
        raw_observation = self._domain.reset()
        observation = {k: next(iter(self._domain.get_observation_space()[k].to_unwrapped([v]))) for k, v in
                       raw_observation.items()}
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
        action = {k: next(iter(self._domain.get_action_space()[k].from_unwrapped([v]))) for k, v in action_dict.items()}
        outcome = self._domain.step(action)
        observations = {k: next(iter(self._domain.get_observation_space()[k].to_unwrapped([v]))) for k, v in
                        outcome.observation.items()}
        rewards = {k: v.reward for k, v in outcome.value.items()}
        done = {'__all__': outcome.termination}
        infos = {k: (v or {}) for k, v in outcome.info.items()}
        return observations, rewards, done, infos

    def unwrapped(self):
        """Unwrap the scikit-decide domain and return it.

        # Returns
        The original scikit-decide domain.
        """
        return self._domain


if __name__ == '__main__':
    from ray.rllib.agents.ppo import PPOTrainer
    from skdecide.hub.domain.rock_paper_scissors import RockPaperScissors
    from skdecide.utils import rollout

    domain_factory = lambda: RockPaperScissors()
    domain = domain_factory()
    if RayRLlib.check_domain(domain):
        solver_factory = lambda: RayRLlib(PPOTrainer, train_iterations=1)
        solver = RockPaperScissors.solve_with(solver_factory, domain_factory)
        rollout(domain, solver, action_formatter=lambda a: str({k: v.name for k, v in a.items()}),
                outcome_formatter=lambda o: f'{ {k: v.name for k, v in o.observation.items()} }'
                                            f' - rewards: { {k: v.reward for k, v in o.value.items()} }')
