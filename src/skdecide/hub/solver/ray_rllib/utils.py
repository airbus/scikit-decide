#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from ray.rllib.algorithms import Algorithm
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode


def compute_action_new_api_stack_multi_agent(algo: Algorithm, observation):
    # Get connectors for
    #  - preprocessing observation for rl_module
    #  - postprocessing rl_module outputs (generally logits) to sample an action
    if algo.env_runner:
        env_to_module = algo.env_runner._env_to_module
        module_to_env = algo.env_runner._module_to_env
    else:
        env_to_module = algo.env_to_module_connector
        module_to_env = algo.module_to_env_connector
    # rl module (multiagent)
    rl_module = algo.env_runner.module
    # wrap observation in an episode
    episode = MultiAgentEpisode(
        observations=[observation],
        # specify submodule to use for each agent
        agent_to_module_mapping_fn=algo.env_runner.config.policy_mapping_fn,
    )
    shared_data = {}  # important: will be updated to hold mapping between model_id and agent_id
    # preprocess observation
    batch = env_to_module(
        episodes=[episode], rl_module=rl_module, explore=False, shared_data=shared_data
    )
    # go through module
    eval_outputs = rl_module.forward_inference(batch)
    # postprocess outputs to sample an action
    action = module_to_env(
        batch=eval_outputs,
        episodes=[episode],
        rl_module=rl_module,
        explore=False,
        shared_data=shared_data,
    )["actions"][0]
    return action
