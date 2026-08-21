import random

from gymnasium.spaces import Dict, Discrete
from ray.rllib import MultiAgentEnv
from ray.rllib.algorithms import PPO
from ray.rllib.connectors.common.flatten_observations import FlattenObservations
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.tune.registry import get_trainable_cls, register_env  # noqa


class MultiagentCorridor(MultiAgentEnv):
    """Example of a custom env in which the agents has to walk down a corridor.

    ------------
    |S........G ....... S|
    ------------
    , where S is the starting position, G is the goal position, and fields with '.'
    mark free spaces, over which the agent may step. The length of the above example
    corridor is 10.
    Allowed actions are left (0) and right (1).
    The reward function is -0.01 per step taken and a uniform random value between
    0.5 and 1.5 when reaching the goal state.

    """

    def __init__(self, config=None):
        super().__init__()

        self.agents = self.possible_agents = ["player1", "player2"]
        self.corridor_length = 7
        self.end_pos = 4
        self.cur_pos = {
            "player1": 0,
            "player2": 0,
        }
        self.nb_resets = 0
        self.action_spaces = {
            "player1": Discrete(2),
            "player2": Discrete(2),
        }
        self.observation_spaces = {
            "player1": Discrete(self.corridor_length),
            "player2": Discrete(self.corridor_length),
        }
        self.action_space = Dict(self.action_spaces)
        self.observation_space = Dict(self.observation_spaces)

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        match self.nb_resets % 4:
            case 0:
                self.cur_pos = {
                    "player1": 0,
                    "player2": 0,
                }
            case 1:
                self.cur_pos = {
                    "player1": 0,
                    "player2": 6,
                }
            case 2:
                self.cur_pos = {
                    "player1": 6,
                    "player2": 0,
                }
            case _:
                self.cur_pos = {
                    "player1": 6,
                    "player2": 6,
                }
        self.nb_resets += 1
        # Return obs and (empty) info dict.
        return self.cur_pos, {"env_state": "reset"}

    def step(self, action):
        for key, agent_action in action.items():
            # Move left.
            if agent_action == 0 and self.cur_pos[key] > 0:
                self.cur_pos[key] -= 1
            # Move right.
            elif agent_action == 1 and self.cur_pos[key] < self.corridor_length - 1:
                self.cur_pos[key] += 1

        # The environment only ever terminates when one agent reaches the goal state.
        terminated = {
            key: cur_pos == self.end_pos for key, cur_pos in self.cur_pos.items()
        }
        truncated = {key: False for key, cur_pos in self.cur_pos.items()}
        terminated["__all__"] = any(
            cur_pos == self.end_pos for cur_pos in self.cur_pos.values()
        )
        truncated["__all__"] = False
        # Produce a random reward from [0.5, 1.5] when we reach the goal.
        reward = {
            key: random.uniform(0.5, 1.5) if agent_terminated else -0.01
            for key, agent_terminated in terminated.items()
        }
        infos = {}
        return (
            self.cur_pos,
            reward,
            terminated,
            truncated,
            infos,
        )


if __name__ == "__main__":
    # Define env-to-module-connector pipeline for the new stack.
    def _env_to_module_pipeline(env, spaces, device):
        return FlattenObservations(multi_agent=True)

    config = (
        PPO.get_default_config()
        .environment(
            MultiagentCorridor,  # or provide the registered string: "corridor-env"
        )
        .multi_agent(
            policies={
                "policy"
            },  # detected as multiagent if len >1 or different from  {DEFAULT_POLICY_ID}
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "policy",
        )
        .learners(num_learners=0)
        .training(train_batch_size_per_learner=256)
        .env_runners(num_env_runners=0, env_to_module_connector=_env_to_module_pipeline)
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "policy": RLModuleSpec(model_config=DefaultModelConfig())
                }
            )
        )
    )

    algo = config.build()
    results = algo.train()

    def compute_action(observation):
        if algo.env_runner:
            env_to_module = algo.env_runner._env_to_module
            module_to_env = algo.env_runner._module_to_env
        else:
            env_to_module = algo.env_to_module_connector
            module_to_env = algo.module_to_env_connector

        rl_module = algo.env_runner.module
        episode = MultiAgentEpisode(
            observations=[observation],
            observation_space=env.observation_space,
            action_space=env.action_space,
            agent_to_module_mapping_fn=algo.env_runner.config.policy_mapping_fn,
        )
        shared_data = {}  # will hold mapping between model_id and agent_id
        batch = env_to_module(
            episodes=[episode],
            rl_module=rl_module,
            explore=False,
            shared_data=shared_data,
        )
        eval_outputs = rl_module.forward_inference(batch)
        action = module_to_env(
            batch=eval_outputs,
            episodes=[episode],
            rl_module=rl_module,
            explore=False,
            shared_data=shared_data,
        )["actions"][0]
        return action

    # inference
    env = MultiagentCorridor()
    # observation, info = env.reset()
    #
    # action = compute_action(observation)
    # print(observation, action)

    observation = {"player1": 0, "player2": 6}
    action = compute_action(observation)
    print(observation, action)

    observation = {"player1": 6, "player2": 6}
    action = compute_action(observation)
    print(observation, action)

    #
    # # before goal
    # observation = 3
    # action = compute_action(observation)
    # print(observation, action)
    #
    # # after goal
    # observation = 6
    # action = compute_action(observation)
    # print(observation, action)
