from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch_geometric as thg
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import (
    MaskableDictRolloutBuffer,
    MaskableRolloutBuffer,
)
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv

from skdecide.hub.solver.utils.gnn.torch_utils import thg_data_to_graph_instance

from .buffers import (
    DictGraphRolloutBuffer,
    Graph2GraphRolloutBuffer,
    GraphRolloutBuffer,
)
from .utils import obs_as_tensor
from .vec_env.dummy_vec_env import wrap_graph_env


class BaseGraphOnPolicyAlgorithm(OnPolicyAlgorithm):
    """Base class for On-Policy algorithms (ex: A2C/PPO) graph observations without __init__()."""

    support_action_masking = False
    """Whether this algorithm supports action masking.

    Useful to share the code between algorithms.

    """

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = False,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        This method is largely identical to the implementation found in the parent class and MaskablePPO.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :param use_masking: Whether to use invalid action masks during training
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if (
            use_masking
            and self.support_action_masking
            and not is_masking_supported(env)
        ):
            raise ValueError(
                "Environment does not support action masking. Consider using ActionMasker wrapper"
            )

        if use_masking and not self.support_action_masking:
            raise ValueError(
                f"The algorithm {self.__class__.__name__} does not support action masking."
            )

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                if use_masking and self.support_action_masking:
                    action_masks = get_action_masks(env)

                if self.support_action_masking:
                    actions, values, log_probs = self.policy(
                        obs_tensor, action_masks=action_masks
                    )
                else:
                    actions, values, log_probs = self.policy(obs_tensor)

            actions = self.actions_from_torch(actions)

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            if isinstance(
                rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
            ):
                rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                    action_masks=action_masks,
                )
            else:
                rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(obs_tensor)  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def actions_from_torch(self, actions: th.Tensor) -> np.ndarray:
        return actions.cpu().numpy()


class GraphOnPolicyAlgorithm(BaseGraphOnPolicyAlgorithm):
    """Base class for On-Policy algorithms (ex: A2C/PPO) with graph observations."""

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: GymEnv,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        **kwargs,
    ):
        # Use proper default rollout buffer class
        if rollout_buffer_class is None:
            if isinstance(env.observation_space, spaces.Graph):
                rollout_buffer_class = GraphRolloutBuffer
            elif isinstance(env.observation_space, spaces.Dict):
                rollout_buffer_class = DictGraphRolloutBuffer

        # Use proper VecEnv wrapper for env with Graph spaces
        env = wrap_graph_env(env)
        if env.num_envs > 1:
            raise NotImplementedError(
                "GraphOnPolicyAlgorithm not implemented for real vectorized environment "
                "(ie. with more than 1 wrapped environment)"
            )

        super().__init__(
            policy=policy,
            env=env,
            rollout_buffer_class=rollout_buffer_class,
            **kwargs,
        )


class Graph2GraphOnPolicyAlgorithm(BaseGraphOnPolicyAlgorithm):
    """Base class for on policy algorithm with both observations and actions being graphs."""

    def update_init_args(
        self,
        env: GymEnv,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
    ) -> tuple[type[RolloutBuffer], GymEnv, Optional[tuple[type[spaces.Space], ...]]]:
        # Use proper default rollout buffer class
        if rollout_buffer_class is None:
            if isinstance(env.observation_space, spaces.Graph) and isinstance(
                env.action_space, spaces.Graph
            ):
                rollout_buffer_class = Graph2GraphRolloutBuffer
            else:
                raise ValueError(
                    f"{self.__class__.__name__} is intended to be used with both observations and actions being graphs."
                )

        # Use proper VecEnv wrapper for env with Graph spaces
        env = wrap_graph_env(env)
        if env.num_envs > 1:
            raise NotImplementedError(
                "Graph2GraphOnPolicyAlgorithm not implemented for real vectorized environment "
                "(ie. with more than 1 wrapped environment)"
            )
        supported_action_spaces = (gym.spaces.Graph,)

        return rollout_buffer_class, env, supported_action_spaces

    def actions_from_torch(
        self, actions: thg.data.Data
    ) -> list[gym.spaces.GraphInstance]:
        return thg_data_to_graph_instance(actions, space=self.action_space)
