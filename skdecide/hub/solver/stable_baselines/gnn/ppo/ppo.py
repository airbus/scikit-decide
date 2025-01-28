from typing import Any, ClassVar, Dict, Optional, Type, Union

import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from ..common.on_policy_algorithm import (
    Graph2GraphOnPolicyAlgorithm,
    GraphOnPolicyAlgorithm,
)
from ..common.policies import (
    FullGNNActorCriticPolicy,
    GNNActorCriticPolicy,
    MultiInputGNNActorCriticPolicy,
)


class GraphPPO(GraphOnPolicyAlgorithm, PPO):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "GraphInputPolicy": GNNActorCriticPolicy,
        "MultiInputPolicy": MultiInputGNNActorCriticPolicy,
    }


class Graph2GraphPPO(Graph2GraphOnPolicyAlgorithm, PPO):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "GraphInputPolicy": FullGNNActorCriticPolicy,
        # "MultiInputPolicy": MultiInputGNNActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        rollout_buffer_class, env, supported_action_spaces = self.update_init_args(
            rollout_buffer_class=rollout_buffer_class,
            env=env,
        )

        super(PPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=supported_action_spaces,
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()
