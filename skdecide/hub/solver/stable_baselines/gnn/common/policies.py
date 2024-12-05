import warnings
from typing import Any, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch_geometric as thg
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from .torch_layers import GraphFeaturesExtractor
from .utils import graph_obs_to_thg_data

PyTorchGraphObs = Union[thg.data.Data, list[thg.data.Data]]


class _BaseGNNActorCriticPolicy(BasePolicy):
    def extract_features(
        self,
        obs: thg.data.Data,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            if features_extractor is None:
                features_extractor = self.features_extractor
            return features_extractor(obs)
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = self.pi_features_extractor(obs)
            vf_features = self.vf_features_extractor(obs)
            return pi_features, vf_features

    def obs_to_tensor(
        self, observation: gym.spaces.GraphInstance
    ) -> tuple[PyTorchGraphObs, bool]:
        if isinstance(observation, list):
            vectorized_env = True
        else:
            vectorized_env = False
        if vectorized_env:
            torch_obs = [
                graph_obs_to_thg_data(obs, device=self.device) for obs in observation
            ]
            if len(torch_obs) == 1:
                torch_obs = torch_obs[0]
        else:
            torch_obs = graph_obs_to_thg_data(observation, device=self.device)
        return torch_obs, vectorized_env

    def get_distribution(self, obs: thg.data.Data) -> Distribution:
        features = self.pi_features_extractor(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: thg.data.Data) -> th.Tensor:
        features = self.vf_features_extractor(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)


class GNNActorCriticPolicy(_BaseGNNActorCriticPolicy, ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Graph,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[list[Union[int, dict[str, list[int]]]]] = None,
        activation_fn: type[th.nn.Module] = th.nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = GraphFeaturesExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


class MaskableGNNActorCriticPolicy(
    _BaseGNNActorCriticPolicy, MaskableActorCriticPolicy
):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[th.nn.Module] = th.nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: type[BaseFeaturesExtractor] = GraphFeaturesExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def get_distribution(
        self, obs: thg.data.Data, action_masks: Optional[np.ndarray] = None
    ) -> MaskableDistribution:
        features = self.pi_features_extractor(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution
