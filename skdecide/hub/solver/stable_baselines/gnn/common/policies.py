import copy
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch_geometric as thg
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from .preprocessing import preprocess_obs
from .torch_layers import CombinedFeaturesExtractor, GraphFeaturesExtractor
from .utils import ObsType, TorchObsType, is_vectorized_observation, obs_as_tensor


class BaseGNNPolicy(BasePolicy):
    def extract_features(
        self, obs: thg.data.Data, features_extractor: BaseFeaturesExtractor
    ) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use.
        :return: The extracted features
        """
        preprocessed_obs = preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )
        return features_extractor(preprocessed_obs)

    def obs_to_tensor(self, observation: ObsType) -> tuple[TorchObsType, bool]:
        vectorized_env = False
        if isinstance(self.observation_space, gym.spaces.Graph):
            vectorized_env = is_vectorized_observation(
                observation, self.observation_space
            )
        elif isinstance(observation, dict):
            assert isinstance(
                self.observation_space, gym.spaces.Dict
            ), f"The observation provided is a dict but the obs space is {self.observation_space}"
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if isinstance(obs_space, gym.spaces.Graph):
                    vectorized_env = vectorized_env or is_vectorized_observation(
                        obs, obs_space
                    )
                else:
                    if is_image_space(obs_space):
                        obs_ = maybe_transpose(obs, obs_space)
                    else:
                        obs_ = np.array(obs)
                    vectorized_env = vectorized_env or is_vectorized_observation(
                        obs_, obs_space
                    )
                    # Add batch dimension if needed
                    observation[key] = obs_.reshape((-1, *self.observation_space[key].shape))  # type: ignore[misc]
        else:
            return super().obs_to_tensor(observation)

        obs_tensor = obs_as_tensor(observation, self.device)
        return obs_tensor, vectorized_env

    def is_vectorized_observation(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> bool:
        vectorized_env = False
        if isinstance(observation, dict):
            assert isinstance(
                self.observation_space, gym.spaces.Dict
            ), f"The observation provided is a dict but the obs space is {self.observation_space}"
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                vectorized_env = vectorized_env or is_vectorized_observation(
                    maybe_transpose(obs, obs_space), obs_space
                )
        else:
            vectorized_env = is_vectorized_observation(
                maybe_transpose(observation, self.observation_space),
                self.observation_space,
            )
        return vectorized_env


class BaseGNNActorCriticPolicy(BaseGNNPolicy):
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
        preprocessed_obs = preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )
        if self.share_features_extractor:
            if features_extractor is None:
                features_extractor = self.features_extractor
            return features_extractor(preprocessed_obs)
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = self.pi_features_extractor(preprocessed_obs)
            vf_features = self.vf_features_extractor(preprocessed_obs)
            return pi_features, vf_features

    def get_distribution(self, obs: thg.data.Data) -> Distribution:
        preprocessed_obs = preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )
        features = self.pi_features_extractor(preprocessed_obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: thg.data.Data) -> th.Tensor:
        preprocessed_obs = preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )
        features = self.vf_features_extractor(preprocessed_obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)


class GNNActorCriticPolicy(BaseGNNActorCriticPolicy, ActorCriticPolicy):
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


class MultiInputGNNActorCriticPolicy(GNNActorCriticPolicy):
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
        features_extractor_class: type[
            BaseFeaturesExtractor
        ] = CombinedFeaturesExtractor,
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


class MaskableGNNActorCriticPolicy(BaseGNNActorCriticPolicy, MaskableActorCriticPolicy):
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
        preprocessed_obs = preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )
        features = self.pi_features_extractor(preprocessed_obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution


class MaskableMultiInputGNNActorCriticPolicy(MaskableGNNActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[th.nn.Module] = th.nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: type[
            BaseFeaturesExtractor
        ] = CombinedFeaturesExtractor,
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
