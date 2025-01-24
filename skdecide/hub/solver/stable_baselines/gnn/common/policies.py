import copy
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch_geometric as thg
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn

from skdecide.hub.solver.utils.gnn.torch_layers import Graph2GraphLayer
from skdecide.hub.solver.utils.gnn.torch_utils import thg_data_to_graph_instance

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


class FullGNNActorCriticPolicy(GNNActorCriticPolicy):
    """Policy mapping graph to graph, without feature reduction.

    Intended to be used with environment having both observations and actions being graphs.

    """

    observation_space: spaces.Graph
    action_space: spaces.Graph

    def __init__(
        self,
        observation_space: spaces.Graph,
        action_space: spaces.Graph,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = 0.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = GraphFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        action_gnn_class: Optional[type[nn.Module]] = None,
        action_gnn_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # Default parameters
        if net_arch is None:
            net_arch = dict(pi=[], vf=[64, 64])
        elif isinstance(net_arch, dict):
            net_arch["pi"] = []
        elif isinstance(net_arch, list):
            net_arch = dict(pi=[], vf=net_arch)
        else:
            raise ValueError("net_arch should be None, dict or list")

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
            normalize_images=normalize_images,
        )

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = True
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        self.pi_features_extractor = self.features_extractor
        self.vf_features_extractor = self.features_extractor
        self.log_std_init = log_std_init
        self.use_sde = False
        self.dist_kwargs = None

        if action_gnn_kwargs is None:
            self.action_gnn_kwargs = {}
        else:
            self.action_gnn_kwargs = action_gnn_kwargs
        self.action_gnn_class = action_gnn_class

        self._build(lr_schedule)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[spaces.GraphInstance, Optional[Tuple[np.ndarray, ...]]]:
        self.set_training_mode(False)
        obs_tensor, vectorized_env = self.obs_to_tensor(observation)
        with th.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic)

        # convert actions to graph instance
        actions = thg_data_to_graph_instance(
            actions, space=self.action_space, vectorized=vectorized_env
        )

        return actions, state

    def _predict(
        self, observation: thg.data.Data, deterministic: bool = False
    ) -> thg.data.Data:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.action_net(observation)

    def forward(
        self, obs: thg.data.Data, deterministic: bool = False
    ) -> Tuple[thg.data.Data, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)
        log_prob = th.zeros_like(values)
        actions = self.action_net(obs)
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        values = self.predict_values(obs)
        log_prob = th.zeros_like(values)
        return values, log_prob, None

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        # action net
        self.action_net = Graph2GraphLayer(
            observation_space=self.observation_space,
            action_space=self.action_space,
            gnn_class=self.action_gnn_class,
            gnn_kwargs=self.action_gnn_kwargs,
        )

        # value net
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: np.sqrt(2),
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]
