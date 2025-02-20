# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, Union

import gymnasium as gym
import numpy as np
import ray
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)
from packaging.version import Version
from ray.rllib import RolloutWorker
from ray.rllib.algorithms import DQN, PPO, SAC
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.multi_agent_env_compatibility import (
    MultiAgentEnvCompatibility,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.from_config import NotProvided
from ray.tune.registry import register_env

from skdecide import Domain, Solver
from skdecide.builders.domain import SingleAgent, UnrestrictedActions
from skdecide.builders.solver import Maskable, Policies, Restorable
from skdecide.core import EnumerableSpace, Mask
from skdecide.domains import MultiAgentRLDomain
from skdecide.hub.domain.gym import AsLegacyGymV21Env
from skdecide.hub.space.gym import GymSpace

from .action_masking.models.tf.parametric_actions import TFParametricActionsModel
from .action_masking.models.torch.masked_actions import TorchMaskedActionsModel
from .action_masking.models.torch.parametric_actions import TorchParametricActionsModel
from .action_masking.utils.spaces.space_utils import (
    ACTION_MASK,
    TRUE_OBS,
    create_agent_action_mask_space,
)
from .gnn.evaluation.rollout_worker import Graph2NodeRolloutWorker, GraphRolloutWorker
from .gnn.models.torch.complex_input_net import GraphComplexInputNetwork
from .gnn.models.torch.gnn import GnnBasedGraph2NodeModel, GnnBasedModel
from .gnn.utils.spaces.space_utils import (
    convert_graph_space_to_dict_space,
    convert_graph_to_dict,
)

logger = logging.getLogger(__name__)


class D(MultiAgentRLDomain):
    pass


class RayRLlib(Solver, Policies, Restorable, Maskable):
    """This class wraps a Ray RLlib solver (ray[rllib]) as a scikit-decide solver.

    !!! warning
        Using this class requires Ray RLlib to be installed.
    """

    T_domain = D

    hyperparameters = [
        IntegerHyperparameter(name="train_iterations", low=1, high=3),
        CategoricalHyperparameter(
            name="algo_class",
            choices={
                "PPO": PPO,
                "DQN": DQN,
                "SAC": SAC,
            },
        ),
        FloatHyperparameter(
            name="gamma_complement_log",  # gamma = 1 - 10 ** gamma_complement_log
            low=-3,
            high=-1,
            step=1,
        ),
        FloatHyperparameter(
            name="lr_log",  # lr = 10 ** lr_log
            low=-5,
            high=-1,
            step=1,
        ),
        IntegerHyperparameter(
            name="train_batch_size_log2",  # train_batch_size = 2 ** train_batch_size_log2
            low=4,
            high=8,
            depends_on=("algo_class", [DQN, SAC]),
        ),
        IntegerHyperparameter(
            name="sgd_minibatch_size_log2",  # sgd_minibatch_size = 2 ** sgd_minibatch_size_log2
            low=4,
            high=8,
            depends_on=("algo_class", [PPO]),
        ),
        FloatHyperparameter(
            name="entropy_coeff_log",  # entropy_coeff = 10 ** entropy_coeff_log
            low=-3,
            high=-1,
            step=1,
            depends_on=("algo_class", [PPO]),
        ),
    ]

    MASKABLE_ALGOS = ["APPO", "BC", "DQN", "Rainbow", "IMPALA", "MARWIL", "PPO"]
    """The only algos being able to handle action masking in ray[rllib]==2.9.0."""

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        algo_class: type[Algorithm],
        train_iterations: int,
        config: Optional[AlgorithmConfig] = None,
        policy_configs: Optional[dict[str, dict]] = None,
        policy_mapping_fn: Optional[
            Callable[[str, Optional["EpisodeV2"], Optional["RolloutWorker"]], str]
        ] = None,
        action_embed_sizes: Optional[dict[str, int]] = None,
        callback: Callable[[RayRLlib], bool] = lambda solver: False,
        graph_feature_extractors_kwargs: Optional[dict[str, Any]] = None,
        graph_node_action: bool = False,
        graph_node_action_net_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize Ray RLlib.

        # Parameters
        domain_factory: A callable with no argument returning the domain to solve (can be a mere domain class).
            The resulting domain will be auto-cast to the level expected by the solver.
        algo_class: The class of Ray RLlib trainer/agent to wrap.
        train_iterations: The number of iterations to call the trainer's train() method.
        config: The configuration dictionary for the trainer.
        policy_configs: The mapping from policy id (str) to additional config (dict) (leave default for single policy).
        policy_mapping_fn: The function mapping agent ids to policy ids (leave default for single policy).
        action_embed_sizes: The mapping from policy id (str) to action embedding size (only used with domains filtering allowed actions per state, default to 2)
        callback: function called at each solver iteration.
            If returning true, the solve process stops and exit the current train iteration.
            However, if train_iterations > 1, another train loop will be entered after that.
            (One can code its callback in such a way that further training loop are stopped directly after that.)
        graph_feature_extractors_kwargs: in case of graph observations, these are the kwargs to the `GraphFeaturesExtractor` model
            used to extract features. See `skdecide.hub.solver.utils.gnn.torch_layers.GraphFeaturesExtractor`.
        graph_node_action: specify if actions are actually nodes of the observation graph. In that case, the actual action space
            size is derived at runtime from the observation graph.
        graph_node_action_net_kwargs: if graph_node_action, these are the kwargs to the `Graph2NodeLayer` model used to
            predict actions from the observation. See `skdecide.hub.solver.utils.gnn.torch_layers.Graph2NodeLayer`.
        **kwargs: used to update the algo config with kwargs automatically filled by optuna.

        #### Masking

        If the domain has not the `UnrestrictedActions` mixin, and if the algo used allows action masking
        (e.g. APPO, BC, DQN, Rainbow, IMPALA, MARWIL, PPO), the observations are automatically wrapped to also present
        the action mask to the algorithm, which will used via a custom model
        (defined in `skdecide.hub.solver.ray_rllib.action_masking.models`).
        During training, a gymnasium environment is created wrapping a domain instantiated from `domain_factory` and
        used during training rollouts to get the observation with the appropriate action mask.
        At inference, we use the method `self.get_action_mask()` which provides the proper action mask provided that
        `self.retrieve_applicable_actions(domain)` has been called before hand with the domain instance used for the inference.
        This is automatically done by `skdecide.utils.rollout()`.

        #### Graph observations

        If the observation space wrapped gymnasium space for each agent is a `gymnasium.spaces.Graph` or a `gymnasium.spaces.Dict`
        whose subspaces contain a `gymnasium.spaces.Graph`, the solver will use custom models adapted to graphs for its policy, using GNNs.

        - If `graph_node_action` is False (default), a GNN will be used to extract (a fixed number of) features from graphs,
          and then classical MLPs will be use for predicting action and value.
          See `skdecide.hub.solver.utils.gnn.torch_layers.GraphFeaturesExtractor` for more details
          and use `graph_feature_extractors_kwargs` to customize it.
        - If `graph_node_action` is True, this means that an agent action is defined by the choice of node in the observation graph.
          The agent action space should wrap a `gymnasium.spaces.Discrete` even though the actual number of actions will be
          derived at runtime from the number of nodes in the observation graph. The agent observation wrapped gymnasium space
          can only be a `gymnasium.spaces.Graph` in that case.
          The value is still predicted as above via a GNN features extractor + classical MLP, customized with same parameters.
          The action logits will be directly predicted via another GNN so that the number of logits correspond to the number
          of nodes.
          See `skdecide.hub.solver.utils.gnn.torch_layers.Graph2NodeLayer` for more details
          and use `graph_node_action_net_kwargs` to customize it.

        """
        Solver.__init__(self, domain_factory=domain_factory)
        self.callback = callback
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
        if graph_feature_extractors_kwargs is None:
            self._graph_feature_extractors_kwargs = {}
        else:
            self._graph_feature_extractors_kwargs = graph_feature_extractors_kwargs
        if graph_node_action_net_kwargs is None:
            self._graph2node_action_net_kwargs = {}
        else:
            self._graph2node_action_net_kwargs = graph_node_action_net_kwargs

        ray.init(ignore_reinit_error=True)
        self._algo_callbacks: Optional[DefaultCallbacks] = None
        self._algo_worker_callbacks: Optional[DefaultCallbacks] = None
        self._algo_evaluation_worker_callbacks: Optional[DefaultCallbacks] = None

        # wrapped action space and observation space
        domain = self._domain_factory()
        self._wrapped_action_space = domain.get_action_space()
        self._wrapped_observation_space = domain.get_observation_space()

        # action masking?
        self._action_masking = (
            (not isinstance(domain, UnrestrictedActions))
            and all(
                isinstance(agent_action_space, EnumerableSpace)
                for agent_action_space in self._wrapped_action_space.values()
            )
            and (
                self._algo_class.__name__ in RayRLlib.MASKABLE_ALGOS
                or self._algo_class.__name__
                in [f"Graph{algo_name}" for algo_name in RayRLlib.MASKABLE_ALGOS]
            )
        )

        # graph obs?
        self._is_graph_obs = _is_multiagent_graph_space(self._wrapped_observation_space)
        self._is_graph_multiinput_obs = _is_multiagent_graph_multiinput_space(
            (self._wrapped_observation_space)
        )

        # graph -> node (ie an action is a node of observation graph)
        self._graph2node = self._is_graph_obs and graph_node_action

        # Handle kwargs (potentially generated by optuna)
        if "train_batch_size_log2" in kwargs:
            # train_batch_size
            train_batch_size_log2 = kwargs.pop("train_batch_size_log2")
            kwargs["train_batch_size"] = 2**train_batch_size_log2
        if "sgd_minibatch_size_log2" in kwargs:
            # sgd_minibatch_size
            sgd_minibatch_size_log2 = kwargs.pop("sgd_minibatch_size_log2")
            kwargs["sgd_minibatch_size"] = 2**sgd_minibatch_size_log2
        if "gamma_complement_log" in kwargs:
            # gamma
            gamma_complement_log = kwargs.pop("gamma_complement_log")
            kwargs["gamma"] = 1 - 10**gamma_complement_log
        if "lr_log" in kwargs:
            # lr
            lr_log = kwargs.pop("lr_log")
            kwargs["lr"] = 10**lr_log
        if "entropy_coeff_log" in kwargs:
            # entropy_coeff
            entropy_coeff_log = kwargs.pop("entropy_coeff_log")
            kwargs["entropy_coeff"] = 10**entropy_coeff_log

        # Update algorithm config with hyperparameters found in kwargs
        if kwargs:
            self._config.update_from_dict(kwargs)

    def using_applicable_actions(self):
        return self._action_masking

    def get_policy(self) -> dict[str, Policy]:
        """Return the computed policy."""
        return {
            policy_id: self._algo.get_policy(policy_id=policy_id)
            for policy_id in self._policy_configs
        }

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

    def _solve(self) -> None:
        # Reuse algo if possible (enables further learning)
        if not hasattr(self, "_algo"):
            self._init_algo()

        # Training loop
        for _ in range(self._train_iterations):
            try:
                self._algo.train()
            except SolveEarlyStop as e:
                # if stopping exception raise, we choose to stop this train iteration
                pass

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
        self.forget_callback()  # avoid serializing issues
        self._algo.save(path)
        self.set_callback()  # put it back in case of further solve

    def _load(self, path: str):
        self._init_algo()
        self._algo.restore(path)
        self.set_callback()  # ensure putting back actual callback

    def _init_algo(self) -> None:
        # monkey patch rllib for graph handling
        if self._is_graph_obs or self._is_graph_multiinput_obs:
            if self._graph2node:
                if not isinstance(
                    self._config.env_runner_cls,
                    (RolloutWorker, Graph2NodeRolloutWorker),
                ):
                    logger.warning(
                        "The EnvRunner class to use for environment rollouts (data collection) will be overriden "
                        "by Graph2NodeRolloutWorker so that buffers manage properly graphs concatenation."
                    )
                self._config.env_runners(env_runner_cls=Graph2NodeRolloutWorker)
            else:
                if not isinstance(
                    self._config.env_runner_cls, (RolloutWorker, GraphRolloutWorker)
                ):
                    logger.warning(
                        "The EnvRunner class to use for environment rollouts (data collection) will be overriden "
                        "by GraphRolloutWorker so that buffers manage properly graphs concatenation."
                    )
                self._config.env_runners(env_runner_cls=GraphRolloutWorker)

        # custom model?
        if self._action_masking:
            if self._is_graph_obs or self._is_graph_multiinput_obs:
                # let the observation pass as is
                self._config.experimental(
                    _disable_preprocessor_api=True,
                )
                if self._config.get("framework") not in ["torch"]:
                    raise RuntimeError(
                        "Graph observation with RLlib requires PyTorch framework."
                    )
            if self._config.get("framework") not in ["tf", "tf2", "torch"]:
                raise RuntimeError(
                    "Action masking (invalid action filtering) for RLlib requires TensorFlow or PyTorch to be installed"
                )
            if self._graph2node:
                ModelCatalog.register_custom_model(
                    "skdecide_rllib_custom_model",
                    TorchMaskedActionsModel
                    if self._config.get("framework") == "torch"
                    else NotProvided,
                )
            else:
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
        elif self._is_graph_obs:
            if self._config.get("framework") not in ["torch"]:
                raise RuntimeError(
                    "Graph observation with RLlib requires PyTorch framework."
                )
            if self._graph2node:
                ModelCatalog.register_custom_model(
                    "skdecide_rllib_graph_model",
                    GnnBasedGraph2NodeModel
                    if self._config.get("framework") == "torch"
                    else NotProvided,
                )
            else:
                ModelCatalog.register_custom_model(
                    "skdecide_rllib_graph_model",
                    GnnBasedModel
                    if self._config.get("framework") == "torch"
                    else NotProvided,
                )
            # let the observation pass as is
            self._config.experimental(
                _disable_preprocessor_api=True,
            )
        elif self._is_graph_multiinput_obs:
            if self._config.get("framework") not in ["torch"]:
                raise RuntimeError(
                    "Graph observation with RLlib requires PyTorch framework."
                )
            ModelCatalog.register_custom_model(
                "skdecide_rllib_graph_multiinput_model",
                GraphComplexInputNetwork
                if self._config.get("framework") == "torch"
                else NotProvided,
            )
            # let the observation pass as is
            self._config.experimental(
                _disable_preprocessor_api=True,
            )

        self._wrap_action = lambda action: _wrap_action(
            action=action, wrapped_action_space=self._wrapped_action_space
        )
        # Trick to assign o's unwrapped value to self._unwrap_obs
        # (no unwrapping method for single elements in enumerable spaces)
        if self._action_masking:
            self._unwrap_obs = lambda obs, agent: _unwrap_agent_obs_with_action_masking(
                obs=obs,
                agent=agent,
                wrapped_observation_space=self._wrapped_observation_space,
                action_mask=self.get_action_mask(),
            )
        else:
            self._unwrap_obs = lambda obs, agent: _unwrap_agent_obs(
                obs=obs,
                agent=agent,
                wrapped_observation_space=self._wrapped_observation_space,
            )

        # Overwrite multi-agent config
        pol_obs_spaces = {
            self._policy_mapping_fn(
                agent, None, None
            ): _create_agent_obs_space_for_rllib(
                wrapped_observation_space=self._wrapped_observation_space,
                wrapped_action_space=self._wrapped_action_space,
                agent=agent,
                action_masking=self._action_masking,
                graph2node=self._graph2node,
            )
            for agent in self._wrapped_observation_space
        }

        pol_act_spaces = {
            self._policy_mapping_fn(k, None, None): agent_action_space.unwrapped()
            for k, agent_action_space in self._wrapped_action_space.items()
        }

        if self._action_masking:
            if self._is_graph_obs or self._is_graph_multiinput_obs:
                extra_custom_model_config_kwargs = {
                    "features_extractor": self._graph_feature_extractors_kwargs,
                    "graph2node": self._graph2node,
                }
            else:
                extra_custom_model_config_kwargs = {}
            policies = {
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
                                        TRUE_OBS
                                    ],
                                    "action_embed_size": action_embed_size,
                                    **extra_custom_model_config_kwargs,
                                },
                            },
                        },
                    },
                )
                for k, action_embed_size in self._action_embed_sizes.items()
            }
        elif self._is_graph_obs:
            policies = {
                self._policy_mapping_fn(k, None, None): (
                    None,
                    pol_obs_spaces[k],
                    pol_act_spaces[k],
                    {
                        **(v or {}),
                        **{
                            "model": {
                                "custom_model": "skdecide_rllib_graph_model",
                                "custom_model_config": {
                                    "features_extractor": self._graph_feature_extractors_kwargs,  # kwargs for GraphFeaturesExtractor
                                },
                            },
                        },
                    },
                )
                for k, v in self._policy_configs.items()
            }
        elif self._is_graph_multiinput_obs:
            policies = {
                self._policy_mapping_fn(k, None, None): (
                    None,
                    pol_obs_spaces[k],
                    pol_act_spaces[k],
                    {
                        **(v or {}),
                        **{
                            "model": {
                                "custom_model": "skdecide_rllib_graph_multiinput_model",
                                "custom_model_config": {
                                    "features_extractor": self._graph_feature_extractors_kwargs,
                                    # kwargs for GraphFeaturesExtractor
                                },
                            },
                        },
                    },
                )
                for k, v in self._policy_configs.items()
            }
        else:
            policies = {
                k: (None, pol_obs_spaces[k], pol_act_spaces[k], v or {})
                for k, v in self._policy_configs.items()
            }
        self._config.multi_agent(
            policies=policies,
            policy_mapping_fn=self._policy_mapping_fn,
        )

        register_env(
            "skdecide_env",
            lambda _, domain_factory=self._domain_factory, rayrllib=self: AsRLlibMultiAgentEnv(
                domain=domain_factory(),
                action_masking=rayrllib._action_masking,
                graph2node=rayrllib._graph2node,
            ),
        )
        if Version(ray.__version__) >= Version("2.20.0"):
            # starting from ray 2.20, no more checks on environment are made,
            # and `disable_env_checking` use raises an error
            self._config.environment(env="skdecide_env")
        else:
            # Disable env checking in case of action masking otherwise RLlib will try to simulate
            # next state transition with invalid actions, which might make some domains crash if
            # they require action masking
            self._config.environment(
                env="skdecide_env", disable_env_checking=self._action_masking
            )

        # set callback class for algo config
        self.set_callback()

        # Instantiate algo
        self._algo = self._algo_class(config=self._config)

    def set_callback(self):
        """Set back callback.

        Useful to do it after serializing/deserializing because of potential issues with
        - lambda functions
        - dynamic classes

        """
        # generate specific callback class
        callbacks_class = generate_rllibcallback_class(
            callback=self.callback, solver=self
        )
        # use it in all algo config, and callbacks attributes
        self._set_callbackclass(callbacks_class=callbacks_class)

    def forget_callback(self):
        """Forget about actual callback to avoid serializing issues."""
        # use default callback class
        callbacks_class = DefaultCallbacks
        # use it in algo config & evaluation_config, worker config, and for algo.callbacks, worker.callbacks
        self._set_callbackclass(callbacks_class=callbacks_class)

    def _set_callbackclass(self, callbacks_class: type[DefaultCallbacks]):
        _set_callbackclass_in_config(
            callbacks_class=callbacks_class, config=self._config
        )
        if hasattr(self, "_algo"):
            tmp = self._algo.callbacks
            if (
                self._algo_callbacks
                and self._algo_callbacks.__class__ is callbacks_class
            ):
                self._algo.callbacks = self._algo_callbacks
            else:
                self._algo.callbacks = callbacks_class()
            self._algo_callbacks = tmp
            if self._algo.evaluation_config:
                _set_callbackclass_in_config(
                    callbacks_class=callbacks_class, config=self._algo.evaluation_config
                )
            if self._algo.workers:
                if Version(ray.__version__) < Version("2.34.0"):
                    # starting from 2.34, algo.workers becomes algo.env_runner_group
                    local_worker: RolloutWorker = self._algo.workers.local_worker()
                else:
                    local_worker: RolloutWorker = (
                        self._algo.env_runner_group.local_worker()
                    )
                if local_worker:
                    _set_callbackclass_in_config(
                        callbacks_class=callbacks_class, config=local_worker.config
                    )
                    self._algo_worker_callbacks = _swap_callbacks(
                        algo_or_worker=local_worker,
                        previous_callbacks=self._algo_worker_callbacks,
                        callbacks_class=callbacks_class,
                    )
                    for pid, policy in local_worker.policy_map.items():
                        policy.config["callbacks"] = callbacks_class


def _set_callbackclass_in_config(
    callbacks_class: type[DefaultCallbacks], config: AlgorithmConfig
) -> None:
    is_frozen = config._is_frozen
    if is_frozen:
        # allow callbacks update
        config._is_frozen = False
    config.callbacks(callbacks_class=callbacks_class)
    config._is_frozen = is_frozen


def _swap_callbacks(
    algo_or_worker: Union[Algorithm, RolloutWorker, Policy],
    previous_callbacks,
    callbacks_class,
):
    tmp = algo_or_worker.callbacks
    if previous_callbacks and previous_callbacks.__class__ is callbacks_class:
        algo_or_worker.callbacks = previous_callbacks
    else:
        algo_or_worker.callbacks = callbacks_class()
    previous_callbacks = tmp
    return previous_callbacks


class AsRLlibMultiAgentEnv(MultiAgentEnvCompatibility):
    def __init__(
        self,
        domain: D,
        action_masking: bool = False,
        graph2node: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        old_env = AsLegacyRLlibMultiAgentEnv(
            domain=domain, action_masking=action_masking, graph2node=graph2node
        )
        self._domain = domain
        super().__init__(old_env=old_env, render_mode=render_mode)

    def get_agent_ids(self) -> set[str]:
        return self._domain.get_agents()


class AsLegacyRLlibMultiAgentEnv(AsLegacyGymV21Env):
    def __init__(
        self,
        domain: D,
        action_masking: bool,
        graph2node: bool = False,
    ) -> None:
        """Initialize AsLegacyRLlibMultiAgentEnv.

        # Parameters
        domain: The scikit-decide domain to wrap as a RLlib multi-agent environment.
        action_masking: Boolean specifying whether action masking is used
        """
        self._graph2node = graph2node
        self._domain = domain
        self._action_masking = action_masking
        self._wrapped_action_space = domain.get_action_space()
        self._wrapped_observation_space = domain.get_observation_space()

        self.observation_space = gym.spaces.Dict(
            {
                agent: _create_agent_obs_space_for_rllib(
                    wrapped_observation_space=self._wrapped_observation_space,
                    wrapped_action_space=self._wrapped_action_space,
                    agent=agent,
                    action_masking=self._action_masking,
                    graph2node=self._graph2node,
                )
                for agent in self._wrapped_observation_space
            }
        )

        self.action_space = gym.spaces.Dict(
            {
                k: agent_action_space.unwrapped()
                for k, agent_action_space in self._wrapped_action_space.items()
            }
        )

    def _wrap_action(self, action_dict: dict[str, Any]) -> dict[str, D.T_event]:
        return _wrap_action(
            action=action_dict, wrapped_action_space=self._wrapped_action_space
        )

    def _unwrap_obs(self, obs: dict[str, D.T_observation]) -> dict[str, Any]:
        if not self._action_masking:
            return {
                agent: _unwrap_agent_obs(
                    obs=obs,
                    agent=agent,
                    wrapped_observation_space=self._wrapped_observation_space,
                )
                for agent in obs
            }
        else:
            return {
                agent: _unwrap_agent_obs_with_action_masking(
                    obs=obs,
                    agent=agent,
                    wrapped_observation_space=self._wrapped_observation_space,
                    action_mask=self._domain.get_action_mask(),
                )
                for agent in obs
            }

    def reset(self):
        """Resets the env and returns observations from ready agents.

        # Returns
        obs (dict): New observations for each ready agent.
        """
        raw_observation = self._domain.reset()
        return self._unwrap_obs(raw_observation)

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
        action = self._wrap_action(action_dict)
        outcome = self._domain.step(action)
        observations = self._unwrap_obs(outcome.observation)
        rewards = {k: v.reward for k, v in outcome.value.items()}
        done = outcome.termination
        done.update({"__all__": all(outcome.termination.values())})
        infos = {k: (v or {}) for k, v in outcome.info.items()}
        return observations, rewards, done, infos


class BaseRLlibCallback(DefaultCallbacks):
    callback: _CallbackWrapper
    solver: RayRLlib

    def on_episode_step(
        self,
        *args,
        **kwargs,
    ) -> None:
        stopping = self.callback(self.solver)
        if stopping:
            raise SolveEarlyStop("Solve process stopped by user callback")


class _CallbackWrapper:
    """Wrapper to avoid surprises with lambda functions"""

    def __init__(self, callback: Callable[[RayRLlib], bool]):
        self.callback = callback

    def __call__(self, solver) -> bool:
        return self.callback(solver)


class SolveEarlyStop(Exception):
    """Exception raised if a callback tells to stop the solve process."""


def generate_rllibcallback_class(
    callback: _CallbackWrapper, solver: RayRLlib, classname=None
) -> type[BaseRLlibCallback]:
    if classname is None:
        classname = f"MyCallbackClass{id(solver)}"
    return type(
        classname,
        (BaseRLlibCallback,),
        dict(solver=solver, callback=_CallbackWrapper(callback=callback)),
    )


def _unwrap_agent_obs_space(
    wrapped_observation_space: dict[str, GymSpace[D.T_observation]],
    agent: str,
) -> gym.Space:
    unwrapped_agent_obs_space = wrapped_observation_space[agent].unwrapped()
    if isinstance(unwrapped_agent_obs_space, gym.spaces.Graph):
        return convert_graph_space_to_dict_space(unwrapped_agent_obs_space)
    elif _is_graph_multiinput_unwrapped_agent_space(unwrapped_agent_obs_space):
        return gym.spaces.Dict(
            {
                k: convert_graph_space_to_dict_space(subspace)
                if isinstance(subspace, gym.spaces.Graph)
                else subspace
                for k, subspace in unwrapped_agent_obs_space.spaces.items()
            }
        )
    else:
        return unwrapped_agent_obs_space


def _create_agent_obs_space_for_rllib(
    wrapped_observation_space: dict[str, GymSpace[D.T_observation]],
    wrapped_action_space: dict[str, EnumerableSpace[D.T_event]],
    agent: str,
    action_masking: bool,
    graph2node: bool,
) -> gym.spaces.Space:
    true_observation_space = _unwrap_agent_obs_space(
        wrapped_observation_space=wrapped_observation_space,
        agent=agent,
    )
    if action_masking:
        return gym.spaces.Dict(
            {
                TRUE_OBS: true_observation_space,
                ACTION_MASK: create_agent_action_mask_space(
                    agent_action_space=wrapped_action_space[agent],
                    graph2node=graph2node,
                ),
            }
        )
    else:
        return true_observation_space


def _unwrap_agent_obs(
    obs: dict[str, D.T_observation],
    agent: str,
    wrapped_observation_space: dict[str, GymSpace[D.T_observation]],
    transform_graph: bool = True,
) -> Any:
    unwrapped_agent_obs_space = wrapped_observation_space[agent].unwrapped()
    if isinstance(unwrapped_agent_obs_space, gym.spaces.Graph) and transform_graph:
        # get original unwrapped graph instance
        unwrapped_agent_obs: gym.spaces.GraphInstance = _unwrap_agent_obs(
            obs=obs,
            agent=agent,
            wrapped_observation_space=wrapped_observation_space,
            transform_graph=False,
        )
        # transform graph instance into a dict
        return convert_graph_to_dict(unwrapped_agent_obs)
    elif (
        _is_graph_multiinput_unwrapped_agent_space((unwrapped_agent_obs_space))
        and transform_graph
    ):
        unwrapped_agent_obs: dict[str, Any] = _unwrap_agent_obs(
            obs=obs,
            agent=agent,
            wrapped_observation_space=wrapped_observation_space,
            transform_graph=False,
        )
        return {
            k: convert_graph_to_dict(v)
            if isinstance(v, gym.spaces.GraphInstance)
            else v
            for k, v in unwrapped_agent_obs.items()
        }
    else:
        # Trick to get obs[agent]'s unwrapped value
        # (no unwrapping method for single elements in enumerable spaces)
        return next(iter(wrapped_observation_space[agent].to_unwrapped([obs[agent]])))


def _unwrap_agent_obs_with_action_masking(
    obs: dict[str, D.T_observation],
    agent: str,
    wrapped_observation_space: dict[str, GymSpace[D.T_observation]],
    action_mask: dict[str, Mask],
) -> dict[str, Union[Any, Mask]]:
    return {
        TRUE_OBS: _unwrap_agent_obs(
            obs=obs, agent=agent, wrapped_observation_space=wrapped_observation_space
        ),
        ACTION_MASK: action_mask[agent],
    }


def _wrap_action(
    action: dict[str, Any], wrapped_action_space: dict[str, GymSpace[D.T_event]]
) -> dict[str, D.T_event]:
    return {
        # Trick to get unwrapped_action's wrapped value
        # (no wrapping method for single unwrapped values in enumerable spaces)
        agent: next(
            iter(wrapped_action_space[agent].from_unwrapped([unwrapped_action]))
        )
        for agent, unwrapped_action in action.items()
    }


def _is_multiagent_graph_space(space: dict[str, GymSpace[Any]]) -> bool:
    return all(
        isinstance(agent_observation_space.unwrapped(), gym.spaces.Graph)
        for agent_observation_space in space.values()
    )


def _is_multiagent_graph_multiinput_space(space: dict[str, GymSpace[Any]]) -> bool:
    return all(
        _is_graph_multiinput_unwrapped_agent_space(agent_observation_space.unwrapped())
        for agent_observation_space in space.values()
    )


def _is_graph_multiinput_unwrapped_agent_space(space: gym.spaces.Space) -> bool:
    return isinstance(space, gym.spaces.Dict) and any(
        isinstance(subspace, gym.spaces.Graph) for subspace in space.spaces.values()
    )
