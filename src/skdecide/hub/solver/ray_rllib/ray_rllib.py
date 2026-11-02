# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)
from ray.rllib import RolloutWorker
from ray.rllib.algorithms import DQN, PPO, SAC
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.rl_module import (
    MultiRLModule,
    MultiRLModuleSpec,
    RLModule,
    RLModuleSpec,
)
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.env.wrappers.multi_agent_env_compatibility import (
    MultiAgentEnvCompatibility,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import ModuleID
from ray.tune.registry import register_env

from skdecide import Domain, Solver, StrDict
from skdecide.builders.domain import SingleAgent, UnrestrictedActions
from skdecide.builders.solver import Policies, Restorable
from skdecide.core import EnumerableSpace, Mask, autocast
from skdecide.domains import MultiAgentRLDomain
from skdecide.hub.domain.gym import AsLegacyGymV21Env
from skdecide.hub.solver.ray_rllib.action_masking.algorithms.dqn.torch.action_masking_dqn_torch_rl_module import (
    ActionMaskingDQNTorchRLModule,
)
from skdecide.hub.solver.ray_rllib.action_masking.algorithms.ppo.torch.action_masking_graph2node_ppo_rl_module import (
    ActionMaskingGraph2NodePPOTorchRLModule,
)
from skdecide.hub.solver.ray_rllib.action_masking.algorithms.ppo.torch.action_masking_ppo_torch_rl_module import (
    ActionMaskingPPOTorchRLModule,
)
from skdecide.hub.solver.ray_rllib.action_masking.connectors.flatten_observations import (
    FlattenMultiagentMaskedObservations,
)
from skdecide.hub.solver.ray_rllib.action_masking.utils.spaces.space_utils import (
    ACTION_MASK,
    TRUE_OBS,
    create_agent_action_mask_space,
)
from skdecide.hub.solver.ray_rllib.autoregressive.algorithms.ppo.ppo_catalog import (
    AutoregressiveGraphPPOCatalog,
    AutoregressivePPOCatalog,
)
from skdecide.hub.solver.ray_rllib.autoregressive.algorithms.ppo.torch.autoregressive_ppo_torch_rl_module import (
    AutoregressivePPOTorchRLModule,
)
from skdecide.hub.solver.ray_rllib.autoregressive.connectors.flatten_observations import (
    FlattenMultiagentAutoregressiveActionObservations,
)
from skdecide.hub.solver.ray_rllib.autoregressive.connectors.pad_action_dist_inputs import (
    PadAutoregressiveGraph2NodeActionDistributionInputs,
)
from skdecide.hub.solver.ray_rllib.autoregressive.utils.spaces.space_utils import (
    APPLICABLE_ACTIONS,
    create_agent_applicable_actions_space,
    create_agent_autoregressive_action_space,
)
from skdecide.hub.solver.ray_rllib.autoregressive.utils.spaces.space_utils import (
    TRUE_OBS as AUTOREGRESSIVE_TRUE_OBS,
)
from skdecide.hub.solver.ray_rllib.common.connectors.flatten_observations import (
    FlattenMultiagentObservations,
)
from skdecide.hub.solver.ray_rllib.common.constants import TORCH_FRAMEWORK
from skdecide.hub.solver.ray_rllib.common.env.multi_agent_env_runner import (
    AutoregressiveGraph2NodeMultiAgentEnvRunner,
    AutoregressiveMultiAgentEnvRunner,
    Graph2NodeMultiAgentEnvRunner,
    SkdecideMultiAgentEnvRunner,
)
from skdecide.hub.solver.ray_rllib.common.utils.monkey_patch import (
    unmonkey_patch_rllib_for_skdecide,
)
from skdecide.hub.solver.ray_rllib.gnn.algorithms.ppo.ppo_catalog import (
    Graph2NodePPOCatalog,
    GraphPPOCatalog,
)
from skdecide.hub.solver.ray_rllib.gnn.algorithms.ppo.torch.graph2node_ppo_torch_rl_module import (
    Graph2NodePPOTorchRLModule,
)
from skdecide.hub.solver.ray_rllib.gnn.connectors.flatten_observations import (
    FlattenMultiagentGraphObservations,
    FlattenMultiagentMultiinputObservations,
)
from skdecide.hub.solver.ray_rllib.gnn.connectors.pad_action_logits import (
    PadGraph2NodeActionLogits,
)
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    convert_graph_space_to_dict_space,
    convert_graph_to_dict,
)
from skdecide.hub.solver.ray_rllib.utils import compute_action_new_api_stack_multi_agent
from skdecide.hub.space.gym import GymSpace, MultiDiscreteSpace

logger = logging.getLogger(__name__)

SK_DEFAULT_MODULE_ID = "policy"  # should be different from ray.rllib.core.DEFAULT_MODULE_ID to avoid being detected as single agent


class D(MultiAgentRLDomain):
    pass


class RayRLlib(Solver, Policies, Restorable):
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
            name="train_batch_size_per_learner_log2",  # train_batch_size_per_learner = 2 ** train_batch_size_per_learner_log2
            low=4,
            high=8,
            depends_on=("algo_class", [DQN, SAC]),
        ),
        IntegerHyperparameter(
            name="minibatch_size_log2",  # minibatch_size = 2 ** minibatch_size_log2
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

    MASKABLE_ALGOS = [DQN, PPO]
    """The only algos being able to handle action masking with our wrapper without user-defined rl-module."""

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        algo_class: type[Algorithm],
        train_iterations: int,
        config: Optional[AlgorithmConfig] = None,
        agent2module_id: Optional[dict[str, ModuleID]] = None,
        model_configs: Optional[dict[ModuleID, dict[str, Any]]] = None,
        module_classes: Optional[dict[str, type[RLModule]]] = None,
        rl_module_spec: Optional[MultiRLModuleSpec] = None,
        env_to_module_connector: Optional[
            Callable[[AsRLlibMultiAgentEnv, Any, Any], ConnectorV2 | list[ConnectorV2]]
        ] = None,
        learner_connector: Optional[
            Callable[
                [gym.spaces.Space, gym.spaces.Space, Any],
                ConnectorV2 | list[ConnectorV2],
            ]
        ] = None,
        callback: Optional[Callable[[RayRLlib], bool]] = None,
        graph_feature_extractors_kwargs: Optional[dict[str, Any]] = None,
        graph_node_action: bool = False,
        graph_node_action_net_kwargs: Optional[dict[str, Any]] = None,
        autoregressive_action: bool = False,
        autoregressive_action_nb_graph_independent_components: int | None = None,
        autoregressive_action_components_node_flag_indices: list[int | None]
        | None = None,
        flatten_3d_obs: bool = True,
        **kwargs,
    ) -> None:
        """Initialize Ray RLlib.

        # Parameters
        domain_factory: A callable with no argument returning the domain to solve (can be a mere domain class).
            The resulting domain will be auto-cast to the level expected by the solver.
        algo_class: The class of Ray RLlib trainer/agent to wrap.
        train_iterations: The number of iterations to call the trainer's train() method.
        config: The configuration dictionary for the trainer.
        agent2module_id: Mapping from agent ids to module ids (leave default for single policy).
        model_configs: The mapping from module id (str) to additional config (dict) to be passed to RLModuleSpec.
            Missing keys correspond to default settings (as if mapped to empty dictionary).
        module_classes: The mapping from policy id (str) to the corresponding module class to be passed to RLModuleSpec.
            If a key is omitted or mapped to None, default module class will be used (e.g. DefaultPPOTorchRLModule for PPO).
        rl_module_spec: RL module spec. To be used only by advanced users.
            If specified, it is overriding parameters `model_configs` and`module_classes`.
            The user managed itself the multiagent RL module to be used.
            The keys of `rl_module_spec.rl_module_specs` should correspond to the values of `agent2module_id`.
        env_to_module_connector: env-to-module-connector pipeline preprocessing (gym) observations to be feed to the rl module.
            Default to flatten everything except graphs (e.g. discrete observations are one-hot encoded, dict/list/tuple spaces are concatenated, ...)
            and transform graph instances into batched torch-geometric data.
        learner_connector: learner-connector pipeline preprocessing (gym) observations to be feed to the rl module.
            Default to transforming graph instances into batched torch-geometric data.
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
        autoregressive_action: if True,
          - context: action is made of several components, a component meaning depending on previous components
          - hypotheses:
            - each agent action space should be a MultiDiscreteSpace (even better a MaskableMultiDiscreteSpace)
            - get_applicable_actions() should return an enumerable space
          - the rl-module will predict each action component by using the observation enriched by the previous components
            and will mask it according to current applicable actions
        autoregressive_action_nb_graph_independent_components: Number of graph independent componennts
            when `autoregressive_action` is True and the observations are graphs,
            meaning that last components are actually observation graph nodes.
            Default to total number of action components or
            to number of None in `autoregressive_action_components_node_flag_indices` if specified.
        autoregressive_action_components_node_flag_indices: indices of binary node features for each action component
            when `autoregressive_action` is True and observations are heterogeneous graphs
            whose node types are given via binary node features.
            If specified, None values mean graph independent components and should correspond to first components.
        flatten_3d_obs: by default the obs (or the sub-obs for multiinput dict obs) are flattened before being passed to the rl-module.
            But we could want to let 3d inputs so that they are encoded by CNNs.
        **kwargs: used to update the algo config with kwargs automatically filled by optuna.

        #### Masking

        If the domain has not the `UnrestrictedActions` mixin, and if the algo used allows action masking
        (PPO or DQN), the observations are automatically wrapped to also present
        the action mask to the algorithm, which will used via a custom rl-module.
        During training, a gymnasium environment is created wrapping a domain instantiated from `domain_factory` and
        used during training rollouts to get the observation with the appropriate action mask.
        At inference, we use the method `self.get_action_mask()` which provides the proper action mask provided that
        `self.retrieve_applicable_actions(domain)` has been called before hand with the domain instance used for the inference.
        This is automatically done by `skdecide.utils.rollout()`.

        #### Graph observations

        If the observation space wrapped gymnasium space for each agent is a `gymnasium.spaces.Graph` or a `gymnasium.spaces.Dict`
        whose subspaces contain a `gymnasium.spaces.Graph`, the solver will use a custom encoder using GNNs.

        - If `graph_node_action` is False (default), a GNN will be used to extract (a fixed number of) features from graphs,
          and then classical MLPs will be used for predicting action and value.
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

        #### Autoregressive actions

        if `autoregressive_action` is True, we assume that
        - actions are multidiscrete, with each component dependening on the previous ones.
        - some actions can have less components (last ones are filled with -1, meaning no component)
        - applicable actions are used to compute masks on components
        - we can have graph observations and some components being graph nodes (and should be the last ones).
          In that case the number of graph independent components (the first ones) are to be specified by the user.
        - the graph observation can be heterogeneous with node types being specified via binary node features.
          In that case, a user-specified list of such binary features index (among node features) tells which node type
          is to be used by which component. (For graph independent components which should be the first ones, we specify
          None instead of an integer)

        """
        Solver.__init__(self, domain_factory=domain_factory)

        # domain and wrapped action space and observation space
        domain = self._domain_factory()
        self._wrapped_action_space = domain.get_action_space()
        self._wrapped_observation_space = domain.get_observation_space()

        self.callback = callback
        self._algo_class = algo_class
        self._train_iterations = train_iterations
        self._config = config or algo_class.get_default_config()
        if agent2module_id is None:
            self._agent2module_id = {
                agent: SK_DEFAULT_MODULE_ID for agent in domain.get_agents()
            }
        else:
            self._agent2module_id = agent2module_id
        if model_configs is None:
            self._model_configs = {}
        else:
            self._model_configs = model_configs
        if module_classes is None:
            self._module_classes = {}
        else:
            self._module_classes = module_classes
        self._env_to_module_connector = env_to_module_connector
        self._learner_connector = learner_connector
        self._rl_module_spec = rl_module_spec
        if graph_feature_extractors_kwargs is None:
            self._graph_feature_extractors_kwargs = {}
        else:
            self._graph_feature_extractors_kwargs = graph_feature_extractors_kwargs
        if graph_node_action_net_kwargs is None:
            self._graph2node_action_net_kwargs = {}
        else:
            self._graph2node_action_net_kwargs = graph_node_action_net_kwargs

        # ray.init(ignore_reinit_error=True)
        self._algo_callbacks: Optional[DefaultCallbacks] = None
        self._algo_worker_callbacks: Optional[DefaultCallbacks] = None
        self._algo_evaluation_worker_callbacks: Optional[DefaultCallbacks] = None

        # action masking?
        self._action_masking = (
            (not isinstance(domain, UnrestrictedActions))
            and all(
                isinstance(agent_action_space, EnumerableSpace)
                for agent_action_space in self._wrapped_action_space.values()
            )
            and (self._algo_class in RayRLlib.MASKABLE_ALGOS)
        )

        # graph obs?
        self._is_graph_obs = _is_multiagent_graph_space(self._wrapped_observation_space)
        self._is_graph_multiinput_obs = _is_multiagent_graph_multiinput_space(
            (self._wrapped_observation_space)
        )

        # autoregressive action
        self._autoregressive_action = autoregressive_action
        self._autoregressive_action_nb_graph_independent_components = None
        self._autoregressive_action_components_node_flag_indices = None

        if self._autoregressive_action:
            assert all(
                isinstance(agent_action_space, MultiDiscreteSpace)
                for agent_action_space in self._wrapped_action_space.values()
            )
            if self._is_graph_obs:
                # potentially graph node components
                # -> after n independent ones
                # -> among proper types according to a node flag (= binary feature)
                nb_action_components = max(
                    len(agent_action_space.unwrapped().nvec)
                    for agent_action_space in self._wrapped_action_space.values()
                )
                if autoregressive_action_components_node_flag_indices is not None:
                    # heterogeneous graph node as action components
                    assert (
                        len(autoregressive_action_components_node_flag_indices)
                        == nb_action_components
                    )
                    # same number of graph independent action components?
                    if autoregressive_action_nb_graph_independent_components is None:
                        autoregressive_action_nb_graph_independent_components = sum(
                            index is None
                            for index in autoregressive_action_components_node_flag_indices
                        )
                    else:
                        assert (
                            sum(
                                index is None
                                for index in autoregressive_action_components_node_flag_indices
                            )
                            == autoregressive_action_nb_graph_independent_components
                        ), (
                            "`autoregressive_action_nb_graph_independent_components` should correspond to the number of `None` in `autoregressive_action_components_node_flag_indices`."
                        )
                    # graph independent action components are the first ones?
                    assert all(
                        isinstance(index, int)
                        for index in autoregressive_action_components_node_flag_indices[
                            autoregressive_action_nb_graph_independent_components:
                        ]
                    ), (
                        "Graph independent components (i.e. with `None` value in autoregressive_action_components_node_flag_indices) "
                        "must be the first components."
                    )
                elif autoregressive_action_nb_graph_independent_components is None:
                    # default to no graph2node components
                    autoregressive_action_nb_graph_independent_components = (
                        nb_action_components
                    )

                # is there really graph node components?
                if (
                    autoregressive_action_nb_graph_independent_components
                    < nb_action_components
                ):
                    # at least one graph node component
                    self._autoregressive_action_nb_graph_independent_components = (
                        autoregressive_action_nb_graph_independent_components
                    )
                    self._autoregressive_action_components_node_flag_indices = (
                        autoregressive_action_components_node_flag_indices
                    )
                else:
                    # only independent components
                    self._autoregressive_action_nb_graph_independent_components = None
                    self._autoregressive_action_components_node_flag_indices = None

        # graph -> node (ie an action is a node of observation graph, or some components are nodes for autoregressive action)
        self._graph2node = (self._is_graph_obs and graph_node_action) or (
            self._is_graph_obs
            and self._autoregressive_action
            and self._autoregressive_action_nb_graph_independent_components is not None
        )

        # flatten 3d obs ?
        self._flatten_3d_obs = flatten_3d_obs

        # Handle kwargs (potentially generated by optuna)
        if "train_batch_size_per_learner_log2" in kwargs:
            # train_batch_size_per_learner
            train_batch_size_per_learner_log2 = kwargs.pop(
                "train_batch_size_per_learner_log2"
            )
            kwargs["train_batch_size_per_learner"] = (
                2**train_batch_size_per_learner_log2
            )
        if "minibatch_size_log2" in kwargs:
            # minibatch_size
            minibatch_size_log2 = kwargs.pop("minibatch_size_log2")
            kwargs["minibatch_size"] = 2**minibatch_size_log2
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

        self._rllib_monkey_patched = False

    def get_policy(self) -> MultiRLModule:
        """Return the computed rl module."""
        return self._algo.env_runner.module

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

        # un-monkey patch rllib for graphs
        if self._needs_unmonkey_patch_rllib():
            self._unmonkey_path_rllib()

    def _sample_action(
        self, observation: D.T_agent[D.T_observation], domain: Optional[Domain] = None
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        if self._action_masking and domain is None:
            raise ValueError(
                "The rollout `domain` cannot be None when using action masking."
            )
        action = compute_action_new_api_stack_multi_agent(
            algo=self._algo, observation=self._unwrap_obs(observation, domain=domain)
        )
        return self._wrap_action(action)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _save(self, path: str) -> None:
        self.forget_callback()  # avoid serializing issues
        # make sure to have absolute path to avoid pyarrow issue
        path = os.path.abspath(path)
        self._algo.save(path)
        self.set_callback()  # put it back in case of further solve

    def _load(self, path: str):
        self._init_algo()
        # make sure to have absolute path to avoid pyarrow issue
        path = os.path.abspath(path)
        self._algo.restore(path)
        self.set_callback()  # ensure putting back actual callback

    def _wrap_action(self, action_dict: dict[str, Any]) -> dict[str, D.T_event]:
        return _wrap_action(
            action=action_dict, wrapped_action_space=self._wrapped_action_space
        )

    def _unwrap_obs(
        self, obs: dict[str, D.T_observation], domain: Optional[D] = None
    ) -> dict[str, Any]:
        action_mask = None
        applicable_actions = None
        if self._autoregressive_action:
            assert domain is not None, (
                "The rollout `domain` cannot be None when using applicable actions for autoregressive action prediction."
            )
            applicable_actions = autocast(
                domain.get_applicable_actions, domain, self.T_domain
            )()
        elif self._action_masking:
            assert domain is not None, (
                "The rollout `domain` cannot be None when using action masking."
            )
            action_mask = (autocast(domain.get_action_mask, domain, self.T_domain))()

        return _unwrap_obs(
            obs=obs,
            wrapped_observation_space=self._wrapped_observation_space,
            action_masking=self._action_masking,
            action_mask=action_mask,
            autoregressive_action=self._autoregressive_action,
            applicable_actions=applicable_actions,
        )

    def _needs_monkey_patch_rllib(self) -> bool:
        return (
            self._is_graph_obs
            or self._is_graph_multiinput_obs
            or self._autoregressive_action
        ) and not self._rllib_monkey_patched

    def _monkey_patch_rllib(self) -> None:
        # NB: We would rather do
        # ```python
        # self._algo.env_runner_group.foreach_env_runner(
        #     lambda env_runner: monkey_patch_rllib_for_skedice()
        # )
        # ```
        # as for unpatching at the end of `_solve()`.
        # But at that point the env_runner_group has not been yet properly initialized with all the workers
        # only the local worker exists. (It will be updated at the beginning of the training process, from the config.)
        # So instead we use a custom EnvRunner class that monkey-patch when initialized.
        if self._autoregressive_action:
            if self._graph2node:
                env_runner_cls = AutoregressiveGraph2NodeMultiAgentEnvRunner
            else:
                env_runner_cls = AutoregressiveMultiAgentEnvRunner
        else:
            if self._graph2node:
                env_runner_cls = Graph2NodeMultiAgentEnvRunner
            else:
                env_runner_cls = SkdecideMultiAgentEnvRunner
        if not isinstance(self._config.env_runner_cls, (type(None), env_runner_cls)):
            logger.warning(
                "The EnvRunner class to use for environment rollouts (data collection) will be overriden "
                f"by {env_runner_cls} so that buffers manage properly graphs/action_masks  concatenation."
            )
        self._config.env_runners(env_runner_cls=env_runner_cls)
        self._rllib_monkey_patched = True

    def _unmonkey_path_rllib(self) -> None:
        self._algo.env_runner_group.foreach_env_runner(
            lambda env_runner: unmonkey_patch_rllib_for_skdecide()
        )
        self._rllib_monkey_patched = False

    def _needs_unmonkey_patch_rllib(self) -> bool:
        return hasattr(self, "_algo") and self._rllib_monkey_patched

    def _init_algo(self) -> None:
        # Fix DQN + multiagent: default replay buffer not suited for multiagent envs
        if (
            self._algo_class.__name__ == "DQN"
            and hasattr(self._config, "replay_buffer_config")
            and isinstance(self._config.replay_buffer_config["type"], str)
            and not self._config.replay_buffer_config["type"].startswith("MultiAgent")
        ):
            self._config.replay_buffer_config["type"] = (
                "MultiAgent" + self._config.replay_buffer_config["type"]
            )

        # monkey patch rllib for graph/applicable actions handling
        if self._needs_monkey_patch_rllib():
            self._monkey_patch_rllib()

        # rl-module config (custom or defined according to classic vs graph obs and masking vs no masking=
        if self._rl_module_spec is None:
            # Update:
            # - catalog class: e.g. encoder for graph using GNN
            # - default rl module class: e.g. apply action mask to logits
            # - default model config: e.g. customize GNN
            default_model_config = {}
            default_module_class = None
            catalog_class = None

            # Catalog
            if self._autoregressive_action:
                # Check framework
                if self._config.get("framework") not in [TORCH_FRAMEWORK]:
                    raise NotImplementedError(
                        "Autoregressive actions with RLlib requires PyTorch framework or to use your own RL module."
                    )
                # action component encoder + action component head
                if self._algo_class is PPO:
                    if self._is_graph_obs or self._is_graph_multiinput_obs:
                        # needs also graph /multiinput obs encoder
                        catalog_class = AutoregressiveGraphPPOCatalog
                        # Customize encoder
                        default_model_config["graph_features_extractor_kwargs"] = (
                            self._graph_feature_extractors_kwargs
                        )
                    else:
                        catalog_class = AutoregressivePPOCatalog
                else:
                    raise NotImplementedError(
                        f"Autoregressive actions not available for {self._algo_class.__name__}, "
                        "use your own RL module."
                    )

            elif self._is_graph_obs or self._is_graph_multiinput_obs:
                # Check framework
                if self._config.get("framework") not in [TORCH_FRAMEWORK]:
                    raise NotImplementedError(
                        "Graph observation with RLlib requires PyTorch framework or to use your own RL module."
                    )
                # Encoder choice
                if self._algo_class is PPO:
                    if self._graph2node:
                        catalog_class = Graph2NodePPOCatalog
                    else:
                        catalog_class = GraphPPOCatalog
                else:
                    raise NotImplementedError(
                        f"Graph observation not available for {self._algo_class.__name__}, "
                        "use your own RL module."
                    )
                # Customize encoder
                default_model_config["graph_features_extractor_kwargs"] = (
                    self._graph_feature_extractors_kwargs
                )

            # RL Module class
            if self._autoregressive_action:
                # Check framework
                if self._config.get("framework") not in [TORCH_FRAMEWORK]:
                    raise NotImplementedError(
                        "Autoregressive actions only available for pytorch framework "
                        "if you do not use your own RL module."
                    )
                # autoregressive action component prediction + masks per component in RL-module
                if self._algo_class is PPO:
                    default_module_class = AutoregressivePPOTorchRLModule
                else:
                    raise NotImplementedError(
                        f"Autoregressive actions not available for {self._algo_class.__name__}, "
                        "use your own RL module."
                    )
            elif self._action_masking:
                # Check framework
                if self._config.get("framework") not in [TORCH_FRAMEWORK]:
                    raise NotImplementedError(
                        "Action masking only available for pytorch framework "
                        "if you do not use your own RL module."
                    )
                # Mask logits in RL-module
                if self._algo_class is PPO:
                    if self._graph2node:
                        default_module_class = ActionMaskingGraph2NodePPOTorchRLModule
                        default_model_config["graph2node_action_net_kwargs"] = (
                            self._graph2node_action_net_kwargs
                        )
                    else:
                        default_module_class = ActionMaskingPPOTorchRLModule
                elif self._algo_class is DQN:
                    if self._graph2node:
                        raise NotImplementedError(
                            f"Graph node actions + action masking not available for {self._algo_class.__name__}, "
                            "use your own RL module."
                        )
                    else:
                        default_module_class = ActionMaskingDQNTorchRLModule
                    self._config.training(
                        hiddens=[],
                        dueling=False,
                    )
                else:
                    raise NotImplementedError(
                        f"Action masking not available for {self._algo_class.__name__}, "
                        "use your own RL module."
                    )
            elif self._graph2node:
                # Check framework
                if self._config.get("framework") not in [TORCH_FRAMEWORK]:
                    raise NotImplementedError(
                        "Graph node actions only available for pytorch framework "
                        "if you do not use your own RL module."
                    )
                if self._algo_class is PPO:
                    default_module_class = Graph2NodePPOTorchRLModule
                    default_model_config["graph2node_action_net_kwargs"] = (
                        self._graph2node_action_net_kwargs
                    )
                else:
                    raise NotImplementedError(
                        f"Graph node actions not available for {self._algo_class.__name__}, "
                        "use your own RL module."
                    )

            rl_module_spec = MultiRLModuleSpec(
                rl_module_specs={
                    module_id: RLModuleSpec(
                        module_class=self._module_classes.get(
                            module_id, default_module_class
                        ),
                        model_config=self._model_configs.get(
                            module_id, default_model_config
                        ),
                        catalog_class=catalog_class,
                    )
                    for module_id in self._agent2module_id.values()
                }
            )
        else:
            rl_module_spec = self._rl_module_spec
        self._config.rl_module(rl_module_spec=rl_module_spec)

        # multiagent settings: mapping agent -> module
        policies = set(rl_module_spec.rl_module_specs)

        # Define policy_mapping_fn from agent2module_id: m
        # make sure to put self._agent2module_id in local context to avoid referencing self and having it pickled when pickling `algo.get_state()`
        def policy_mapping_fn(
            agent: str, episode: MultiAgentEpisode, mapping=self._agent2module_id
        ) -> str:
            return mapping[agent]

        self._config.multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )

        # connector preprocessing observations for rl_module  => flatten observations
        if self._env_to_module_connector is None:
            if self._autoregressive_action:
                env_to_module_connector = (
                    lambda env,
                    spaces,
                    device,
                    is_multiinput=self._is_graph_multiinput_obs,
                    is_graph_dict=self._is_graph_obs,
                    flatten_3d_obs=self._flatten_3d_obs: FlattenMultiagentAutoregressiveActionObservations(
                        is_multiinput=is_multiinput,
                        is_graph_dict=is_graph_dict,
                        flatten_3d_obs=flatten_3d_obs,
                    )
                )
            elif self._action_masking:
                env_to_module_connector = (
                    lambda env,
                    spaces,
                    device,
                    is_multiinput=self._is_graph_multiinput_obs,
                    is_graph_dict=self._is_graph_obs,
                    flatten_3d_obs=self._flatten_3d_obs: FlattenMultiagentMaskedObservations(
                        is_multiinput=is_multiinput,
                        is_graph_dict=is_graph_dict,
                        flatten_3d_obs=flatten_3d_obs,
                    )
                )
            elif self._is_graph_obs:
                env_to_module_connector = (
                    lambda env, spaces, device: FlattenMultiagentGraphObservations()
                )
            elif self._is_graph_multiinput_obs:
                env_to_module_connector = (
                    lambda env,
                    spaces,
                    device,
                    flatten_3d_obs=self._flatten_3d_obs: FlattenMultiagentMultiinputObservations(
                        flatten_3d_obs=flatten_3d_obs
                    )
                )
            else:
                env_to_module_connector = (
                    lambda env,
                    spaces,
                    device,
                    flatten_3d_obs=self._flatten_3d_obs: FlattenMultiagentObservations(
                        flatten_3d_obs=flatten_3d_obs
                    )
                )

        else:
            env_to_module_connector = self._env_to_module_connector

        self._config.env_runners(
            env_to_module_connector=env_to_module_connector,
        )

        # learner to module
        if self._graph2node:
            # pad action distribution inputs (as number of nodes can vary)
            if self._autoregressive_action:
                # action component logits and mask <-> number of nodes
                learner_connector = (
                    lambda observation_space,
                    action_space: PadAutoregressiveGraph2NodeActionDistributionInputs()
                )
            else:
                # action logits <-> number of nodes
                learner_connector = (
                    lambda observation_space, action_space: PadGraph2NodeActionLogits()
                )
        else:
            learner_connector = None
        self._config.learners(learner_connector=learner_connector)

        # gym env wrapper for env runners
        register_env(
            "skdecide_env",
            lambda _,
            domain_factory=self._domain_factory,
            rayrllib=self: AsRLlibMultiAgentEnv(
                domain=domain_factory(),
                action_masking=rayrllib._action_masking,
                graph2node=rayrllib._graph2node,
                autoregressive_action=rayrllib._autoregressive_action,
                action_components_node_flag_indices=rayrllib._autoregressive_action_components_node_flag_indices,
                nb_graph_independent_components=rayrllib._autoregressive_action_nb_graph_independent_components,
            ),
        )
        self._config.environment(
            env="skdecide_env",
            # Disable env checking in case of action masking otherwise RLlib will try to simulate
            # next state transition with invalid actions, which might make some domains crash if
            # they require action masking
            disable_env_checking=True,
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
        if self.callback is None:
            callbacks_class = RLlibCallback
        else:
            callbacks_class = generate_rllibcallback_class(
                callback=self.callback, solver=self
            )
        # use it in all algo config, and callbacks attributes
        self._set_callbackclass(callbacks_class=callbacks_class)

    def forget_callback(self):
        """Forget about actual callback to avoid serializing issues."""
        # use default callback class
        callbacks_class = RLlibCallback
        # use it in algo config & evaluation_config, worker config, and for algo.callbacks, worker.callbacks
        self._set_callbackclass(callbacks_class=callbacks_class)

    def _set_callbackclass(self, callbacks_class: type[DefaultCallbacks]):
        _set_callbackclass_in_config(
            callbacks_class=callbacks_class, config=self._config
        )
        if hasattr(self, "_algo"):
            self._algo_callbacks = _swap_callbacks(
                self._algo,
                previous_callbacks=self._algo_callbacks,
                callbacks_class=callbacks_class,
            )
            if self._algo.config:
                _set_callbackclass_in_config(
                    callbacks_class=callbacks_class, config=self._algo.config
                )
            if self._algo.evaluation_config:
                _set_callbackclass_in_config(
                    callbacks_class=callbacks_class, config=self._algo.evaluation_config
                )
            if self._algo.learner_group:
                if self._algo.learner_group.config:
                    _set_callbackclass_in_config(
                        callbacks_class=callbacks_class,
                        config=self._algo.learner_group.config,
                    )
                if self._algo.learner_group._learner:
                    if self._algo.learner_group._learner.config:
                        _set_callbackclass_in_config(
                            callbacks_class=callbacks_class,
                            config=self._algo.learner_group._learner.config,
                        )
            if self._algo.env_runner:
                _set_callbackclass_in_config(
                    callbacks_class=callbacks_class, config=self._algo.env_runner.config
                )


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
        algo_or_worker.callbacks = [callbacks_class()]
    previous_callbacks = tmp
    return previous_callbacks


class AsRLlibMultiAgentEnv(MultiAgentEnvCompatibility):
    def __init__(
        self,
        domain: D,
        action_masking: bool = False,
        graph2node: bool = False,
        render_mode: Optional[str] = None,
        autoregressive_action: bool = False,
        nb_graph_independent_components: int | None = None,
        action_components_node_flag_indices: list[int | None] | None = None,
    ) -> None:
        self.possible_agents = domain.get_agents()
        self.agents = domain.get_agents()
        old_env = AsLegacyRLlibMultiAgentEnv(
            domain=domain,
            action_masking=action_masking,
            graph2node=graph2node,
            autoregressive_action=autoregressive_action,
            nb_graph_independent_components=nb_graph_independent_components,
            action_components_node_flag_indices=action_components_node_flag_indices,
        )
        super().__init__(old_env=old_env, render_mode=render_mode)
        # new MultiAgentEnv attributes not set by MultiAgentEnvCompatibility.__init__
        self.observation_spaces = self.observation_space.spaces
        self.action_spaces = self.action_space.spaces


class AsLegacyRLlibMultiAgentEnv(AsLegacyGymV21Env):
    def __init__(
        self,
        domain: D,
        action_masking: bool,
        autoregressive_action: bool,
        graph2node: bool = False,
        nb_graph_independent_components: bool | int = None,
        action_components_node_flag_indices: list[int | None] | None = None,
    ) -> None:
        """Initialize AsLegacyRLlibMultiAgentEnv.

        # Parameters
        domain: The scikit-decide domain to wrap as a RLlib multi-agent environment.
        action_masking: Boolean specifying whether action masking is used
        """
        self._autoregressive_action = autoregressive_action
        self._nb_graph_independent_components = nb_graph_independent_components
        self._action_components_node_flag_indices = action_components_node_flag_indices
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
                    autoregressive_action=self._autoregressive_action,
                    graph2node=self._graph2node,
                )
                for agent in self._wrapped_observation_space
            }
        )

        self.action_space = gym.spaces.Dict(
            {
                agent: _create_agent_action_space_for_rllib(
                    wrapped_action_space=self._wrapped_action_space,
                    agent=agent,
                    autoregressive_action=self._autoregressive_action,
                    nb_graph_independent_components=self._nb_graph_independent_components,
                    action_components_node_flag_indices=self._action_components_node_flag_indices,
                )
                for agent in self._wrapped_action_space
            }
        )

    def _wrap_action(self, action_dict: dict[str, Any]) -> dict[str, D.T_event]:
        return _wrap_action(
            action=action_dict, wrapped_action_space=self._wrapped_action_space
        )

    def _unwrap_obs(self, obs: dict[str, D.T_observation]) -> dict[str, Any]:
        action_mask = None
        applicable_actions = None
        if self._autoregressive_action:
            applicable_actions = self._domain.get_applicable_actions()
        elif self._action_masking:
            action_mask = self._domain.get_action_mask()
        return _unwrap_obs(
            obs=obs,
            wrapped_observation_space=self._wrapped_observation_space,
            action_masking=self._action_masking,
            action_mask=action_mask,
            autoregressive_action=self._autoregressive_action,
            applicable_actions=applicable_actions,
        )

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


class BaseRLlibCallback(RLlibCallback):
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
    wrapped_action_space: dict[str, GymSpace[D.T_event]],
    agent: str,
    action_masking: bool,
    autoregressive_action: bool,
    graph2node: bool,
) -> gym.spaces.Space:
    true_observation_space = _unwrap_agent_obs_space(
        wrapped_observation_space=wrapped_observation_space,
        agent=agent,
    )
    if autoregressive_action:
        return gym.spaces.Dict(
            {
                AUTOREGRESSIVE_TRUE_OBS: true_observation_space,
                APPLICABLE_ACTIONS: create_agent_applicable_actions_space(
                    agent_action_space=wrapped_action_space[agent],
                ),
            }
        )
    elif action_masking:
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


def _unwrap_obs(
    obs: dict[str, D.T_observation],
    wrapped_observation_space: dict[str, GymSpace[D.T_observation]],
    action_masking: bool = False,
    action_mask: Optional[dict[str, Mask]] = None,
    autoregressive_action: bool = False,
    applicable_actions: Optional[dict[str, EnumerableSpace]] = None,
) -> Any:
    if autoregressive_action:
        assert autoregressive_action is not None, (
            "applicable_actions cannot be None if autoregressive_action is True"
        )
        return {
            agent: _unwrap_agent_obs_with_autoregressive_action(
                obs=obs,
                agent=agent,
                wrapped_observation_space=wrapped_observation_space,
                applicable_actions=applicable_actions,
            )
            for agent in obs
        }
    elif action_masking:
        assert action_mask is not None, (
            "action_mask cannot be None if action_masking is True"
        )
        return {
            agent: _unwrap_agent_obs_with_action_masking(
                obs=obs,
                agent=agent,
                wrapped_observation_space=wrapped_observation_space,
                action_mask=action_mask,
            )
            for agent in obs
        }
    else:
        return {
            agent: _unwrap_agent_obs(
                obs=obs,
                agent=agent,
                wrapped_observation_space=wrapped_observation_space,
            )
            for agent in obs
        }


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


def _unwrap_agent_obs_with_autoregressive_action(
    obs: dict[str, D.T_observation],
    agent: str,
    wrapped_observation_space: dict[str, GymSpace[D.T_observation]],
    applicable_actions: dict[str, EnumerableSpace[D.T_event]],
) -> dict[str, Union[Any, Mask]]:
    return {
        AUTOREGRESSIVE_TRUE_OBS: _unwrap_agent_obs(
            obs=obs, agent=agent, wrapped_observation_space=wrapped_observation_space
        ),
        APPLICABLE_ACTIONS: np.array(applicable_actions[agent].get_elements()),
    }


def _create_agent_action_space_for_rllib(
    wrapped_action_space: dict[str, GymSpace[D.T_event]],
    agent: str,
    autoregressive_action: bool,
    action_components_node_flag_indices: list[int | None] | None = None,
    nb_graph_independent_components: int | None = None,
) -> gym.spaces.Space:
    if autoregressive_action:
        return create_agent_autoregressive_action_space(
            agent_action_space=wrapped_action_space[agent],
            action_components_node_flag_indices=action_components_node_flag_indices,
            nb_graph_independent_components=nb_graph_independent_components,
        )
    else:
        return wrapped_action_space[agent].unwrapped()


def _wrap_action(
    action: dict[str, Any], wrapped_action_space: dict[str, GymSpace[D.T_event]]
) -> StrDict[D.T_event]:
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
