# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, Optional, Union

import gymnasium as gym
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
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.env_to_module import FlattenObservations
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
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
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
from skdecide.hub.space.gym import GymSpace

from .action_masking.connectors.flatten_observations import (
    FlattenMultiagentMaskedObservations,
)
from .action_masking.utils.spaces.space_utils import (
    ACTION_MASK,
    TRUE_OBS,
    create_agent_action_mask_space,
)
from .gnn.evaluation.rollout_worker import Graph2NodeRolloutWorker, GraphRolloutWorker
from .gnn.utils.monkey_patch import (
    unmonkey_patch_rllib_for_graph,
)
from .gnn.utils.spaces.space_utils import (
    convert_graph_space_to_dict_space,
    convert_graph_to_dict,
)
from .utils import compute_action_new_api_stack_multi_agent

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

    MASKABLE_ALGOS = ["APPO", "BC", "DQN", "Rainbow", "IMPALA", "MARWIL", "PPO"]
    """The only algos being able to handle action masking in ray[rllib]==2.9.0."""

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        algo_class: type[Algorithm],
        train_iterations: int,
        config: Optional[AlgorithmConfig] = None,
        agent2module_id: Optional[dict[str, ModuleID]] = None,
        model_configs: Optional[dict[str, dict[str, Any]]] = None,
        module_classes: Optional[dict[str, type[RLModule]]] = None,
        rl_module_spec: Optional[MultiRLModuleSpec] = None,
        env_to_module_connector: Optional[
            Callable[[AsRLlibMultiAgentEnv, Any, Any], ConnectorV2 | list[ConnectorV2]]
        ] = None,
        callback: Optional[Callable[[RayRLlib], bool]] = None,
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
            Default to flatten everything (e.g. discrete observations are one-hot encoded, dict/list/tuple spaces are concatenated, ...)
            by using `ray.rllib.connectors.common.flatten_observations.FlattenObservations`.
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
        if self._is_graph_obs or self._is_graph_multiinput_obs:
            self._algo.env_runner_group.foreach_worker(
                lambda worker: unmonkey_patch_rllib_for_graph()
            )

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
        if self._action_masking:
            assert domain is not None, (
                "The rollout `domain` cannot be None when using action masking."
            )
            action_mask = (autocast(domain.get_action_mask, domain, self.T_domain))()
        else:
            action_mask = None
        return _unwrap_obs(
            obs=obs,
            wrapped_observation_space=self._wrapped_observation_space,
            action_masking=self._action_masking,
            action_mask=action_mask,
        )

    def _init_algo(self) -> None:
        # monkey patch rllib for graph handling
        # NB: We would rather do
        # ```python
        # self._algo.env_runner_group.foreach_worker(
        #     lambda worker: monkey_patch_rllib_for_graph()
        # )
        # ```
        # as for unpatching at the end of `_solve()`.
        # But at that point the env_runner_group has not been yet properly initialized with all the workers
        # only the local worker exists. (It will be updated at the beginning of the training process, from the config.)
        # So instead we use a custom RolloutWorker class that monkey-patch when initialized.
        if self._is_graph_obs or self._is_graph_multiinput_obs:
            if self._graph2node:
                if not isinstance(
                    self._config.env_runner_cls,
                    (type(None), Graph2NodeRolloutWorker),
                ):
                    logger.warning(
                        "The EnvRunner class to use for environment rollouts (data collection) will be overriden "
                        "by Graph2NodeRolloutWorker so that buffers manage properly graphs concatenation."
                    )
                self._config.env_runners(env_runner_cls=Graph2NodeRolloutWorker)
            else:
                if not isinstance(
                    self._config.env_runner_cls, (type(None), GraphRolloutWorker)
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
                    raise NotImplementedError(
                        "Graph observation with RLlib requires PyTorch framework."
                    )
            if self._config.get("framework") not in ["torch"]:
                raise NotImplementedError(
                    "Action masking (invalid action filtering) with RLlib requires PyTorch framework"
                )
            if self._algo_class.__name__ not in ["PPO"]:
                raise NotImplementedError(
                    "Action masking (invalid action filtering) with RLlib only available for PPO for now."
                )
            if self._graph2node:
                raise NotImplementedError(
                    "RLlib + GNN +action masking not yet implemented with new api stack"
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
                raise NotImplementedError(
                    "Graph observation with RLlib requires PyTorch framework."
                )
            raise NotImplementedError(
                "RLlib + GNN not yet implemented with new api stack"
            )
        elif self._is_graph_multiinput_obs:
            if self._config.get("framework") not in ["torch"]:
                raise NotImplementedError(
                    "Graph observation with RLlib requires PyTorch framework."
                )
            raise NotImplementedError(
                "RLlib + GNN not yet implemented with new api stack"
            )

        # connector preprocessing observations for rl_module
        if self._env_to_module_connector is None:
            if self._action_masking:
                env_to_module_connector = (
                    lambda env, spaces, device: FlattenMultiagentMaskedObservations()
                )
            else:
                env_to_module_connector = (
                    lambda env, spaces, device: FlattenObservations(multi_agent=True)
                )

        else:
            env_to_module_connector = self._env_to_module_connector
        self._config.env_runners(env_to_module_connector=env_to_module_connector)

        # rl-module config
        if self._rl_module_spec is None:
            # rl_module_obs_spaces = {
            #     module_id: _create_agent_obs_space_for_rllib(
            #         wrapped_observation_space=self._wrapped_observation_space,
            #         wrapped_action_space=self._wrapped_action_space,
            #         agent=agent,
            #         action_masking=self._action_masking,
            #         graph2node=self._graph2node,
            #     )
            #     for agent, module_id in self._agent2module_id.items()
            # }
            #
            # rl_module_act_spaces = {
            #     module_id: self._wrapped_action_space[agent].unwrapped()
            #     for agent, module_id in self._agent2module_id.items()
            # }

            if self._action_masking:
                if self._algo_class.__name__ != "PPO":
                    raise NotImplementedError(
                        "For now action masking with new api stack only available for PPO."
                    )
                default_module_class = ActionMaskingTorchRLModule
            else:
                default_module_class = None

            rl_module_spec = MultiRLModuleSpec(
                rl_module_specs={
                    module_id: RLModuleSpec(
                        module_class=self._module_classes.get(
                            module_id, default_module_class
                        ),
                        # action_space=rl_module_act_spaces[module_id],
                        # observation_space=rl_module_act_spaces[module_id],
                        model_config=self._model_configs.get(module_id, {}),
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

        register_env(
            "skdecide_env",
            lambda _,
            domain_factory=self._domain_factory,
            rayrllib=self: AsRLlibMultiAgentEnv(
                domain=domain_factory(),
                action_masking=rayrllib._action_masking,
                graph2node=rayrllib._graph2node,
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
        # TMP: deactivate callback
        return
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
        self.possible_agents = domain.get_agents()
        self.agents = domain.get_agents()
        old_env = AsLegacyRLlibMultiAgentEnv(
            domain=domain, action_masking=action_masking, graph2node=graph2node
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
        if self._action_masking:
            action_mask = self._domain.get_action_mask()
        else:
            action_mask = None
        return _unwrap_obs(
            obs=obs,
            wrapped_observation_space=self._wrapped_observation_space,
            action_masking=self._action_masking,
            action_mask=action_mask,
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


def _unwrap_obs(
    obs: dict[str, D.T_observation],
    wrapped_observation_space: dict[str, GymSpace[D.T_observation]],
    action_masking: bool = False,
    action_mask: Optional[dict[str, Mask]] = None,
) -> Any:
    if action_masking:
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
