import asyncio
import importlib.util
import logging
import os
import sys
import tempfile
import uuid
from abc import abstractmethod
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Union

from openevolve import OpenEvolve
from openevolve.config import Config, load_config
from openevolve.database import Program
from openevolve.evaluation_result import EvaluationResult

from skdecide import Domain, Solver, Space
from skdecide.builders.domain import Sequential, SingleAgent, UnrestrictedActions
from skdecide.builders.solver import ApplicableActions, Policies, Restorable

from .api_extraction import ApiExtractionParams, generate_public_api
from .evaluator_builder import build_evaluator
from .initial_program_builder import build_initial_program

logger = logging.getLogger(__name__)

EVOLVE_BLOCKS_INSTRUCTION = "\nYou must only perform edits between the # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END comments.\n"
PUBLIC_API_INSTRUCTION = "\nYou are not allowed to use private attributes or methods of the domain, only its public API.\n"
MODULE_IN_PROMPT_TEMPLATE = (
    "\nFor context, here is the content of the module defining the {domain_cls}:\n"
    "```python\n{module_code}```\n"
)
PUBLIC_API_IN_PROMPT_TEMPLATE = (
    "\nFor context, here is the public api of {domain_cls}:\n{api_str}"
)
PUBLIC_API_RECURSIVE_IN_PROMPT_TEMPLATE = "\nFor context, here is the public api of {domain_cls} and of other relevant classes:\n{api_str}"


class D(Domain, SingleAgent, Sequential): ...


class Planner(Protocol):
    """Protocol for the class defined in the evolving program."""

    @abstractmethod
    def sample_action(self, obs: Any, applicable_actions: Optional[Any] = None) -> Any:
        """Sample action method from the planner.

        Potentially the types are different from the scikit-decide solver's `sample_action` method.

        # Parameters
        obs: current observation
        applicable_actions: applicable actions depending on current domain state
            Not used if the domain derives from `UnrestrictedActions`.

        # Returns
        sampled action

        """
        ...


class _BaseOpenEvolve(Solver, Policies, Restorable, ApplicableActions):
    """Wrapper around openevolve to evolve a greedy heuristic adapted to a given domain class.

    We start from an initial program that should
    - define a `Planner` class (instantiated from characteristics of the domain)
    - define a `Planner.sample_action(obs, applicable_actions)` sampling an action from a domain observation (or a characteristics of it)
    - have `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` around the class definition

    Openevolve makes it evolve and the resulting program is loaded to be called in the solver's `sample_action()`.

    """

    T_domain = D

    evolved_planner_cls_name = "Planner"
    """Name of the planner class in the initial (and evolved) program(s)."""

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        initial_program_path: str,
        evaluator_path: str,
        config: Union[str, Path, Config, None] = None,
        output_dir: Union[str, None] = None,
        target_score: Union[float, None] = None,
        prompt_add_blockevolve_instruction: bool = True,
        nb_iterations: Optional[int] = None,
        **kwargs: Any,
    ):
        """

        # Parameters
        domain_factory: domain factory on which the solver works
        initial_program_path: path to the initial program to evolve (see above for hypotheses on it)
        evaluator_path: path to the evaluation script used by openevolve
        config: config yaml file to be used by openevolve (or openevolve.Config object). Potentially contains the
            prompt header used to describe the domain to work on. If a config object is given, it is copied to avoid
            prompt automatic modifications to be stored in the original config.
        output_dir: output directory for openevolve evolution
        target_score: target score used by openevolve
        prompt_add_blockevolve_instruction: if True, add an instruction at the end of the system prompt found in the config
            to restrict changes between the `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` token.
        nb_iterations: override number of iterations found in config
        kwargs:
        """
        super().__init__(domain_factory)

        # load openevolve config
        if not isinstance(config, Config):
            # load from file
            self._config = load_config(config)
        else:
            # copy the config (as we potentially update the prompt config)
            # 1. Replace none values by valid ones to make Config.from_dict() work
            def fix_none(obj):
                for field in fields(obj):
                    value = getattr(obj, field.name)

                    # Check if the field is a nested dataclass
                    if is_dataclass(value):
                        fix_none(value)

                    # Check if the field is typed as a string and is currently None
                    elif field.type is str and value is None:
                        setattr(obj, field.name, "")

                    elif field.type is float and value is None:
                        setattr(obj, field.name, 0.0)

            fix_none(config)
            self._config = Config.from_dict(config.to_dict())

        # override nb of iterations
        if nb_iterations is not None:
            self._config.max_iterations = nb_iterations

        # define output directory (or use default one)
        self._output_dir = output_dir

        # optional target score
        self._target_score = target_score

        # define initial program and evaluator
        self._initial_program_path = initial_program_path
        self._evaluator_path = evaluator_path

        # default values
        self._checkpoint_path = None

        # store an instance of the domain *not autocast* for later checks or attributes lookup
        self._original_domain = domain_factory()

        # check unrestricted actions or not
        self._unrestricted_actions = isinstance(
            self._original_domain, UnrestrictedActions
        )

        # enrich automatically prompt with context, instructions
        self._enrich_prompt(
            prompt_add_blockevolve_instruction=prompt_add_blockevolve_instruction,
            **kwargs,
        )

    def _enrich_prompt(
        self, prompt_add_blockevolve_instruction: bool = True, **kwargs
    ) -> None:
        if prompt_add_blockevolve_instruction:
            self._config.prompt.system_message += EVOLVE_BLOCKS_INSTRUCTION

    def _init_openevolve_controller(self) -> None:
        self._openevolve = OpenEvolve(
            initial_program_path=self._initial_program_path,
            evaluation_file=self._evaluator_path,
            config=self._config,
            output_dir=self._output_dir,
        )

    def _build_planner_from_database(self) -> None:
        if not hasattr(self, "_openevolve"):
            self._init_openevolve_controller()
        self._build_planner_from_evolved_program(
            self._openevolve.database.get_best_program()
        )

    def _build_planner_from_evolved_program(
        self, program: Union[Program, None]
    ) -> None:
        code = self._get_code_from_evolved_program(program)
        self._build_planner_from_program_code(code)

    def _get_code_from_evolved_program(self, program: Union[Program, None]) -> str:
        if program is not None:
            code = program.code
        else:
            code = self._openevolve.initial_program_code
        return code

    def _build_planner_from_program_code(self, program_code: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(program_code.encode("utf-8"))
            program_path = temp_file.name
        try:
            module_name = "evolved_program"
            spec = importlib.util.spec_from_file_location(module_name, program_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            evolved_planner_cls = vars(module)[self.evolved_planner_cls_name]
        finally:
            if os.path.exists(program_path):
                os.unlink(program_path)

        self._evolved_planner = self._build_planner_from_loaded_cls(evolved_planner_cls)

    @abstractmethod
    def _build_planner_from_loaded_cls(
        self, evolved_planner_cls: type[Planner]
    ) -> Planner: ...

    def _solve(self) -> None:
        # Init controller if not existing (keep existing one to evolve further)
        if not hasattr(self, "_openevolve"):
            self._init_openevolve_controller()
        # Run evolution
        best_program = asyncio.run(
            self._openevolve.run(
                checkpoint_path=self._checkpoint_path,
                target_score=self._target_score,
            )
        )
        # Build new evolved planner
        self._build_planner_from_evolved_program(best_program)

    def _sample_action(self, observation: D.T_observation) -> D.T_event:
        if not hasattr(self, "_evolved_planner"):
            # one can call sample_action before solve => use initial program
            self._build_planner_from_database()
        if self.using_applicable_actions():
            return self._convert_action_from_planner(
                self._evolved_planner.sample_action(
                    self._convert_obs_for_planner(observation),
                    self._convert_applicable_actions_for_planner(
                        self._applicable_actions
                    ),
                )
            )
        else:
            return self._convert_action_from_planner(
                self._evolved_planner.sample_action(
                    self._convert_obs_for_planner(observation),
                )
            )

    def _convert_action_from_planner(self, pre_action: Any) -> D.T_event:
        """Conversion of planner action.

        By default, identity.

        """
        return pre_action

    def _convert_obs_for_planner(self, observation: D.T_observation) -> Any:
        """Convert into planner observation format.

        By default, identity.

        """
        return observation

    def _convert_applicable_actions_for_planner(
        self, applicable_actions: Space[D.T_event]
    ) -> Any:
        """Convert into planner actions format.

        By default, identity.

        """
        return applicable_actions

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _retrieve_applicable_actions(self, domain: Domain) -> None:
        self._applicable_actions = domain.get_applicable_actions()

    def _using_applicable_actions(self) -> bool:
        return not self._unrestricted_actions

    def _save(self, path: str) -> None:
        """Save the solver state to given path.

        Save the openevolve database to the given path (as it does for checkpoint)

        # Parameters
        path: The path to store the saved state.
            (Must be a directory path, potentially to be created)
        """
        if hasattr(self, "_openevolve"):
            self._openevolve.database.save(path)

    def _load(self, path: str) -> None:
        """Restore the solver state from given path.

        Load the given checkpoint directory into the openevolve database.

        # Parameters
        path: The path where the solver state was saved.
        """
        self._init_openevolve_controller()
        # load checkpoint right now to retrieve best program
        # and be able to use `sample_actions()` right away
        self._openevolve.database.load(path)
        self._build_planner_from_database()
        # store checkpoint path so that solve() knows it and avoid overriding the last iteration from the checkpoint
        # (it will reload the checkpoint though, but should be better than forgetting about last iteration)
        self._checkpoint_path = path

    def get_best_program_code(self) -> str:
        """Get best program code found so far.

        # Returns


        """
        if not hasattr(self, "_openevolve"):
            self._init_openevolve_controller()
        return self._get_code_from_evolved_program(
            self._openevolve.database.get_best_program()
        )

    def evaluate_program_code(self, code: str) -> EvaluationResult:
        """Evaluate the program code as openevolve does during evolution.

        # Parameters
        code: code of the program to evaluate

        # Returns
        evaluation result computed by the opensolve controller using the underlying evaluator file

        """
        if not hasattr(self, "_openevolve"):
            self._init_openevolve_controller()

        # assign random pid (to retrieve artifacts)
        pid = str(uuid.uuid4())
        metrics = asyncio.run(self._openevolve.evaluator.evaluate_program(code, pid))
        artifacts = self._openevolve.evaluator.get_pending_artifacts(pid)
        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    def sample_prompt(self) -> dict[str, str]:
        """Sample a prompt for LLM as done during openevolve evolution with current settings.

        # Returns
        a dictionary
        - system -> system prompt
        - user -> user prompt

        """
        if not hasattr(self, "_openevolve"):
            self._init_openevolve_controller()

        database = self._openevolve.database
        prompt_sampler = self._openevolve.prompt_sampler
        config = self._openevolve.config

        if len(database.programs) == 0:
            # no iteration performed => no program in database
            parent_program = self._openevolve.initial_program_code
            res = self.evaluate_program_code(parent_program)
            parent_metrics = res.metrics
            parent_artifacts = res.artifacts

            island_previous_programs = []
            island_top_programs = []
            inspirations = []

        else:
            # regular case, do as during openevolve evolution

            # Sample parent and inspirations from database
            parent, inspirations = database.sample(
                num_inspirations=config.prompt.num_top_programs
            )

            # Get artifacts for the parent program if available
            parent_artifacts = database.get_artifacts(parent.id)

            # Get island-specific top programs for prompt context (maintain island isolation)
            parent_island = parent.metadata.get("island", database.current_island)
            island_top_programs = database.get_top_programs(5, island_idx=parent_island)
            island_previous_programs = database.get_top_programs(
                3, island_idx=parent_island
            )

            parent_program = parent.code
            parent_metrics = parent.metrics

        # Build prompt
        prompt = prompt_sampler.build_prompt(
            current_program=parent_program,
            parent_program=parent_program,
            program_metrics=parent_metrics,
            previous_programs=[p.to_dict() for p in island_previous_programs],
            top_programs=[p.to_dict() for p in island_top_programs],
            inspirations=[p.to_dict() for p in inspirations],
            language=config.language,
            diff_based_evolution=config.diff_based_evolution,
            program_artifacts=parent_artifacts if parent_artifacts else None,
            feature_dimensions=database.config.feature_dimensions,
        )

        return prompt


class ProxyOpenEvolve(_BaseOpenEvolve):
    """Wrapper around openevolve without LLM knowing about scikit-decide.

    We start from an initial program that should
    - define a `Planner` class (instantiated from characteristics of the domain)
    - define a `Planner.sample_action(obs, applicable_actions=None)` sampling an action from a domain observation (or a characteristics of it),
        and applicable actions (potentially in a custom format) in the case of a domain with restricted actions
    - have `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` around the class definition
    - avoid any reference to scikit-decide api (not mandatory but recommended)

    The idea here is that the initial program should be self-contained to make openevolve works more efficiently
    without the LLM needing to know about scikit-decide.

    We then have functions to be provided by the user:
    - `planner_kwargs_factory`: extract needed args to instantiate `Planner` from the domain
    - `planner_obs_converter`: convert domain observation into first arg needed by `Planner.sample_action()`
    - `planner_action_converter`: convert output of `Planner.sample_action` into domain action
    - `planner_applicable_actions_converter`: convert applicable actions from `domain.get_applicable_actions()`
        into second arg needed by `Planner.sample_action()` (this arg is not used for domains deriving from `UnrestrictedActions`)

    Note: the evaluator can reference the scikit-decide as it is not sent to the LLM.

    """

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        initial_program_path: str,
        evaluator_path: str,
        planner_kwargs_factory: Callable[[Domain], dict[str, Any]],
        planner_obs_converter: Callable[[D.T_observation], Any] = lambda obs: obs,
        planner_action_converter: Callable[[Any], D.T_event] = lambda action: action,
        planner_applicable_actions_converter: Callable[
            [Space[D.T_event]], Any
        ] = lambda applicable_actions: applicable_actions,
        config: Union[str, Path, Config, None] = None,
        output_dir: Union[str, None] = None,
        target_score: Union[float, None] = None,
        prompt_add_blockevolve_instruction: bool = True,
        nb_iterations: Optional[int] = None,
        **kwargs: Any,
    ):
        """

        # Parameters
        domain_factory: domain factory on which the solver works
        initial_program_path: path to the initial program to evolve (see above for hypotheses on it)
        evaluator_path: path to the evaluation script used by openevolve
        planner_kwargs_factory: function generating the kwargs to instanciate the evolved planner from the domain
        planner_obs_converter: function converting domain observation into first arg needed by the evolved planner
            `sample_action` method (default to identity)
        planner_action_converter: function converting the action returned by the evolved planner into an actual action
            of the domain (default to identity)
        planner_applicable_actions_converter: function converting applicable actions space into second arg needed
            by the evolved planner `sample_action` method (default to identity).
            Not used if the domain derives from `UnrestrictedActions`.
        config: config yaml file to be used by openevolve (or openevolve.Config object). Potentially contains the
            prompt header used to describe the domain to work on.
        output_dir: output directory for openevolve evolution
        target_score: target score used by openevolve
        prompt_add_blockevolve_instruction: if True, add an instruction at the end of the system prompt found in the config
            to restrict changes between the `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` token.
        nb_iterations: override number of iterations found in config
        kwargs:
        """
        super().__init__(
            domain_factory=domain_factory,
            initial_program_path=initial_program_path,
            evaluator_path=evaluator_path,
            config=config,
            output_dir=output_dir,
            target_score=target_score,
            prompt_add_blockevolve_instruction=prompt_add_blockevolve_instruction,
            nb_iterations=nb_iterations,
            **kwargs,
        )

        # bridge functions between the evolved planner and the scikit-decide world
        self._planner_kwargs_factory = planner_kwargs_factory
        self._planner_obs_converter = planner_obs_converter
        self._planner_action_converter = planner_action_converter
        self._planner_applicable_actions_converter = (
            planner_applicable_actions_converter
        )

    def _build_planner_from_loaded_cls(
        self, evolved_planner_cls: type[Planner]
    ) -> Planner:
        return evolved_planner_cls(
            **self._planner_kwargs_factory(self._original_domain)
        )

    def _convert_action_from_planner(self, pre_action: Any) -> D.T_event:
        return self._planner_action_converter(pre_action)

    def _convert_obs_for_planner(self, observation: D.T_observation) -> Any:
        return self._planner_obs_converter(observation)

    def _convert_applicable_actions_for_planner(
        self, applicable_actions: Space[D.T_event]
    ) -> Any:
        return self._planner_applicable_actions_converter(applicable_actions)


class IntegratedOpenEvolve(_BaseOpenEvolve):
    """Wrapper around openevolve with the LLM directly using the scikit-decide domain API.

    We start from an initial program that should
    - define a `Planner` class initialized with a domain instance
    - define a `Planner.sample_action()` sampling a domain action from a domain observation
    - have `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` around the class definition
    - import the scikit-decide domain

    The idea here is that the initial program has access to the scikit-decide domain public API.
    Could potentially create a solver for all domains sharing the domain characteristics.

    """

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        initial_program_path: Optional[str] = None,
        initial_program_include_rollout: bool = True,
        evaluator_path: Optional[str] = None,
        evaluator_domain_factories: Optional[list[Callable[[], Domain]]] = None,
        evaluator_rollout_num_episodes: int = 3,
        evaluator_rollout_max_steps: Union[int, Callable[[Domain], int]] = 100,
        evaluator_rollout_normalize: bool = True,
        evaluator_timeout: int = 60,
        evaluator_enforce_using_public_api: bool = False,
        config: Union[str, Path, Config, None] = None,
        output_dir: Union[str, None] = None,
        target_score: Union[float, None] = None,
        prompt_add_blockevolve_instruction: bool = True,
        prompt_add_public_api_instruction: bool = True,
        prompt_include_domain_module: bool = False,
        prompt_include_public_api: bool = True,
        prompt_include_public_api_params: Optional[ApiExtractionParams] = None,
        prompt_update_function: Optional[Callable[[str], str]] = None,
        nb_iterations: Optional[int] = None,
        **kwargs: Any,
    ):
        """

        # Parameters
        domain_factory: domain factory on which the solver works
        initial_program_path: path to the initial program to evolve. Default to a generated one with a random sampler
            of actions available in domain_factory(). If given, other initial program parameters are ignored.
        initial_program_include_rollout: whether to add an example of rollout at the end of the initial program or not.
        evaluator_path: path to the evaluation script used by openevolve. Default to a generated one with rollouts on
            `evaluator_domain_factories`. If given, other evaluator parameters are ignored.
        evaluator_domain_factories: list of domain factories on which a rollout is applied to generate the evaluator.
            Default to `[domain_factory]`.
        evaluator_rollout_num_episodes: number of episodes to perform on each domain
        evaluator_rollout_max_steps: max steps for each episode. Either an integer or a function mapping a domain to an
            integer, usually in relation with the domain size.
        evaluator_rollout_normalize: whether normalizing the cost of a rollout by the `evaluator_rollout_max_steps`
            (i.e. potentially the size of the domain)
        evaluator_timeout: time (in s) allowed for the evaluation of an evolved program.
        evaluator_enforce_using_public_api: if True, the evaluator domains will be wrapped in a proxy that
            allows only calling their public api (else raising an AttributeError with a message telling not using the private attributes).
            The proxy class still derives from the relevant characteristics found in skdecide.builder.domain so that
            checks `isinstance()` based are still working.
        planner_kwargs_factory: function generating the kwargs to instanciate the evolved planner from the domain
        planner_obs_converter: function converting domain observation into args needed by the evolved planner
            `sample_action` method (default to identity)
        planner_action_converter: function converting the action returned by the evolved planner into an actual action
            of the domain (default to identity)
        config: config yaml file to be used by openevolve (or openevolve.Config object). Potentially contains the
            prompt header used to describe the domain to work on.
        output_dir: output directory for openevolve evolution
        target_score: target score used by openevolve
        prompt_add_blockevolve_instruction: if True, add an instruction at the end of the system prompt found in the config
            to restrict changes between the `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` token.
        prompt_add_public_api_instruction: if True, add an instruction in prompt to use only the public api of the domain
        prompt_include_domain_module: if True, include domain class module code in the prompt
        prompt_include_public_api: if True, include the public api of the domain class in the prompt
        prompt_include_public_api_params: parameters for public api extraction
        prompt_update_function: callable to apply on the system prompt included in the config to update it
        nb_iterations: override number of iterations found in config
        kwargs:
        """
        create_initial_program = False
        create_evaluator = False

        if initial_program_path is None:
            if output_dir is None:
                output_dir = f"{os.getcwd()}/openevolve_output"
            os.makedirs(output_dir, exist_ok=True)
            initial_program_path = f"{output_dir}/generated_initial_program.py"
            create_initial_program = True

        if evaluator_path is None:
            if output_dir is None:
                output_dir = (
                    f"{os.path.dirname(initial_program_path)}/openevolve_output"
                )
            os.makedirs(output_dir, exist_ok=True)
            evaluator_path = f"{output_dir}/generated_evaluator.py"
            create_evaluator = True

        super().__init__(
            domain_factory=domain_factory,
            initial_program_path=initial_program_path,
            evaluator_path=evaluator_path,
            config=config,
            output_dir=output_dir,
            target_score=target_score,
            prompt_add_blockevolve_instruction=prompt_add_blockevolve_instruction,
            prompt_include_domain_module=prompt_include_domain_module,
            prompt_add_public_api_instruction=prompt_add_public_api_instruction,
            prompt_include_public_api=prompt_include_public_api,
            prompt_include_public_api_params=prompt_include_public_api_params,
            prompt_update_function=prompt_update_function,
            nb_iterations=nb_iterations,
            **kwargs,
        )
        if create_initial_program:
            domain_cls = type(self._original_domain)
            with open(initial_program_path, "w") as f:
                f.write(
                    build_initial_program(
                        domain_cls=domain_cls,
                        include_rollout=initial_program_include_rollout,
                    )
                )
        if create_evaluator:
            if evaluator_domain_factories is None:
                evaluator_domain_factories = [domain_factory]
            with open(evaluator_path, "w") as f:
                f.write(
                    build_evaluator(
                        domain_factories=evaluator_domain_factories,
                        max_steps=evaluator_rollout_max_steps,
                        num_episodes=evaluator_rollout_num_episodes,
                        normalize=evaluator_rollout_normalize,
                        timeout=evaluator_timeout,
                        enforce_using_public_api=evaluator_enforce_using_public_api,
                    )
                )

    def _build_planner_from_loaded_cls(
        self, evolved_planner_cls: type[Planner]
    ) -> Planner:
        """Build the planner.

        Use the domain as only argument (according to assumptions made on the program).

        """
        # use a new domain to avoid any "pollution" by a previous program version
        return evolved_planner_cls(domain=self.original_domain_factory())

    def _enrich_prompt(
        self,
        prompt_add_blockevolve_instruction: bool = True,
        prompt_include_domain_module: bool = False,
        prompt_add_public_api_instruction: bool = True,
        prompt_include_public_api: bool = True,
        prompt_include_public_api_params: Optional[ApiExtractionParams] = None,
        prompt_update_function: Optional[Callable[[str], str]] = None,
        **kwargs,
    ) -> None:
        if prompt_update_function is not None:
            self._config.prompt.system_message = prompt_update_function(
                self._config.prompt.system_message
            )
        super()._enrich_prompt(
            prompt_add_blockevolve_instruction=prompt_add_blockevolve_instruction,
            **kwargs,
        )
        if prompt_add_public_api_instruction:
            self._config.prompt.system_message += PUBLIC_API_INSTRUCTION

        if prompt_include_public_api:
            if prompt_include_public_api_params is None:
                prompt_include_public_api_params = ApiExtractionParams()
            domain = self._original_domain
            domain_cls = type(domain)
            api_str = generate_public_api(
                cls=domain_cls,
                domain=domain,
                params=prompt_include_public_api_params,
            )
            if prompt_include_public_api_params.recursive:
                template = PUBLIC_API_RECURSIVE_IN_PROMPT_TEMPLATE
            else:
                template = PUBLIC_API_IN_PROMPT_TEMPLATE
            api_in_prompt = template.format(domain_cls=domain_cls, api_str=api_str)
            self._config.prompt.system_message += api_in_prompt

        if prompt_include_domain_module:
            domain_cls = type(self.domain_factory())
            with open(sys.modules[domain_cls.__module__].__file__, "r") as f:
                module_code = f.read()
            module_in_prompt = MODULE_IN_PROMPT_TEMPLATE.format(
                domain_cls=domain_cls, module_code=module_code
            )
            self._config.prompt.system_message += module_in_prompt
