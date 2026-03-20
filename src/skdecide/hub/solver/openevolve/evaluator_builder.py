import concurrent.futures
import importlib
import importlib.util
import logging
import os
import traceback
import uuid
from functools import partial
from typing import Any, Callable, Optional, Union

from openevolve.evaluation_result import EvaluationResult

from skdecide import Domain, rollout
from skdecide.builders.domain import Goals, UnrestrictedActions
from skdecide.builders.solver import Policies

from .public_proxy import create_public_proxy

logger = logging.getLogger(__name__)

template_dir = f"{os.path.dirname(__file__)}/templates"
DOMAIN_MODULE_PLACEHOLDER = "{{domain_module}}"
DOMAIN_CLS_PLACEHOLDER = "{{domain_cls}}"


def build_evaluator(
    domain_factories: list[Callable[[], Domain]],
    max_steps: Union[int, Callable[[Domain], int]] = 100,
    num_episodes: int = 3,
    normalize: bool = True,
    timeout: int = 60,
    enforce_using_public_api: bool = False,
) -> str:
    """Generate code for the evaluate function used by openevolve.

    - The `evaluate` function must take only `program_path` as argument,
    - It must be written in a separate file, so this wrapper make necessary argument importable
      by storing it in globals of this module.

    # Parameters
    domain_factories: domain factories on which rolling out
    max_steps: maximum number of steps per episode.
        Either an integer or a callable mapping a domain to an integer.
    num_episodes: number of episodes to do per domain
    normalize: whether to normalize cost by max steps (potentially related to domain size)
    timeout: seconds before timeout
    enforce_using_public_api: if True, wraps the domains so that only their public API can be seen by the program

    # Returns
    evaluation code to be used by openevolve

    """
    # evaluator from evaluate with pre-filled args
    evaluator = partial(
        evaluate,
        domain_factories=domain_factories,
        max_steps=max_steps,
        num_episodes=num_episodes,
        normalize=normalize,
        timeout=timeout,
        enforce_using_public_api=enforce_using_public_api,
    )

    # Create a unique global name for this evaluator
    evaluator_id = f"_openevolve_evaluator_{uuid.uuid4().hex[:8]}"

    # Store in globals so the wrapper can find it
    globals()[evaluator_id] = evaluator

    return f"""
# Wrapper for user-provided evaluator function
import {__name__} as evaluator_builder_module

def evaluate(program_path):
    '''Wrapper for auto-generated evaluator function from user parameters.

    Based on skdecide.hub.solver.openevolve.evaluator_builder.evaluate with pre-filled args.

    '''
    user_evaluator = getattr(evaluator_builder_module, '{evaluator_id}')
    return user_evaluator(program_path)
"""


def run_with_timeout(
    func: Callable[..., Any], *args, timeout_seconds: int = 30, **kwargs
) -> Any:
    """Run a function with a timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Solver timed out after {timeout_seconds} seconds")


def evaluate_solver_on_single_domain(
    program_path: str,
    domain_factory: Callable[[], Domain],
    max_steps: Union[int, Callable[[Domain], int]] = 100,
    num_episodes: int = 3,
    normalize: bool = True,
    render: bool = False,
    enforce_using_public_api: bool = False,
) -> tuple[float, ...]:
    """Evaluate an evolved solver on a domain instance.

    We evaluate by performing a rollout on the domain and taking the total cost, potentially normalized by max_steps.

    # Parameters
    program_path: path to the program to evaluate
    domain_factory: domain factory to generate a domain for the planner and for the rollout
    max_steps: maximum number of steps per episode.
        Either an integer or a callable mapping a domain to an integer.
    num_episodes: number of episodes to do
    normalize: whether to normalize cost by max steps (potentially related to domain size)
    render: render the domain during rollout
    enforce_using_public_api: if True, wraps the domain so that only its public API can be seen by the program

    # Returns
    total cost of the rollout, potentially normalized by max_steps

    """
    # avoid planner accessing the same domain as for rollout to avoid "cheating"
    # (e.g. using `set_memory()`)
    rollout_domain = domain_factory()
    solver_domain = domain_factory()
    if not isinstance(max_steps, int):
        max_steps_domain = max_steps(rollout_domain)
    else:
        max_steps_domain = max_steps
    solver = EvolvedPolicies(
        domain=solver_domain,
        program_path=program_path,
        enforce_using_public_api=enforce_using_public_api,
    )
    episodes = rollout(
        solver=solver,
        domain=rollout_domain,
        render=render,
        verbose=False,
        num_episodes=num_episodes,
        max_steps=max_steps_domain,
        return_episodes=True,
    )
    total_cost = sum(
        sum(c.cost for c in ep_costs) for ep_obs, ep_actions, ep_costs in episodes
    )
    # normalization
    if normalize:
        total_cost /= max_steps_domain

    # goal reached?
    if isinstance(rollout_domain, Goals):
        nb_reached_goals = sum(
            rollout_domain.is_goal(ep_obs[-1])
            for ep_obs, ep_actions, ep_costs in episodes
        )
    else:
        nb_reached_goals = None

    # concatenate metrics
    if nb_reached_goals is None:
        return (total_cost,)
    else:
        return total_cost, nb_reached_goals


def evaluate_solver_on_several_domains(
    program_path: str,
    domain_factories: list[Callable[[], Domain]],
    max_steps: Union[int, Callable[[Domain], int]] = 100,
    num_episodes: int = 3,
    normalize: bool = True,
    render: bool = False,
    enforce_using_public_api: bool = False,
) -> tuple[float, ...]:
    """Evaluate an evolved solver on several domain instances.

    We evaluate by performing a rollout on domains and taking the total cost, potentially normalized by max_steps.

    # Parameters
    program_path: path to the program to evaluate on the given domain
    domain_factories: domain factories on which rolling out
    max_steps: maximum number of steps per episode.
        Either an integer or a callable mapping a domain to an integer.
    num_episodes: number of episodes to do per domain
    normalize: whether to normalize cost by max steps (potentially related to domain size)
    render: render the domain during rollout
    enforce_using_public_api: if True, wraps the domains so that only their public API can be seen by the program

    # Returns
    - total cost of the rollout, potentially normalized by max_steps
    - total number of reached goals (when domain inherits `Goals`)

    """
    results_by_domain = (
        evaluate_solver_on_single_domain(
            program_path=program_path,
            domain_factory=domain_factory,
            max_steps=max_steps,
            normalize=normalize,
            num_episodes=num_episodes,
            render=render,
            enforce_using_public_api=enforce_using_public_api,
        )
        for domain_factory in domain_factories
    )
    results_by_type = zip(*results_by_domain)
    return tuple(sum(results_type) for results_type in results_by_type)


def evaluate_solver_on_several_domains_with_timeout(
    program_path: str,
    domain_factories: list[Callable[[], Domain]],
    max_steps: Union[int, Callable[[Domain], int]] = 100,
    num_episodes: int = 3,
    normalize: bool = True,
    render: bool = False,
    timeout: int = 60,
    enforce_using_public_api: bool = False,
) -> tuple[float, ...]:
    """Evaluate an evolved solver on several domain instances, with timeout.

    We evaluate by performing a rollout on domains and taking the total cost, potentially normalized by max_steps.

    # Parameters
    program_path: path to the program to evaluate on the given domain
    domain_factories: domain factories on which rolling out
    max_steps: maximum number of steps per episode.
        Either an integer or a callable mapping a domain to an integer.
    num_episodes: number of episodes to do per domain
    normalize: whether to normalize cost by max steps (potentially related to domain size)
    render: render the domain during rollout
    timeout: seconds before timeout
    enforce_using_public_api: if True, wraps the domains so that only their public API can be seen by the program

    # Returns
    total cost of the rollout, potentially normalized by max_steps

    """
    return run_with_timeout(
        func=evaluate_solver_on_several_domains,
        program_path=program_path,
        domain_factories=domain_factories,
        max_steps=max_steps,
        num_episodes=num_episodes,
        normalize=normalize,
        render=render,
        timeout_seconds=timeout,
        enforce_using_public_api=enforce_using_public_api,
    )


D = Domain


class EvolvedPolicies(Policies):
    """Policies wrapper around evolved program."""

    def __init__(
        self,
        domain: Domain,
        program_path: str,
        enforce_using_public_api: bool = False,
    ):
        """Initialize the wrapper

        # Parameters
        program_path: path to the program to evaluate
        domain: domain to pass to the wrapped planner
        enforce_using_public_api: if True, wraps the domain so that only its public API can be seen by the program

        """
        # retrieve the evolved Planner (and its method) sample_action()
        module_name = "evolved_program"
        spec = importlib.util.spec_from_file_location(module_name, program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.program_path = program_path
        evolved_planner_cls = module.Planner

        # get static arguments for sample_action
        self._unrestricted_actions = isinstance(domain, UnrestrictedActions)
        if enforce_using_public_api:
            domain = create_public_proxy(domain)
        self._planner = evolved_planner_cls(domain=domain)

    def _sample_action(
        self, observation: D.T_agent[D.T_observation], domain: Optional[Domain] = None
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        if self._unrestricted_actions:
            return self._planner.sample_action(
                observation,
            )
        else:
            applicable_actions = domain.get_applicable_actions()
            return self._planner.sample_action(observation, applicable_actions)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True


def evaluate(
    program_path: str,
    domain_factories: list[Callable[[], Domain]],
    max_steps: Union[int, Callable[[Domain], int]] = 100,
    num_episodes: int = 3,
    normalize: bool = True,
    render: bool = False,
    timeout: int = 60,
    enforce_using_public_api: bool = False,
) -> EvaluationResult:
    """Evaluate function to use in openevolve evolution.

    Calls `evaluate_solver_on_several_domains_with_timeout` and wraps result in the proper
    openevolve object.

    # Parameters
    program_path: path to the program to evaluate on the given domain
    domain_factories: domain factories on which rolling out
    max_steps: maximum number of steps per episode.
        Either an integer or a callable mapping a domain to an integer.
    num_episodes: number of episodes to do per domain
    normalize: whether to normalize cost by max steps (potentially related to domain size)
    render: render the domain during rollout
    timeout: seconds before timeout

    # Returns
    combined_score: `-inf` for incorrect programs, else `-total_cost`

    """
    try:
        res = evaluate_solver_on_several_domains_with_timeout(
            program_path=program_path,
            domain_factories=domain_factories,
            max_steps=max_steps,
            num_episodes=num_episodes,
            normalize=normalize,
            render=render,
            timeout=timeout,
            enforce_using_public_api=enforce_using_public_api,
        )
    except Exception as e:
        logger.info(f"Error during evaluation: {type(e).__name__}: {e}")
        return EvaluationResult(
            metrics=dict(combined_score=-float("inf")),
            artifacts=dict(
                error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()
            ),
        )
    if len(res) > 1:
        cost, nb_reached_goals = res
        return EvaluationResult(
            metrics=dict(
                cost=cost,
                nb_reached_goals=nb_reached_goals,
                combined_score=-cost,  # openevolve maximize the combined score
            )
        )
    else:
        (cost,) = res
        return EvaluationResult(
            metrics=dict(
                cost=cost,
                combined_score=-cost,  # openevolve maximize the combined score
            )
        )
