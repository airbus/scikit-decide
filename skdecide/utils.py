# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module contains utility functions."""
from __future__ import annotations

import copy
import datetime
import importlib.metadata
import logging
import os
import sys
import time
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

from skdecide import (
    D,
    Domain,
    EnvironmentOutcome,
    Solver,
    Space,
    Value,
    autocast_all,
    autocastable,
)
from skdecide.builders.domain import (
    FullyObservable,
    Goals,
    Initializable,
    Markovian,
    Renderable,
)
from skdecide.builders.solver import DeterministicPolicies, Policies

__all__ = [
    "get_registered_domains",
    "get_registered_solvers",
    "load_registered_domain",
    "load_registered_solver",
    "match_solvers",
    "rollout",
]

SKDECIDE_DEFAULT_DATAHOME = "~/skdecide_data"
SKDECIDE_DEFAULT_DATAHOME_ENVVARNAME = "SKDECIDE_DATA"

logger = logging.getLogger("skdecide.utils")

logger.setLevel(logging.INFO)

if not len(logger.handlers):
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
    logger.propagate = False


def get_data_home(data_home: Optional[str] = None) -> str:
    """Return the path of the scikit-decide data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times, as for instance the weather data used by the flight planning domain.
    By default the data dir is set to a folder named 'skdecide_data' in the
    user home folder.
    Alternatively, it can be set by the 'SKDECIDE_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.

    Params:
        data_home : The path to scikit-decide data directory. If `None`, the default path
        is `~/skdecide_data`.

    """
    if data_home is None:
        data_home = os.environ.get(
            SKDECIDE_DEFAULT_DATAHOME_ENVVARNAME, SKDECIDE_DEFAULT_DATAHOME
        )
    data_home = os.path.expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return data_home


def _get_registered_entries(entry_type: str) -> List[str]:
    if (
        sys.version_info.minor < 10 and sys.version_info.major == 3
    ):  # different behaviour for 3.8 and 3.9
        return [e.name for e in importlib.metadata.entry_points()[entry_type]]
    else:
        return [e.name for e in importlib.metadata.entry_points(group=entry_type)]


def _load_registered_entry(entry_type: str, entry_name: str) -> Optional[Any]:
    if (
        sys.version_info.minor < 10 and sys.version_info.major == 3
    ):  # different behaviour for 3.8 and 3.9
        potential_entry_points = tuple(
            e
            for e in importlib.metadata.entry_points()[entry_type]
            if e.name == entry_name
        )
    else:
        potential_entry_points = tuple(
            importlib.metadata.entry_points(group=entry_type, name=entry_name)
        )
    if len(potential_entry_points) == 0:
        logger.warning(
            rf'/!\ {entry_name} could not be loaded because it is not registered in group "{entry_type}".'
        )
    else:
        try:
            return potential_entry_points[0].load()
        except Exception as e:
            logger.warning(rf"/!\ {entry_name} could not be loaded ({e}).")


def get_registered_domains() -> List[str]:
    return _get_registered_entries("skdecide.domains")


def get_registered_solvers() -> List[str]:
    return _get_registered_entries("skdecide.solvers")


def load_registered_domain(name: str) -> Type[Domain]:
    return _load_registered_entry("skdecide.domains", name)


def load_registered_solver(name: str) -> Type[Solver]:
    return _load_registered_entry("skdecide.solvers", name)


# TODO: implement ranking heuristic
def match_solvers(
    domain: Domain,
    candidates: Optional[Iterable[Type[Solver]]] = None,
    ranked: bool = False,
) -> Union[List[Type[Solver]], List[Tuple[Type[Solver], int]]]:
    if candidates is None:
        candidates = [load_registered_solver(s) for s in get_registered_solvers()]
        candidates = [
            c for c in candidates if c is not None
        ]  # filter out None entries (failed loadings)
    matches = []
    for solver_type in candidates:
        if solver_type.check_domain(domain):
            matches.append(solver_type)
    return matches


class ReplayOutOfActionMethod(Enum):
    LOOP = "loop"
    LAST = "last"
    ERROR = "error"


class ReplaySolver(DeterministicPolicies):
    """Wrapper around a list of actions mimicking a computed policy.

    The goal is to be able to replay a rollout from a previous episode.

    # Attributes
    actions: list of actions to wrap
    out_of_action_method: method to use when we run out of actions
      - LOOP: we loop on actions, beginning back with the first action,
      - LAST: we keep returning the last action,
      - ERROR: we raise a RuntimeError.

    # Example
    ```python
    # rollout with actual solver
    episodes = rollout(
        domain,
        solver,
        return_episodes=True
    )
    # take the first episode
    observations, actions, values = episodes[0]
    # wrap the corresponding actions in a replay solver
    replay_solver = ReplaySolver(actions)
    # replay the rollout
    replayed_episodes = rollout(
        domain=domain,
        solver=replay_solver,
        return_episodes=True
    )
    # same outputs (for deterministic domain)
    assert episodes == replayed_episodes
    ```

    """

    def __init__(
        self,
        actions: List[D.T_agent[D.T_concurrency[D.T_event]]],
        out_of_action_method: ReplayOutOfActionMethod = ReplayOutOfActionMethod.LAST,
    ):
        self.actions = actions
        self.out_of_action_method = out_of_action_method
        self._i_action = 0

    @autocastable
    def reset(self) -> None:
        self._i_action = 0

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        if self._i_action >= len(self.actions):
            if self.out_of_action_method == ReplayOutOfActionMethod.LOOP:
                self._i_action = 0
            elif self.out_of_action_method == ReplayOutOfActionMethod.LAST:
                self._i_action = len(self.actions) - 1
            else:
                raise RuntimeError(
                    "You require more actions than available in the registered plan!"
                )

        action = self.actions[self._i_action]
        self._i_action += 1
        return action


def rollout(
    domain: Domain,
    solver: Optional[Union[Solver, Policies]] = None,
    from_memory: Optional[D.T_memory[D.T_state]] = None,
    from_action: Optional[D.T_agent[D.T_concurrency[D.T_event]]] = None,
    num_episodes: int = 1,
    max_steps: Optional[int] = None,
    render: bool = True,
    max_framerate: Optional[float] = None,
    verbose: bool = True,
    action_formatter: Optional[Callable[[D.T_event], str]] = lambda a: str(a),
    outcome_formatter: Optional[Callable[[EnvironmentOutcome], str]] = lambda o: str(o),
    return_episodes: bool = False,
    goal_logging_level: int = logging.INFO,
    rollout_callback: Optional[RolloutCallback] = None,
) -> Optional[
    List[
        Tuple[
            List[D.T_agent[D.T_observation]],
            List[D.T_agent[D.T_concurrency[D.T_event]]],
            List[D.T_agent[Value[D.T_value]]],
        ]
    ]
]:
    """This method will run one or more episodes in a domain according to the policy of a solver.

    # Parameters
    domain: The domain in which the episode(s) will be run.
    solver: The solver whose policy will select actions to take (if None, a random policy is used).
    from_memory: The memory or state to consider as rollout starting point (if None, the domain is reset first).
    from_action: The last applied action when from_memory is used (if necessary for initial observation computation).
    num_episodes: The number of episodes to run.
    max_steps: The maximum number of steps for each episode (if None, no limit is set).
    render: Whether to render the episode(s) during rollout if the domain is renderable.
    max_framerate: The maximum number of steps/renders per second (if None, steps/renders are never slowed down).
    verbose: Whether to print information to the console during rollout.
    action_formatter: The function transforming actions in the string to print (if None, no print).
    outcome_formatter: The function transforming EnvironmentOutcome objects in the string to print (if None, no print).
    return_episodes: if True, return the list of episodes, each episode as a tuple of observations, actions, and values.
        else return nothing.
    goal_logging_level: logging level at which we want to display if goal has been reached or not
    """
    previous_log_level = logger.level
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug(
            "Logger is in verbose mode: all debug messages will be there for you to enjoy （〜^∇^ )〜"
        )

    if rollout_callback is None:
        rollout_callback = RolloutCallback()

    if solver is None:
        # Create solver-like random walker that works for any domain
        class RandomWalk(Policies):
            T_domain = Domain
            T_agent = Domain.T_agent
            T_event = Domain.T_event

            def __init__(self):
                class CastDomain:  # trick to autocast domain's get_applicable_actions() without mutating domain
                    T_agent = domain.T_agent
                    T_event = domain.T_event

                    @autocastable
                    def get_applicable_actions(self) -> D.T_agent[Space[D.T_event]]:
                        return domain.get_applicable_actions()

                self._domain = CastDomain()
                autocast_all(self._domain, self._domain, self)

            @autocastable
            def reset(self) -> None:
                pass

            @autocastable
            def sample_action(
                self, observation: D.T_agent[D.T_observation]
            ) -> D.T_agent[D.T_concurrency[D.T_event]]:
                return {
                    agent: [space.sample()]
                    for agent, space in self._domain.get_applicable_actions().items()
                }

            @autocastable
            def is_policy_defined_for(
                self, observation: D.T_agent[D.T_observation]
            ) -> bool:
                return True

        solver = RandomWalk()
        autocast_all(solver, solver.T_domain, domain)

    episodes: List[
        Tuple[
            List[D.T_agent[D.T_observation]],
            List[D.T_agent[D.T_concurrency[D.T_event]]],
            List[D.T_agent[Value[D.T_value]]],
        ]
    ] = []

    if num_episodes > 1 and from_memory is None and not hasattr(domain, "reset"):
        raise ValueError(
            "If from_memory is not specified and domain has no reset() method, "
            "num_episodes should be equal to 1."
        )

    rollout_callback.at_rollout_start()

    has_render = isinstance(domain, Renderable)
    has_goal = isinstance(domain, Goals)
    has_memory = not isinstance(domain, Markovian)
    for i_episode in range(num_episodes):
        rollout_callback.at_episode_start()

        # Initialize episode
        solver.reset()
        if from_memory is None:
            if isinstance(domain, Initializable):
                observation = domain.reset()
            else:
                raise ValueError(
                    "The domain must be initializable if from_memory is None."
                )
        else:
            if hasattr(domain, "set_memory"):
                domain.set_memory(from_memory)
                last_state = from_memory[-1] if has_memory else from_memory
                observation = domain.get_observation_distribution(
                    last_state, from_action
                ).sample()
            else:
                raise ValueError(
                    "from_memory must be None if domain has no set_memory() method."
                )
        logger.debug(f"Episode {i_episode + 1} started with following observation:")
        logger.debug(observation)
        # Run episode
        step = 1

        observations: List[D.T_agent[D.T_observation]] = []
        actions: List[D.T_agent[D.T_concurrency[D.T_event]]] = []
        values: List[D.T_agent[Value[D.T_value]]] = []
        # save the initial observation
        observations.append(observation)

        while max_steps is None or step <= max_steps:
            old_time = time.perf_counter()
            if render and has_render:
                domain.render()
            action = solver.sample_action(observation)
            if action_formatter is not None:
                logger.debug("Action: {}".format(action_formatter(action)))
            outcome = domain.step(action)
            observation = outcome.observation
            if return_episodes:
                observations.append(observation)
                actions.append(action)
                values.append(outcome.value)
            if outcome_formatter is not None:
                logger.debug("Result: {}".format(outcome_formatter(outcome)))
            termination = (
                outcome.termination
                if domain.T_agent == Union
                else all(t for a, t in outcome.termination.items())
            )
            if termination:
                logger.debug(
                    f"Episode {i_episode + 1} terminated after {step + 1} steps."
                )
                break
            # user callback -> stopping?
            stopping = rollout_callback.at_episode_step(
                i_episode=i_episode,
                step=step,
                domain=domain,
                solver=solver,
                action=action,
                outcome=outcome,
            )
            if stopping:
                break
            if max_framerate is not None:
                wait = 1 / max_framerate - (time.perf_counter() - old_time)
                if wait > 0:
                    time.sleep(wait)
            step += 1

        if render and has_render:
            domain.render()
        if has_goal:
            logger.log(
                goal_logging_level,
                f'The goal was{"" if domain.is_goal(observation) else " not"} reached '
                f"in episode {i_episode + 1}.",
            )
        if return_episodes:
            episodes.append((observations, actions, values))
        rollout_callback.at_episode_end()
    rollout_callback.at_rollout_end()
    if verbose:
        logger.setLevel(previous_log_level)
    if return_episodes:
        return episodes


class RolloutCallback:
    """Callback used during rollout to add custom behaviour.

    One should derives from this one in order to hook in different stages of the rollout.

    """

    def at_rollout_start(self):
        """Called at rollout start."""
        ...

    def at_rollout_end(self):
        """Called at rollout end."""
        ...

    def at_episode_start(self):
        """Called before each episode."""
        ...

    def at_episode_end(self):
        """Called after each episode."""
        ...

    def at_episode_step(
        self,
        i_episode: int,
        step: int,
        domain: Domain,
        solver: Union[Solver, Policies],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        outcome: EnvironmentOutcome[
            D.T_agent[D.T_observation],
            D.T_agent[Value[D.T_value]],
            D.T_agent[D.T_predicate],
            D.T_agent[D.T_info],
        ],
    ) -> bool:
        """

        # Parameters
        i_episode: current episode number
        step: current step number within the episode
        domain: domain considered
        solver: solver considered (or randomwalk policy if solver was None in rollout)
        action: last action sampled
        outcome: outcome of the last action applied to the domain

        # Returns
        stopping: if True, the rollout for the current episode stops and the next episode starts.

        """
        return False
