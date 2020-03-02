# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module contains utility functions."""
from __future__ import annotations

import re
import time
import simplejson as json
import os
import copy
import logging
import datetime
from typing import Any, Union, Optional, Type, Iterable, Tuple, List, Dict, Callable

from pkg_resources import iter_entry_points, EntryPoint, DistributionNotFound

from skdecide import Domain, Solver, D, Space, EnvironmentOutcome, autocast_all, autocastable
from skdecide.builders.domain import Goals, Markovian, Renderable, FullyObservable
from skdecide.builders.solver import Policies

__all__ = ['get_registered_domains', 'get_registered_solvers', 'load_registered_domain', 'load_registered_solver',
           'match_solvers', 'rollout']

logger = logging.getLogger('skdecide.utils')

logger.setLevel(logging.INFO)

if not len(logger.handlers):
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
    logger.propagate = False


def _get_registered_entries(entry_type: str) -> List[str]:
    return [e.name for e in iter_entry_points(entry_type)]


def _load_registered_entry(entry_type: str, entry_name: str) -> Optional[Any]:
    try:
        for entry_point in iter_entry_points(entry_type):
            if entry_point.name == entry_name:
                return entry_point.load()
        logger.error(rf'/!\ {entry_name} could not be loaded because it is not registered in group "{entry_type}".')
    except DistributionNotFound as e:
        logger.error(rf'/!\ {entry_name} could not be loaded because of missing dependency ({e}).')
        extra_match = re.search(r'\bextra\s*==\s*"(?P<extra>[^"]+)"', str(e))
        if extra_match:
            extra = extra_match.group('extra')
            logger.error(f'    ==> Try following command in your Python environment: pip install skdecide[{extra}]')
    except Exception as e:
        logger.error(rf'/!\ {entry_name} could not be loaded ({e}).')


def get_registered_domains() -> List[str]:
    return _get_registered_entries('skdecide.domains')


def get_registered_solvers() -> List[str]:
    return _get_registered_entries('skdecide.solvers')


def load_registered_domain(name: str) -> Type[Domain]:
    return _load_registered_entry('skdecide.domains', name)


def load_registered_solver(name: str) -> Type[Solver]:
    return _load_registered_entry('skdecide.solvers', name)


# TODO: implement ranking heuristic
def match_solvers(domain: Domain, candidates: Optional[Iterable[Type[Solver]]] = None, ranked: bool = False) -> Union[
    List[Type[Solver]], List[Tuple[Type[Solver], int]]]:
    if candidates is None:
        candidates = [load_registered_solver(s) for s in get_registered_solvers()]
        candidates = [c for c in candidates if c is not None]  # filter out None entries (failed loadings)
    matches = []
    for solver_type in candidates:
        if solver_type.check_domain(domain):
            matches.append(solver_type)
    return matches


def rollout(domain: Domain, solver: Optional[Solver] = None, from_memory: Optional[D.T_memory[D.T_state]] = None,
            from_action: Optional[D.T_agent[D.T_concurrency[D.T_event]]] = None, num_episodes: int = 1,
            max_steps: Optional[int] = None, render: bool = True, max_framerate: Optional[float] = None,
            verbose: bool = True,
            action_formatter: Optional[Callable[[D.T_event], str]] = lambda a: str(a),
            outcome_formatter: Optional[Callable[[EnvironmentOutcome], str]] = lambda o: str(o),
            save_result_directory: str = None) -> str:
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
    save_result: Directory in which state visited, actions applied and Transition Value are saved to json. 
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug('Logger is in verbose mode: all debug messages will be there for you to enjoy （〜^∇^ )〜')

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
            def sample_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
                return {agent: [space.sample()] for agent, space in self._domain.get_applicable_actions().items()}

            @autocastable
            def is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
                return True

        solver = RandomWalk()
        autocast_all(solver, solver.T_domain, domain)

    has_render = isinstance(domain, Renderable)
    has_goal = isinstance(domain, Goals)
    has_memory = not isinstance(domain, Markovian)
    for i_episode in range(num_episodes):
        # Initialize episode
        solver.reset()
        if from_memory is None:
            observation = domain.reset()
        else:
            domain.set_memory(from_memory)
            last_state = from_memory[-1] if has_memory else from_memory
            observation = domain.get_observation_distribution(last_state, from_action).sample()
        logger.debug(f'Episode {i_episode + 1} started with following observation:')
        logger.debug(observation)
        # Run episode
        step = 1

        if save_result_directory is not None:
            observations = dict()
            transitions = dict()
            actions = dict()
            # save the initial observation
            observations[0] = observation

        while max_steps is None or step <= max_steps:
            old_time = time.perf_counter()
            if render and has_render:
                domain.render()
            # assert solver.is_policy_defined_for(observation)
            if save_result_directory is not None:
                previous_observation = copy.deepcopy(observation)
            action = solver.sample_action(observation)
            if action_formatter is not None:
                logger.debug('Action: {}'.format(action_formatter(action)))
            outcome = domain.step(action)
            observation = outcome.observation
            if save_result_directory is not None:
                if isinstance(domain, FullyObservable):
                    observations[step] = observation
                    actions[step] = action
                    transitions[step] = {
                        "s": hash(previous_observation),
                        "a": hash(action),
                        "cost": outcome.value.cost,
                        "s'": hash(observation)
                    }
            if outcome_formatter is not None:
                logger.debug('Result: {}'.format(outcome_formatter(outcome)))
            if outcome.termination:
                logger.debug(f'Episode {i_episode + 1} terminated after {step + 1} steps.')
                break
            if max_framerate is not None:
                wait = 1 / max_framerate - (time.perf_counter() - old_time)
                if wait > 0:
                    time.sleep(wait)
            step += 1
        if render and has_render:
            domain.render()
        if has_goal:
            logger.info(f'The goal was{"" if domain.is_goal(observation) else " not"} reached '
                        f'in episode {i_episode + 1}.')
        if save_result_directory is not None:
            if not os.path.exists(save_result_directory):
                os.mkdir(save_result_directory)
            elif not os.path.isdir(save_result_directory):
                raise FileExistsError

            now = datetime.datetime.now()
            str_timestamp = now.strftime("%Y%m%dT%H%M%S")
            directory = os.path.join(save_result_directory, str_timestamp)
            os.mkdir(directory)
            try:
                with open(os.path.join(directory, 'actions.json'), 'w') as f:
                    json.dump(actions, f, indent=2)
            except TypeError:
                logger.error("Action is not serializable")
            try:
                with open(os.path.join(directory, 'transitions.json'), 'w') as f:
                    json.dump(transitions, f, indent=2)
            except TypeError:
                logger.error("Transition is not serializable")
            try:
                with open(os.path.join(directory, 'observations.json'), 'w') as f:
                    json.dump(observations, f, indent=2)
            except TypeError:
                logger.error("Observation is not serializable")

            return directory
