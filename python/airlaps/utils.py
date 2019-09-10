"""This module contains utility functions."""
from __future__ import annotations

import time
from typing import Union, Optional, Type, Iterable, Tuple, List, Callable

from airlaps import hub, Domain, Solver, D, Space, EnvironmentOutcome, autocast_all, autocastable
from airlaps.builders.domain import Goals, Markovian, Renderable
from airlaps.builders.solver import Policies

__all__ = ['match_solvers', 'rollout']


# TODO: implement ranking heuristic
def match_solvers(domain: Domain, candidates: Optional[Iterable[Type[Solver]]] = None, add_local_hub: bool = True,
                  ranked: bool = False) -> Union[List[Type[Solver]], List[Tuple[Type[Solver], int]]]:

    if candidates is None:
        candidates = set()
    else:
        candidates = set(candidates)
    if add_local_hub:
        candidates |= set(hub.local_search(Solver))
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
            outcome_formatter: Optional[Callable[[EnvironmentOutcome], str]] = lambda o: str(o)) -> None:
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
    """
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
        if verbose:
            print(f'Episode {i_episode + 1} started with following observation:')
            print(observation)
        # Run episode
        step = 0
        while max_steps is None or step < max_steps:
            old_time = time.perf_counter()
            if render and has_render:
                domain.render()
            # assert solver.is_policy_defined_for(observation)
            action = solver.sample_action(observation)
            if action_formatter is not None:
                print('Action:', action_formatter(action))
            outcome = domain.step(action)
            observation = outcome.observation
            if outcome_formatter is not None:
                print('Result:', outcome_formatter(outcome))
            if outcome.termination:
                if verbose:
                    print(f'Episode {i_episode + 1} terminated after {step + 1} steps.')
                break
            if max_framerate is not None:
                wait = 1/max_framerate - (time.perf_counter() - old_time)
                if wait > 0:
                    time.sleep(wait)
            step += 1
        if render and has_render:
            domain.render()
        if verbose and has_goal:
            print(f'The goal was{"" if observation in domain.get_goals() else " not"} reached '
                  f'in episode {i_episode + 1}.')

# # TODO: replace rollout_saver() by additional features on rollout()
# def rollout_saver(domain: Domain, solver: Solver, from_memory: Optional[Union[Memory[T_state], T_state]] = None,
#                   from_event: Optional[T_event] = None, num_episodes: int = 1, max_steps: Optional[int] = None,
#                   render: bool = True, verbose: bool = True,
#                   outcome_formatter: Callable[[EnvironmentOutcome], str] = lambda o: str(o)) -> Tuple[
#         float, List[TransitionValue], List[T_observation], List[T_event]]:
#     """This method will run one or more episodes in a domain according to the policy of a solver."""
#     for i_episode in range(num_episodes):
#         # Initialize episode
#         if from_memory is None:
#             observation = domain.reset()
#         else:
#             domain.set_memory(from_memory)
#             observation = domain.get_observation_distribution(from_memory[-1], from_event).sample()
#         if verbose:
#             print(f'Episode {i_episode + 1} started.')
#             print(observation)
#         # Run episode
#         step = 0
#         total_cost = 0.
#         cost = []
#         obs = [observation]
#         actions = []
#         while max_steps is None or step < max_steps:
#             if render and isinstance(domain, Renderable):
#                 domain.render()
#             action = solver.sample_action(Memory([observation]))
#             outcome = domain.step(action)
#             observation = outcome.observation
#             cost += [outcome.value]
#             total_cost += outcome.value.cost
#             obs += [observation]
#             actions += [action]
#             if verbose:
#                 print(outcome_formatter(outcome))
#             if outcome.termination:
#                 if verbose:
#                     print(f'Episode {i_episode + 1} terminated after {step + 1} steps.')
#                 break
#             step += 1
#         if verbose and isinstance(domain, Goals):
#             print(f'The goal was{"" if observation in domain.get_goals() else " not"} reached '
#                   f'in episode {i_episode + 1}.')
#         return total_cost, cost, obs, actions
