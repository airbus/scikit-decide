"""This module contains utility functions."""

from typing import Optional, Union, Callable
from typing import Tuple, List

from airlaps.builders.domain.goals import GoalDomain
from airlaps.builders.domain.renderability import RenderableDomain
from airlaps.core import T_state, T_event, Memory, EnvironmentOutcome, TransitionValue, T_observation
from airlaps.domains import Domain
from airlaps.solvers import Solver


def rollout(domain: Domain, solver: Solver, from_memory: Optional[Union[Memory[T_state], T_state]] = None,
            from_event: Optional[T_event] = None, num_episodes: int = 1, max_steps: Optional[int] = None,
            render: bool = True, verbose: bool = True,
            outcome_formatter: Callable[[EnvironmentOutcome], str] = lambda o: str(o)) -> None:
    """This method will run one or more episodes in a domain according to the policy of a solver.

    !!! tip
        If a state is passed as from_memory parameter, the boilerplate code will automatically wrap it in a Memory first
        (initialized according to the domain's memory characteristic).

    # Parameters
    domain: The domain in which the episode(s) will be run.
    solver: The solver whose policy will select actions to take.
    from_memory: The memory or state to consider as rollout starting point (if None, the domain is reset first).
    from_event: The last applied event when from_memory is used (if necessary for initial observation computation).
    num_episodes: The number of episodes to run.
    max_steps: The maximum number of steps for each episode (if None, no limit is set).
    render: Whether to render the episode(s) during rollout if the domain is renderable.
    verbose: Whether to print information to the console during rollout.
    outcome_formatter: The function transforming EnvironmentOutcome objects into the string representation to print.
    """
    for i_episode in range(num_episodes):
        # Initialize episode
        if from_memory is None:
            observation = domain.reset()
        else:
            domain.set_memory(from_memory)
            observation = domain.get_observation_distribution(from_memory[-1], from_event).sample()
        if verbose:
            print(f'Episode {i_episode + 1} started.')
            print(observation)
        # Run episode
        step = 0
        while max_steps is None or step < max_steps:
            if render and isinstance(domain, RenderableDomain):
                domain.render()
            action = solver.sample_action(Memory([observation]))
            outcome = domain.step(action)
            observation = outcome.observation
            print(outcome_formatter(outcome))
            if outcome.termination:
                if verbose:
                    print(f'Episode {i_episode + 1} terminated after {step + 1} steps.')
                break
            step += 1
        if verbose and isinstance(domain, GoalDomain):
            print(f'The goal was{"" if observation in domain.get_goals() else " not"} reached '
                  f'in episode {i_episode + 1}.')


# TODO: replace rollout_saver() by additional features on rollout()
def rollout_saver(domain: Domain, solver: Solver, from_memory: Optional[Union[Memory[T_state], T_state]] = None,
                  from_event: Optional[T_event] = None, num_episodes: int = 1, max_steps: Optional[int] = None,
                  render: bool = True, verbose: bool = True,
                  outcome_formatter: Callable[[EnvironmentOutcome], str] = lambda o: str(o)) -> Tuple[
        float, List[TransitionValue], List[T_observation], List[T_event]]:
    """This method will run one or more episodes in a domain according to the policy of a solver."""
    for i_episode in range(num_episodes):
        # Initialize episode
        if from_memory is None:
            observation = domain.reset()
        else:
            domain.set_memory(from_memory)
            observation = domain.get_observation_distribution(from_memory[-1], from_event).sample()
        if verbose:
            print(f'Episode {i_episode + 1} started.')
            print(observation)
        # Run episode
        step = 0
        total_cost = 0.
        cost = []
        obs = [observation]
        actions = []
        while max_steps is None or step < max_steps:
            if render and isinstance(domain, RenderableDomain):
                domain.render()
            action = solver.sample_action(Memory([observation]))
            outcome = domain.step(action)
            observation = outcome.observation
            cost += [outcome.value]
            total_cost += outcome.value.cost
            obs += [observation]
            actions += [action]
            if verbose:
                print(outcome_formatter(outcome))
            if outcome.termination:
                if verbose:
                    print(f'Episode {i_episode + 1} terminated after {step + 1} steps.')
                break
            step += 1
        if verbose and isinstance(domain, GoalDomain):
            print(f'The goal was{"" if observation in domain.get_goals() else " not"} reached '
                  f'in episode {i_episode + 1}.')
        return total_cost, cost, obs, actions
