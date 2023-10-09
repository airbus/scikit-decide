from typing import Callable

import gymnasium as gym

from skdecide.hub.domain.gym import (
    GymDiscreteActionDomain,
    GymDomain,
    GymPlanningDomain,
    GymWidthDomain,
)
from skdecide.hub.domain.gym.gym import AsGymnasiumEnv
from skdecide.hub.domain.maze.maze import Maze
from skdecide.hub.solver.cgp import CGP


class D(GymPlanningDomain, GymWidthDomain, GymDiscreteActionDomain):
    pass


class GymDomainForWidthSolvers(D):
    def __init__(
        self,
        gym_env: gym.Env,
        set_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
        get_state: Callable[[gym.Env], D.T_memory[D.T_state]] = None,
        termination_is_goal: bool = True,
        continuous_feature_fidelity: int = 5,
        discretization_factor: int = 3,
        branching_factor: int = None,
        max_depth: int = 1000,
    ) -> None:
        GymPlanningDomain.__init__(
            self,
            gym_env=gym_env,
            set_state=set_state,
            get_state=get_state,
            termination_is_goal=termination_is_goal,
            max_depth=max_depth,
        )
        GymDiscreteActionDomain.__init__(
            self,
            discretization_factor=discretization_factor,
            branching_factor=branching_factor,
        )
        GymWidthDomain.__init__(
            self, continuous_feature_fidelity=continuous_feature_fidelity
        )
        gym_env._max_episode_steps = max_depth


def test_gymdomain():
    ENV_NAME = "MountainCarContinuous-v0"
    domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
    solver = CGP("TEMP_CGP", n_it=2, verbose=False)
    GymDomain.solve_with(solver, domain_factory)
    domain = domain_factory()
    observation = domain.reset()
    domain.render()
    domain.step(action=solver.get_next_action(observation))
    domain.render()


def test_gymdomain4iw():
    ENV_NAME = "MountainCarContinuous-v0"
    domain = GymDomainForWidthSolvers(gym.make(ENV_NAME, render_mode="rgb_array"))
    domain.reset()
    domain.render()
    domain.step(action=domain.get_action_space().sample())


def test_asgymnasiumenv():
    domain = Maze()
    domain.reset()
    env = AsGymnasiumEnv(domain=domain, render_mode="human")
    env.reset()
    env.step(env.action_space.sample())
    env.render()
    env.close()
