# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Example mixing various domains with various solvers.

NB: to be able to launch this example, you need to install scikit-decide with all optional dependencies
    + atari and autorom to play pacman (see https://gymnasium.farama.org/environments/atari/).

    ```
    pip install scikit-decide[all]
    pip install gymnasium[atari,accept-rom-license]
    ```

    In doing so (`pip install gymnasium[accept-rom-license]`), you agree to own a license to these Atari 2600 ROMs
    and agree to not distribution these ROMS.

    If you still does not have the ROMs after that, and getting the following error:

    > gymnasium.error.Error: We're Unable to find the game "MsPacman". Note: Gymnasium no longer distributes ROMs.

    it may be due to a (silent) ssl error that can be resolved by setting the environment variable CURL_CA_BUNDLE
    to the proper certificate (https://stackoverflow.com/a/31060428).

"""
from dataclasses import dataclass
from math import sqrt
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from skdecide import Value
from skdecide.hub.domain.gym import (
    GymDiscreteActionDomain,
    GymPlanningDomain,
    GymWidthDomain,
)
from skdecide.utils import (
    load_registered_domain,
    load_registered_solver,
    match_solvers,
    rollout,
)


class D(GymPlanningDomain, GymWidthDomain, GymDiscreteActionDomain):
    pass


class GymDomainForWidthSolvers(D):
    def __init__(
        self,
        gym_env: gym.Env,
        set_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
        get_state: Callable[[gym.Env], D.T_memory[D.T_state]] = None,
        gym_env_for_rendering: Optional[gym.Env] = None,
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
            gym_env_for_rendering=gym_env_for_rendering,
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

    def state_features(self, s):
        return self.bee2_features(s)

    def heuristic(self, s):
        return Value(cost=0)


def get_state_continuous_mountain_car(env):
    return env.unwrapped.state


def set_state_continuous_mountain_car(env, state):
    env.unwrapped.state = state


@dataclass
class CartPoleState:
    state: np.array
    steps_beyond_terminated: Optional[int]

    def __eq__(self, other: "CartPoleState"):
        return (
            np.array_equal(self.state, other.state)
            and self.steps_beyond_terminated == other.steps_beyond_terminated
        )


def get_state_cart_pole(env):
    return CartPoleState(
        state=env.unwrapped.state,
        steps_beyond_terminated=env.unwrapped.steps_beyond_terminated,
    )


def set_state_get_state_cart_pole(env, state: CartPoleState):
    env.unwrapped.state = state.state
    env.unwrapped.steps_beyond_terminated = state.steps_beyond_terminated


if __name__ == "__main__":

    try_domains = [
        # Simple Grid World
        {
            "name": "Simple Grid World",
            "entry": "SimpleGridWorld",
            "config": {},
            "rollout": {
                "max_steps": 1000,
                "outcome_formatter": lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
            },
        },
        # Maze
        {
            "name": "Maze",
            "entry": "Maze",
            "config": {},
            "rollout": {
                "max_steps": 1000,
                "max_framerate": 30,
                "outcome_formatter": lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
            },
        },
        # Mastermind
        {
            "name": "Mastermind",
            "entry": "MasterMind",
            "config": {"n_colours": 3, "n_positions": 3},
            "rollout": {
                "max_steps": 1000,
                "outcome_formatter": lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
            },
        },
        # Cart Pole (Gymnasium)
        {
            "name": "Cart Pole (Gymnasium)",
            "entry": "GymDomain",
            "config": dict(gym_env=gym.make("CartPole-v1")),  # for solving
            "config_rollout": dict(  # for rollout
                gym_env=gym.make("CartPole-v1", render_mode="human")
            ),
            "config_gym4width": dict(  # for solving with GymForWidthSolvers
                gym_env=gym.make("CartPole-v1"),
                get_state=get_state_cart_pole,
                set_state=set_state_get_state_cart_pole,
                gym_env_for_rendering=gym.make("CartPole-v1", render_mode="human"),
            ),
            "config_gym4width_rollout": dict(),  # for rollout with GymForWidthSolvers
            "rollout": {
                "num_episodes": 3,
                "max_steps": 1000,
                "max_framerate": None,
                "outcome_formatter": None,
            },
        },
        # Mountain Car continuous (Gymnasium)
        {
            "name": "Mountain Car continuous (Gymnasium)",
            "entry": "GymDomain",
            "config": dict(gym_env=gym.make("MountainCarContinuous-v0")),  # for solving
            "config_rollout": dict(  # for rollout
                gym_env=gym.make("MountainCarContinuous-v0", render_mode="human")
            ),
            "config_gym4width": dict(  # for solving with GymForWidthSolvers
                gym_env=gym.make("MountainCarContinuous-v0"),
                get_state=get_state_continuous_mountain_car,
                set_state=set_state_continuous_mountain_car,
                gym_env_for_rendering=gym.make(
                    "MountainCarContinuous-v0", render_mode="human"
                ),
            ),
            "config_gym4width_rollout": dict(),  # for rollout with GymForWidthSolvers
            "rollout": {
                "num_episodes": 3,
                "max_steps": 1000,
                "max_framerate": None,
                "outcome_formatter": None,
            },
        },
        # ATARI Pacman (Gymnasium)
        {
            "name": "ATARI Pacman (Gymnasium)",
            "entry": "GymDomain",
            "config": {"gym_env": gym.make("ALE/MsPacman-v5", render_mode="human")},
            "rollout": {
                "num_episodes": 3,
                "max_steps": 1000,
                "max_framerate": 30,
                "outcome_formatter": None,
            },
        },
    ]

    try_solvers = [
        # Simple greedy
        {
            "name": "Simple greedy",
            "entry": "SimpleGreedy",
            "config": {},
        },
        # Lazy A* (classical planning)
        {
            "name": "Lazy A* (classical planning)",
            "entry": "LazyAstar",
            "config": {"heuristic": lambda d, s: d.heuristic(s), "verbose": False},
        },
        # A* (planning)
        {
            "name": "A* (planning)",
            "entry": "Astar",
            "config": {
                "heuristic": lambda d, s: d.heuristic(s),
                "parallel": False,
                "verbose": False,
            },
        },
        # LRTA* (classical planning)
        {
            "name": "LRTAStar",
            "entry": "LRTAstar",
            "config": {
                "max_depth": 200,
                "max_iter": 1000,
                "heuristic": lambda d, s: d.heuristic(s),
                "verbose": True,
            },
        },
        # UCT (reinforcement learning / search)
        {
            "name": "UCT (reinforcement learning / search)",
            "entry": "UCT",
            "config": {
                "time_budget": 200,
                "rollout_budget": 100000,
                "heuristic": lambda d, s: (d.heuristic(s), 10000),
                "online_node_garbage": True,
                "max_depth": 1000,
                "ucb_constant": 1.0 / sqrt(2.0),
                "parallel": False,
                "verbose": False,
            },
        },
        # PPO: Proximal Policy Optimization (deep reinforcement learning)
        {
            "name": "PPO: Proximal Policy Optimization (deep reinforcement learning)",
            "entry": "StableBaseline",
            "config": {
                "algo_class": PPO,
                "baselines_policy": "MlpPolicy",
                "learn_config": {"total_timesteps": 30000},
                "verbose": 1,
            },
        },
        # POMCP: Partially Observable Monte-Carlo Planning (online planning for POMDP)
        {
            "name": "POMCP: Partially Observable Monte-Carlo Planning (online planning for POMDP)",
            "entry": "POMCP",
            "config": {},
        },
        # CGP: Cartesian Genetic Programming (evolution strategy)
        {
            "name": "CGP: Cartesian Genetic Programming (evolution strategy)",
            "entry": "CGP",
            "config": {"folder_name": "TEMP", "n_it": 25},
        },
        # Rollout-IW (classical planning)
        {
            "name": "Rollout-IW (classical planning)",
            "entry": "RIW",
            "config": {
                "state_features": lambda d, s: d.state_features(s),
                "use_state_feature_hash": False,
                "use_simulation_domain": True,
                "time_budget": 200,
                "rollout_budget": 100000,
                "max_depth": 1000,
                "exploration": 0.25,
                "online_node_garbage": True,
                "continuous_planning": True,
                "parallel": False,
                "verbose": False,
            },
        },
        # IW (classical planning)
        {
            "name": "IW (classical planning)",
            "entry": "IW",
            "config": {
                "state_features": lambda d, s: d.state_features(s),
                "node_ordering": lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: a_novelty
                > b_novelty,
                "parallel": False,
                "verbose": False,
            },
        },
        # BFWS (classical planning)
        {
            "name": "BFWS (planning) - (num_rows * num_cols) binary encoding (1 binary variable <=> 1 cell)",
            "entry": "BFWS",
            "config": {
                "state_features": lambda d, s: d.state_features(s),
                "heuristic": lambda d, s: d.heuristic(s),
                "parallel": False,
                "verbose": False,
            },
        },
    ]

    # Load domains (filtering out badly installed ones)
    domains = map(
        lambda d: dict(d, entry=load_registered_domain(d["entry"])), try_domains
    )
    domains = list(filter(lambda d: d["entry"] is not None, domains))

    # Load solvers (filtering out badly installed ones)
    solvers = map(
        lambda s: dict(s, entry=load_registered_solver(s["entry"])), try_solvers
    )
    solvers = list(filter(lambda s: s["entry"] is not None, solvers))
    solvers.insert(
        0, {"name": "Random Walk", "entry": None}
    )  # Add Random Walk as option

    # Run loop to ask user input
    solver_candidates = [s["entry"] for s in solvers if s["entry"] is not None]
    while True:
        # Ask user input to select domain
        domain_choice = int(
            input(
                "\nChoose a domain:\n{domains}\n".format(
                    domains="\n".join(
                        [f'{i + 1}. {d["name"]}' for i, d in enumerate(domains)]
                    )
                )
            )
        )
        selected_domain = domains[domain_choice - 1]
        domain_type = selected_domain["entry"]
        domain = domain_type(**selected_domain["config"])

        while True:
            # Match solvers compatible with selected domain
            compatible = [None] + match_solvers(domain, candidates=solver_candidates)
            if (
                selected_domain["name"] == "Cart Pole (Gymnasium)"
                or selected_domain["name"] == "Mountain Car continuous (Gymnasium)"
            ):
                # Those gym domain actually have more capabilities than they pretend,
                # so we will transform them later to GymDomainForWidthSolvers (which
                # includes planning domains that UCT can solve)
                compatible += [
                    load_registered_solver("IW"),
                    load_registered_solver("RIW"),
                    load_registered_solver("UCT"),
                ]

            # Ask user input to select compatible solver
            solver_choice = int(
                input(
                    "\nChoose a compatible solver:\n{solvers}\n".format(
                        solvers="\n".join(
                            ["0. [Change domain]"]
                            + [
                                f'{i + 1}. {s["name"]}'
                                for i, s in enumerate(solvers)
                                if s["entry"] in compatible
                            ]
                        )
                    )
                )
            )

            if solver_choice == 0:  # the user wants to change domain
                break
            else:
                selected_solver = solvers[solver_choice - 1]
                solver_type = selected_solver["entry"]
                # Set the domain-dependent heuristic for search algorithms
                if selected_domain["name"] == "Simple Grid World":
                    setattr(
                        domain_type,
                        "heuristic",
                        lambda self, s: Value(
                            cost=sqrt(
                                (self.num_cols - 1 - s.x) ** 2
                                + (self.num_rows - 1 - s.y) ** 2
                            )
                        ),
                    )
                elif selected_domain["name"] == "Maze":
                    setattr(
                        domain_type,
                        "heuristic",
                        lambda self, s: Value(
                            cost=sqrt(
                                (self._goal.x - s.x) ** 2 + (self._goal.y - s.y) ** 2
                            )
                        ),
                    )
                elif selected_domain["name"] == "Cart Pole (Gymnasium)":
                    setattr(domain_type, "heuristic", lambda self, s: Value(cost=1))
                elif selected_domain["name"] == "Mountain Car continuous (Gymnasium)":
                    setattr(domain_type, "heuristic", lambda self, s: Value(cost=150))
                else:
                    setattr(domain_type, "heuristic", lambda self, s: Value(cost=0))
                # Set the domain-dependent stat features for width-based algorithms
                if selected_domain["name"] == "Simple Grid World":
                    setattr(domain_type, "state_features", lambda self, s: [s.x, s.y])
                elif selected_domain["name"] == "Maze":
                    setattr(domain_type, "state_features", lambda self, s: [s.x, s.y])
                elif selected_domain["entry"].__name__ == "GymDomain":
                    setattr(
                        domain_type,
                        "state_features",
                        lambda self, s: self.bee1_features(s),
                    )
                else:
                    setattr(domain_type, "state_features", lambda self, s: s)
                # Test solver solution on domain
                print("==================== TEST SOLVER ====================")
                # Check if Random Walk selected or other
                if solver_type is None:
                    rollout_domain_config = dict(selected_domain["config"])
                    if "config_rollout" in selected_domain:
                        rollout_domain_config.update(
                            selected_domain["config_rollout"]
                        )  # specificities for rollout
                    rollout_domain = domain_type(**rollout_domain_config)
                    rollout(rollout_domain, solver=None, **selected_domain["rollout"])
                else:
                    # Solve with selected solver
                    actual_domain_type = domain_type
                    actual_domain_config = dict(selected_domain["config"])  # copy
                    rollout_domain_config = dict(actual_domain_config)  # copy
                    if "config_rollout" in selected_domain:
                        rollout_domain_config.update(
                            selected_domain["config_rollout"]
                        )  # specificities for rollout
                    if selected_domain["entry"].__name__ == "GymDomain" and (
                        selected_solver["entry"].__name__ == "IW"
                        or selected_solver["entry"].__name__ == "RIW"
                        or selected_solver["entry"].__name__ == "BFWS"
                        or selected_solver["entry"].__name__ == "UCT"
                    ):
                        actual_domain_type = GymDomainForWidthSolvers
                        if selected_domain["name"] == "Cart Pole (Gymnasium)":
                            actual_domain_config["termination_is_goal"] = False
                        if "config_gym4width" in selected_domain:
                            actual_domain_config.update(
                                selected_domain[
                                    "config_gym4width"
                                ]  # specificities for GymDomainForWidthSolvers
                            )
                        rollout_domain_config = dict(actual_domain_config)  # copy
                        if "config_gym4width_rollout" in selected_domain:
                            rollout_domain_config.update(
                                selected_domain[
                                    "config_gym4width_rollout"
                                ]  # specificities for rollout
                            )
                    selected_solver["config"][
                        "domain_factory"
                    ] = lambda: actual_domain_type(**actual_domain_config)

                    with solver_type(**selected_solver["config"]) as solver:
                        solver.solve()
                        rollout_domain = actual_domain_type(**rollout_domain_config)
                        rollout(rollout_domain, solver, **selected_domain["rollout"])

                if hasattr(domain, "close"):
                    # error when relaunching due to pygame.quit() called by classic control gymnasium environments
                    # domain.close()
                    pass
