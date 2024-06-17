import logging
from typing import Union

from skdecide import D, Domain, EnvironmentOutcome, Solver, Value
from skdecide.builders.solver import Policies
from skdecide.hub.domain.maze import Maze
from skdecide.hub.solver.lazy_astar import LazyAstar
from skdecide.utils import (
    RolloutCallback,
    get_registered_domains,
    get_registered_solvers,
    load_registered_domain,
    load_registered_solver,
    rollout,
)

logger = logging.getLogger(__name__)


def test_get_registered_domains():
    domains = get_registered_domains()
    assert "Maze" in domains


def test_load_registered_domain():
    domain_class = load_registered_domain("NotExistingDomain")
    assert domain_class is None
    maze_class = load_registered_domain("Maze")
    assert maze_class is Maze


def test_get_registered_solvers():
    domains = get_registered_solvers()
    assert "LazyAstar" in domains


def test_load_registered_solver():
    solver_class = load_registered_solver("NotExistingSolver")
    assert solver_class is None
    lazyastar_class = load_registered_solver("LazyAstar")
    assert lazyastar_class is LazyAstar


class MyCallback(RolloutCallback):
    max_steps = 3

    def at_rollout_start(self):
        super().at_rollout_start()
        logger.warning("rollout start")

    def at_rollout_end(self):
        super().at_rollout_end()
        logger.warning("rollout end")

    def at_episode_start(self):
        super().at_episode_start()
        logger.warning("episode start")

    def at_episode_end(self):
        super().at_episode_end()
        logger.warning("episode end")

    def at_episode_step(
        self,
        i_episode: int,
        step: int,
        domain: Domain,
        solver: Union[Solver, Policies],
        action,
        outcome,
    ) -> bool:
        logger.warning("episode step")
        return step >= self.max_steps


def test_rollout_return_episodes():
    domain = Maze()
    num_episodes = 3
    max_steps = 10
    episodes = rollout(
        domain=domain,
        render=False,
        verbose=False,
        action_formatter=None,
        outcome_formatter=None,
        return_episodes=True,
        num_episodes=num_episodes,
        max_steps=max_steps,
    )
    assert len(episodes) == num_episodes
    for observations, actions, values in episodes:
        assert (
            len(observations) == max_steps + 1
        )  # one observation per step + starting point


def test_rollout_cb(caplog):
    domain = Maze()
    num_episodes = 3
    max_steps = 10
    episodes = rollout(
        domain=domain,
        render=False,
        verbose=False,
        action_formatter=None,
        outcome_formatter=None,
        return_episodes=True,
        num_episodes=num_episodes,
        max_steps=max_steps,
        rollout_callback=MyCallback(),
    )
    lines = caplog.text.splitlines()
    assert len([line for line in lines if "rollout start" in line]) == 1
    assert len([line for line in lines if "rollout end" in line]) == 1
    assert len([line for line in lines if "episode start" in line]) == num_episodes
    assert len([line for line in lines if "episode end" in line]) == num_episodes
    assert (
        len([line for line in lines if "episode step" in line])
        == num_episodes * MyCallback.max_steps
    )
    assert len(episodes) == num_episodes
    for observations, actions, values in episodes:
        assert (
            len(observations) == MyCallback.max_steps + 1
        )  # one observation per step + starting point
