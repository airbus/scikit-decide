import logging
import random
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import pytest

from skdecide import Domain, Solver, Space, Value
from skdecide.builders.domain import (
    Actions,
    DeterministicInitialized,
    DeterministicTransitions,
    FullyObservable,
    Initializable,
    Markovian,
    SingleAgent,
)
from skdecide.builders.solver import DeterministicPolicies, Policies
from skdecide.hub.domain.maze import Maze
from skdecide.hub.solver.lazy_astar import LazyAstar
from skdecide.hub.space.gym import DiscreteSpace, GymSpace
from skdecide.utils import (
    ReplayOutOfActionMethod,
    ReplaySolver,
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


def test_replay_solver_rollout():
    domain = Maze()
    # rollout with random walk
    episodes = rollout(
        domain=domain,
        render=False,
        verbose=False,
        action_formatter=None,
        outcome_formatter=None,
        return_episodes=True,
        num_episodes=1,
        max_steps=10,
    )
    # take the first episode
    observations, actions, values = episodes[0]
    # wrap the corresponding actions in a replay solver
    replay_solver = ReplaySolver(actions)
    # replay the rollout
    replayed_episodes = rollout(
        domain=domain,
        solver=replay_solver,
        return_episodes=True,
        render=False,
        verbose=False,
        action_formatter=None,
        outcome_formatter=None,
        num_episodes=1,
        max_steps=10,
    )
    # same outputs (for deterministic domain)
    assert episodes == replayed_episodes

    # check out-of-action behaviour
    replay_solver = ReplaySolver(
        actions, out_of_action_method=ReplayOutOfActionMethod.LOOP
    )
    replayed_actions = []


@pytest.mark.parametrize("out_of_action_method", list(ReplayOutOfActionMethod))
def test_replay_solver_out_of_action(out_of_action_method):
    domain = Maze()
    # rollout with random walk
    episodes = rollout(
        domain=domain,
        render=False,
        verbose=False,
        action_formatter=None,
        outcome_formatter=None,
        return_episodes=True,
        num_episodes=1,
        max_steps=10,
    )
    # take the first episode
    observations, actions, values = episodes[0]
    # wrap the corresponding actions in a replay solver
    replay_solver = ReplaySolver(actions, out_of_action_method=out_of_action_method)
    # ask for more actions
    n_more_actions = 2
    if out_of_action_method == ReplayOutOfActionMethod.ERROR:
        with pytest.raises(RuntimeError):
            [
                replay_solver.sample_action(None)
                for _ in range(len(actions) + n_more_actions)
            ]
    else:
        replayed_actions = [
            replay_solver.sample_action(None)
            for _ in range(len(actions) + n_more_actions)
        ]
        if out_of_action_method == ReplayOutOfActionMethod.LOOP:
            assert replayed_actions == actions + actions[:n_more_actions]
        elif out_of_action_method == ReplayOutOfActionMethod.LOOP:
            assert replayed_actions == actions + [
                actions[-1] for _ in range(n_more_actions)
            ]


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


class D(
    Domain,
    Actions,
    FullyObservable,
    DeterministicTransitions,
    SingleAgent,
    DeterministicInitialized,
    Markovian,
):
    T_state = int  # Type of states
    T_observation = T_state  # Type of observations
    T_event = int  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of logical checks
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class MyRestrictedActionDomain(D):
    def __init__(self, n=10):
        self.n = n

    def _get_initial_state_(self) -> D.T_state:
        return int(self.n / 2)

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        return memory + int(2 * (action - 0.5))  # +/- 1

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        return 1.0

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        return state <= 0 or state >= self.n - 1

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return DiscreteSpace(2)

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        return GymSpace(gym.spaces.Discrete(1, start=1))

    def _is_applicable_action_from(
        self, action: D.T_agent[D.T_event], memory: D.T_memory[D.T_state]
    ) -> bool:
        return action == 1

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return DiscreteSpace(self.n)


class MyActionsMaskedDomain(MyRestrictedActionDomain):
    def action_masks(
        self,
    ):  # contradictory with _is_applicable_action_from to check use of it
        return np.array([True, False])


class MyMaskedRandomSolver(Solver, DeterministicPolicies):
    """Ramdom solver proposing all actions except is `action_masks` in `sample_action()` kwargs."""

    T_domain = D

    def __init__(self, domain_factory):
        super().__init__(domain_factory=domain_factory)
        self.actions = list(self._domain_factory().get_action_space().get_elements())
        self.default_action_masks = [True for _ in self.actions]

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation], **kwargs: Any
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        action_masks = kwargs.get("action_masks", self.default_action_masks)
        available_actions = [a for a, ok in zip(self.actions, action_masks) if ok]
        return np.random.choice(available_actions)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _solve(self) -> None:
        pass


@pytest.fixture
def random_seed():
    np.random.seed(0)
    random.seed(0)


def test_rollout_use_action_masking(random_seed):
    # no masking => both actions possible
    domain_factory = MyRestrictedActionDomain
    solver = MyMaskedRandomSolver(domain_factory=domain_factory)
    episodes = rollout(
        domain=domain_factory(),
        solver=solver,
        render=False,
        verbose=False,
        action_formatter=None,
        outcome_formatter=None,
        return_episodes=True,
        num_episodes=1,
        max_steps=10,
    )
    observations, actions, values = episodes[0]
    assert 0 in actions
    assert 1 in actions

    # with masking, using is_applicable_action() => only 1's
    domain_factory = MyRestrictedActionDomain
    solver = MyMaskedRandomSolver(domain_factory=domain_factory)
    episodes = rollout(
        domain=domain_factory(),
        solver=solver,
        render=False,
        verbose=False,
        action_formatter=None,
        outcome_formatter=None,
        return_episodes=True,
        num_episodes=1,
        max_steps=10,
        use_action_masking=True,
    )
    observations, actions, values = episodes[0]
    assert 0 not in actions
    assert 1 in actions

    # with masking, using custom action_masks() => only 0's
    domain_factory = MyActionsMaskedDomain
    solver = MyMaskedRandomSolver(domain_factory=domain_factory)
    episodes = rollout(
        domain=domain_factory(),
        solver=solver,
        render=False,
        verbose=False,
        action_formatter=None,
        outcome_formatter=None,
        return_episodes=True,
        num_episodes=1,
        max_steps=10,
        use_action_masking=True,
    )
    observations, actions, values = episodes[0]
    assert 0 in actions
    assert 1 not in actions


class MyPseudoRandomSolver(Solver, DeterministicPolicies):
    """Ramdom solver being forced by arg `action` if existing in `sample_action()` kwargs."""

    T_domain = D

    def __init__(self, domain_factory):
        super().__init__(domain_factory=domain_factory)
        self.actions = list(self._domain_factory().get_action_space().get_elements())

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation], **kwargs: Any
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        if "action" in kwargs:
            return kwargs["action"]
        else:
            return np.random.choice(self.actions)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _solve(self) -> None:
        pass


def test_rollout_kwargs_sample_action_fn(random_seed):
    domain_factory = MyRestrictedActionDomain
    solver = MyPseudoRandomSolver(domain_factory=domain_factory)
    # we force to alternate between state 4 and 5 by going up and down
    kwargs_sample_action_fn = lambda obs: {"action": 1} if obs < 5 else {"action": 0}

    episodes = rollout(
        domain=domain_factory(),
        solver=solver,
        render=False,
        verbose=False,
        action_formatter=None,
        outcome_formatter=None,
        return_episodes=True,
        num_episodes=1,
        max_steps=10,
        kwargs_sample_action_fn=kwargs_sample_action_fn,
    )
    observations, actions, values = episodes[0]
    assert set(observations) == {4, 5}
