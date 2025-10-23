import logging
import random
from typing import Optional, Union

import numpy as np
import pytest

from skdecide import Domain, Solver, Space, Value
from skdecide.builders.domain import (
    Actions,
    DeterministicInitialized,
    DeterministicTransitions,
    FullyObservable,
    Markovian,
    Sequential,
    SingleAgent,
)
from skdecide.builders.solver import DeterministicPolicies, Maskable, Policies
from skdecide.hub.domain.maze import Maze
from skdecide.hub.solver.lazy_astar import LazyAstar
from skdecide.hub.space.gym import DiscreteSpace, ListSpace
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


class D(Domain, Sequential): ...


class MyMaskedRandomSolver(Solver, DeterministicPolicies, Maskable):
    """Ramdom solver proposing all actions except is `action_masks` in `sample_action()` kwargs."""

    T_domain = D

    def __init__(self, domain_factory):
        super().__init__(domain_factory=domain_factory)
        self.actions = {
            agent: list(agent_action_space.get_elements())
            for agent, agent_action_space in self._domain_factory()
            .get_action_space()
            .items()
        }
        self.default_action_masks = {
            agent: [True for _ in agent_actions]
            for agent, agent_actions in self.actions.items()
        }

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        action_masks = self.get_action_mask()
        if action_masks is None:
            action_masks = self.default_action_masks
        return {
            agent: np.random.choice(
                [a for a, ok in zip(self.actions[agent], agent_action_masks) if ok]
            )
            for agent, agent_action_masks in action_masks.items()
        }

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _solve(self) -> None:
        pass


class D(
    Domain,
    Actions,
    FullyObservable,
    DeterministicTransitions,
    DeterministicInitialized,
    Markovian,
    Sequential,
    SingleAgent,
):
    T_state = int  # Type of states
    T_observation = T_state  # Type of observations
    T_event = int  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of logical checks
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class MyRestrictedActionSingleAgentDomain(D):
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
        return ListSpace([1])

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return DiscreteSpace(self.n)


class D(
    Domain,
    Actions,
    FullyObservable,
    DeterministicTransitions,
    DeterministicInitialized,
    Markovian,
    Sequential,
):
    T_state = int  # Type of states
    T_observation = T_state  # Type of observations
    T_event = int  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of logical checks
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class MyRestrictedActionMultiAgentDomain(D):
    def __init__(self, n=10):
        self.n = n

    def _get_initial_state_(self) -> D.T_state:
        return int(self.n / 2)

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        return memory + int(2 * (action["superman"] - 0.5))  # +/- 1

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        return 1.0

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        return {"superman": state <= 0 or state >= self.n - 1}

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return {"superman": DiscreteSpace(2)}

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        return {"superman": ListSpace([1])}

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return {"superman": DiscreteSpace(self.n)}


@pytest.fixture
def random_seed():
    np.random.seed(0)
    random.seed(0)


@pytest.fixture(
    params=[MyRestrictedActionSingleAgentDomain, MyRestrictedActionMultiAgentDomain]
)
def domain_factory(request):
    return request.param


def test_rollout_with_action_masking(random_seed, domain_factory):
    domain = domain_factory()

    # no masking => both actions possible
    with MyMaskedRandomSolver(domain_factory=domain_factory) as solver:
        solver.solve()  # needed to autocast methods like step()
        episodes = rollout(
            domain=domain,
            solver=solver,
            render=False,
            verbose=False,
            action_formatter=None,
            outcome_formatter=None,
            return_episodes=True,
            num_episodes=1,
            max_steps=10,
            use_applicable_actions=False,
        )
    observations, actions, values = episodes[0]
    assert 0 in actions or {"superman": 0} in actions
    assert 1 in actions or {"superman": 1} in actions

    # with masking => only 1's
    with MyMaskedRandomSolver(domain_factory=domain_factory) as solver:
        solver.solve()  # needed to autocast methods like step()
        episodes = rollout(
            domain=domain,
            solver=solver,
            render=False,
            verbose=False,
            action_formatter=None,
            outcome_formatter=None,
            return_episodes=True,
            num_episodes=1,
            max_steps=10,
            use_applicable_actions=True,
        )
    observations, actions, values = episodes[0]
    assert not (0 in actions or {"superman": 0} in actions)
    assert 1 in actions or {"superman": 1} in actions

    # should be using masking by default => only 1's
    with MyMaskedRandomSolver(domain_factory=domain_factory) as solver:
        solver.solve()  # needed to autocast methods like step()
        episodes = rollout(
            domain=domain,
            solver=solver,
            render=False,
            verbose=False,
            action_formatter=None,
            outcome_formatter=None,
            return_episodes=True,
            num_episodes=1,
            max_steps=10,
            use_applicable_actions=True,
        )
    observations, actions, values = episodes[0]
    assert not (0 in actions or {"superman": 0} in actions)
    assert 1 in actions or {"superman": 1} in actions
