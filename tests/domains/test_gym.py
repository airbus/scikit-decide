from collections import OrderedDict
from collections.abc import Callable

import gymnasium as gym
import numpy as np

from skdecide.hub.domain.gym import (
    GymDiscreteActionDomain,
    GymDomain,
    GymPlanningDomain,
    GymWidthDomain,
)
from skdecide.hub.domain.gym.gym import AsGymnasiumEnv
from skdecide.hub.domain.maze.maze import Maze
from skdecide.hub.solver.cgp import CGP
from skdecide.hub.space.gym.gym import ListSpace


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


class D(GymDomain, GymDiscreteActionDomain):
    pass


class GymDomainForDiscreteSolvers(D):
    def __init__(
        self,
        gym_env: gym.Env,
        discretization_factor: int = 3,
        branching_factor: int = None,
    ) -> None:
        GymDomain.__init__(
            self,
            gym_env=gym_env,
        )
        GymDiscreteActionDomain.__init__(
            self,
            discretization_factor=discretization_factor,
            branching_factor=branching_factor,
        )


def test_gymdomain():
    ENV_NAME = "MountainCarContinuous-v0"
    domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
    solver = CGP(
        domain_factory=domain_factory, folder_name="TEMP_CGP", n_it=2, verbose=False
    )
    solver.solve()
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
    domain.step(action=domain.get_applicable_actions().sample())


def test_asgymnasiumenv():
    domain = Maze()
    domain.reset()
    env = AsGymnasiumEnv(domain=domain, render_mode="human")
    env.reset()
    env.step(env.action_space.sample())
    env.render()
    env.close()


def test_discretisation():
    ENV_NAME = "MountainCarContinuous-v0"
    gym_env = gym.make(ENV_NAME)
    gym_action_space = gym_env.action_space
    discretization_factor = 3

    # Box 1d
    assert isinstance(gym_action_space, gym.spaces.Box)
    an_original_action = gym_action_space.sample()
    domain = GymDomainForDiscreteSolvers(
        gym_env=gym_env, discretization_factor=discretization_factor
    )
    domain.reset()
    skdecide_applicable_actions_space = domain.get_applicable_actions()
    assert isinstance(skdecide_applicable_actions_space, ListSpace)
    actions = skdecide_applicable_actions_space.get_elements()
    assert len(actions) == discretization_factor
    action = actions[0]
    assert action in gym_action_space
    assert type(action) == type(an_original_action)
    assert isinstance(action, np.ndarray)
    assert action.dtype == gym_action_space.dtype
    assert action.shape == gym_action_space.shape

    # Box with a multidimensional shape
    shape = (2, 3)
    negative_row = 1
    negative_col = 0
    dtype = np.float16
    low = np.zeros(shape, dtype=dtype)
    low[negative_row, negative_col] = -1
    high = np.ones(shape, dtype=dtype)
    high[negative_row, negative_col] = -0.1
    gym_action_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    gym_env.action_space = gym_action_space
    an_original_action = gym_action_space.sample()
    domain = GymDomainForDiscreteSolvers(
        gym_env=gym_env, discretization_factor=discretization_factor
    )
    domain.reset()
    skdecide_applicable_actions_space = domain.get_applicable_actions()
    assert isinstance(skdecide_applicable_actions_space, ListSpace)
    actions = skdecide_applicable_actions_space.get_elements()
    assert len(actions) == discretization_factor ** int(np.prod(shape))
    action = actions[0]
    assert action in gym_action_space
    assert type(action) == type(an_original_action)
    assert isinstance(action, np.ndarray)
    assert action.dtype == gym_action_space.dtype
    assert action.shape == gym_action_space.shape

    # Discrete
    n = 4
    start = 2
    gym_action_space = gym.spaces.Discrete(n=n, start=start)
    gym_env.action_space = gym_action_space
    an_original_action = gym_action_space.sample()
    domain = GymDomainForDiscreteSolvers(
        gym_env=gym_env, discretization_factor=discretization_factor
    )
    domain.reset()
    skdecide_applicable_actions_space = domain.get_applicable_actions()
    assert isinstance(skdecide_applicable_actions_space, ListSpace)
    actions = skdecide_applicable_actions_space.get_elements()
    assert len(actions) == n
    for action in actions:
        assert action in gym_action_space
        assert type(action) == type(an_original_action)
        assert action < n + start
        assert action >= start

    # MultiDiscrete (flattened shape)
    nvec = [2, 1, 3]
    gym_action_space = gym.spaces.MultiDiscrete(nvec)
    gym_env.action_space = gym_action_space
    an_original_action = gym_action_space.sample()
    domain = GymDomainForDiscreteSolvers(
        gym_env=gym_env, discretization_factor=discretization_factor
    )
    domain.reset()
    skdecide_applicable_actions_space = domain.get_applicable_actions()
    assert isinstance(skdecide_applicable_actions_space, ListSpace)
    actions = skdecide_applicable_actions_space.get_elements()
    assert len(actions) == int(np.prod(nvec))
    for action in actions:
        assert action in gym_action_space
        assert type(action) == type(an_original_action)
        assert action.dtype == an_original_action.dtype
        assert action.shape == an_original_action.shape
        for i in range(len(nvec)):
            assert action[i] < nvec[i]
            assert action[i] >= 0

    # MultiDiscrete (2d shape + specified dtype)
    nvec = np.array([[2, 1, 3], [1, 1, 2]])
    dtype = np.int32
    gym_action_space = gym.spaces.MultiDiscrete(nvec, dtype)
    gym_env.action_space = gym_action_space
    an_original_action = gym_action_space.sample()
    domain = GymDomainForDiscreteSolvers(
        gym_env=gym_env, discretization_factor=discretization_factor
    )
    domain.reset()
    skdecide_applicable_actions_space = domain.get_applicable_actions()
    assert isinstance(skdecide_applicable_actions_space, ListSpace)
    actions = skdecide_applicable_actions_space.get_elements()
    assert len(actions) == int(np.prod(nvec))
    for action in actions:
        assert action in gym_action_space
        assert type(action) == type(an_original_action)
        assert action.dtype == an_original_action.dtype
        assert action.shape == an_original_action.shape
        for i in range(nvec.shape[0]):
            for j in range(nvec.shape[1]):
                assert action[i, j] < nvec[i, j]
                assert action[i, j] >= 0

    # MultiBinary (1d shape)
    n = 4
    gym_action_space = gym.spaces.MultiBinary(n)
    gym_env.action_space = gym_action_space
    an_original_action = gym_action_space.sample()
    domain = GymDomainForDiscreteSolvers(
        gym_env=gym_env, discretization_factor=discretization_factor
    )
    domain.reset()
    skdecide_applicable_actions_space = domain.get_applicable_actions()
    assert isinstance(skdecide_applicable_actions_space, ListSpace)
    actions = skdecide_applicable_actions_space.get_elements()
    assert len(actions) == 2**n
    action = actions[0]
    assert action in gym_action_space
    assert type(action) == type(an_original_action)
    assert action.dtype == an_original_action.dtype
    assert action.shape == an_original_action.shape
    for bit in action:
        bit == 0 or bit == 1

    # MultiBinary (2d shape)
    n = [2, 3]
    gym_action_space = gym.spaces.MultiBinary(n)
    gym_env.action_space = gym_action_space
    an_original_action = gym_action_space.sample()
    domain = GymDomainForDiscreteSolvers(
        gym_env=gym_env, discretization_factor=discretization_factor
    )
    domain.reset()
    skdecide_applicable_actions_space = domain.get_applicable_actions()
    assert isinstance(skdecide_applicable_actions_space, ListSpace)
    actions = skdecide_applicable_actions_space.get_elements()
    assert len(actions) == 2 ** int(np.prod(n))
    action = actions[0]
    assert action in gym_action_space
    assert type(action) == type(an_original_action)
    assert action.dtype == an_original_action.dtype
    assert action.shape == an_original_action.shape
    for bit in action.ravel():
        bit == 0 or bit == 1

    # Tuple of spaces
    n = [2, 3]
    shape = (2, 1)
    negative_row = 1
    negative_col = 0
    dtype = np.float16
    low = np.zeros(shape, dtype=dtype)
    low[negative_row, negative_col] = -1
    high = np.ones(shape, dtype=dtype)
    high[negative_row, negative_col] = -0.1
    gym_action_space = gym.spaces.Tuple(
        (
            gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype),
            gym.spaces.MultiBinary(n),
        )
    )
    gym_env.action_space = gym_action_space
    an_original_action = gym_action_space.sample()
    domain = GymDomainForDiscreteSolvers(
        gym_env=gym_env, discretization_factor=discretization_factor
    )
    domain.reset()
    skdecide_applicable_actions_space = domain.get_applicable_actions()
    assert isinstance(skdecide_applicable_actions_space, ListSpace)
    actions = skdecide_applicable_actions_space.get_elements()
    assert len(actions) == 2 ** int(np.prod(n)) * discretization_factor ** int(
        np.prod(shape)
    )
    action = actions[0]
    assert action in gym_action_space
    assert type(action) == type(an_original_action)
    assert action[0].dtype == an_original_action[0].dtype
    assert action[0].shape == an_original_action[0].shape
    assert action[0][negative_row, negative_col] < 0
    assert action[1].dtype == an_original_action[1].dtype
    assert action[1].shape == an_original_action[1].shape
    for bit in action[1].ravel():
        bit == 0 or bit == 1

    # Dict of spaces
    n = [2, 3]
    shape = (2, 1)
    negative_row = 1
    negative_col = 0
    dtype = np.float16
    low = np.zeros(shape, dtype=dtype)
    low[negative_row, negative_col] = -1
    high = np.ones(shape, dtype=dtype)
    high[negative_row, negative_col] = -0.1
    gym_action_space = gym.spaces.Dict(
        OrderedDict(
            position=gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype),
            status=gym.spaces.MultiBinary(n),
        )
    )
    assert isinstance(gym_action_space.spaces, OrderedDict)
    keys = list(gym_action_space.spaces.keys())
    gym_env.action_space = gym_action_space
    an_original_action = gym_action_space.sample()
    domain = GymDomainForDiscreteSolvers(
        gym_env=gym_env, discretization_factor=discretization_factor
    )
    domain.reset()
    skdecide_applicable_actions_space = domain.get_applicable_actions()
    assert isinstance(skdecide_applicable_actions_space, ListSpace)
    actions = skdecide_applicable_actions_space.get_elements()
    assert len(actions) == 2 ** int(np.prod(n)) * discretization_factor ** int(
        np.prod(shape)
    )
    action = actions[0]
    assert action in gym_action_space
    assert type(action) == type(an_original_action)
    assert list(action.keys()) == list(an_original_action.keys())
    assert action["position"].dtype == an_original_action["position"].dtype
    assert action["position"].shape == an_original_action["position"].shape
    assert action["position"][negative_row, negative_col] < 0
    assert action["status"].dtype == an_original_action["status"].dtype
    assert action["status"].shape == an_original_action["status"].shape
    for bit in action["status"].ravel():
        bit == 0 or bit == 1
