# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import numpy as np

from typing import Callable, Any
from math import sqrt, exp, fabs
from gym.envs.classic_control import rendering

from skdecide import TransitionOutcome, TransitionValue
from skdecide.builders.domain import Renderable
from skdecide.hub.domain.gym import DeterministicInitializedGymDomain, GymWidthDomain, GymDiscreteActionDomain
from skdecide.hub.solver.iw import IW
from skdecide.hub.solver.riw import RIW
from skdecide.utils import rollout

HORIZON = 500


class FakeGymEnv:
    """This class mimics an OpenAI Gym environment
    """

    def __init__(self):
        """Initialize GymDomain.

        # Parameters
        gym_env: The Gym environment (gym.env) to wrap.
        """
        inf = np.finfo(np.float32).max
        self.action_space = gym.spaces.Box(np.array([-1.0]), np.array([1.0]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(np.array([-inf, -inf, -inf, -inf]),
                                                np.array([inf, inf, inf, inf]),
                                                dtype=np.float32)
        self._delta_t = 0.001
        self._init_pos_x = 0.0
        self._init_pos_y = 0.0
        self._init_speed_x = 10.0
        self._init_speed_y = 0
        self._pos_x = None
        self._pos_y =  None
        self._speed_x = None
        self._speed_y = None
        self.viewer = None
        self._path = []
    
    def get_state(self):
        return np.array([self._pos_x, self._pos_y, self._speed_x, self._speed_y], dtype=np.float32)
    
    def set_state(self, state):
        self._pos_x = state[0]
        self._pos_y = state[1]
        self._speed_x = state[2]
        self._speed_y = state[3]

    def reset(self):
        self._pos_x = self._init_pos_x
        self._pos_y = self._init_pos_y
        self._speed_x = self._init_speed_x
        self._speed_y = self._init_speed_y
        self._path = []
        return np.array([self._pos_x, self._pos_y, self._speed_x, self._speed_y], dtype=np.float32)

    def step(self, action):
        speed = sqrt(self._speed_x * self._speed_x  +  self._speed_y * self._speed_y)
        self._speed_y = self._speed_y + action * self._delta_t
        self._pos_y = self._pos_y + self._delta_t * self._speed_y
        self._speed_x = sqrt(speed * speed - self._speed_y * self._speed_y)
        self._pos_x = self._pos_x + self._delta_t * self._speed_x
        obs = np.array([self._pos_x, self._pos_y, self._speed_x, self._speed_y], dtype=np.float32)
        reward = exp(-sqrt(self._pos_y * self._pos_y))
        done = bool(fabs(self._pos_y > 1.0))
        self._path.append((self._pos_x, self._pos_y))
        return obs, reward, done, {}

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.track = rendering.Line((0, screen_height/2), (screen_width, screen_height/2))
            self.track.set_color(0, 0, 1)
            self.viewer.add_geom(self.track)
            self.traj = rendering.PolyLine([], False)
            self.traj.set_color(1, 0, 0)
            self.traj.set_linewidth(3)
            self.viewer.add_geom(self.traj)
        
        if len(self.traj.v) != len(self._path):
            self.traj.v = []
            for p in self._path:
                self.traj.v.append((p[0]*100, screen_height/2 + p[1]*100))
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class D(DeterministicInitializedGymDomain, GymWidthDomain, GymDiscreteActionDomain, Renderable):
    pass


class GymRIWDomain(D):
    """This class wraps a cost-based deterministic OpenAI Gym environment as a domain
        usable by a width-based planner

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       set_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
                       get_state: Callable[[gym.Env], D.T_memory[D.T_state]] = None,
                       continuous_feature_fidelity: int = 1,
                       discretization_factor: int = 10,
                       branching_factor: int = None,
                       max_depth: int = None) -> None:
        """Initialize GymRIWDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        set_state: Function to call to set the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        get_state: Function to call to get the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        continuous_feature_fidelity: Number of integers to represent a continuous feature
                                     in the interval-based feature abstraction (higher is more precise)
        discretization_factor: Number of discretized action variable values per continuous action variable
        branching_factor: if not None, sample branching_factor actions from the resulting list of discretized actions
        max_depth: maximum depth of states to explore from the initial state
        """
        DeterministicInitializedGymDomain.__init__(self,
                                                   gym_env=gym_env,
                                                   set_state=set_state,
                                                   get_state=get_state)
        GymDiscreteActionDomain.__init__(self,
                                         discretization_factor=discretization_factor,
                                         branching_factor=branching_factor)
        GymWidthDomain.__init__(self, continuous_feature_fidelity=continuous_feature_fidelity)
        self._map = None
        self._current_point = None
    
    def _state_step(self, action: D.T_agent[D.T_concurrency[D.T_event]]) -> TransitionOutcome[
            D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]:
        o = super()._state_step(action)
        return TransitionOutcome(state=o.state, value=TransitionValue(reward=o.value.reward - 1), termination=o.termination, info=o.info)


domain_factory = lambda: GymRIWDomain(gym_env=FakeGymEnv(),
                                      set_state=lambda e, s: e.set_state(s),
                                      get_state=lambda e: e.get_state(),
                                      continuous_feature_fidelity=3,
                                      discretization_factor=5)
domain = domain_factory()

if RIW.check_domain(domain):
    solver_factory = lambda: RIW(domain_factory=domain_factory,
                                 state_features=lambda d, s: d.bee1_features(s),
                                 use_state_feature_hash=False,
                                 use_simulation_domain=False,
                                 time_budget=200,
                                 rollout_budget=1000,
                                 max_depth=10,
                                 exploration=0.25,
                                 parallel=False,
                                 debug_logs=False)
    with solver_factory() as solver:
        GymRIWDomain.solve_with(solver, domain_factory)
        initial_state = solver._domain.reset()
        rollout(domain, solver, from_memory=initial_state, num_episodes=1, max_steps=HORIZON, max_framerate=30,
                outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
