# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import gym_jsbsim
import numpy as np
import folium
import os
import json

from typing import Callable, Any

from airlaps import TransitionOutcome, TransitionValue
from airlaps.hub.domain.gym import DeterministicInitializedGymDomain, GymWidthDomain, GymDiscreteActionDomain
from airlaps.hub.solver.iw import IW
from airlaps.hub.solver.riw import RIW
from airlaps.utils import rollout

from gym_jsbsim.catalogs.catalog import Catalog as prp

# ENV_NAME = 'GymJsbsim-HeadingControlTask-v0'
ENV_NAME = 'GymJsbsim-TaxiapControlTask-v0'
HORIZON = 200

PATH_FILE = os.path.join(os.path.join(os.path.dirname(gym_jsbsim.__file__), 'docs'), 'points.json')

gym_env = gym.make(ENV_NAME)
gym_env._max_episode_steps = HORIZON


class D(DeterministicInitializedGymDomain, GymWidthDomain, GymDiscreteActionDomain):
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
    
    def _render_from(self, memory: D.T_memory[D.T_state], **kwargs: Any) -> Any:
        # Get rid of the current state and just look at the gym env's current intneral state
        lon = self._gym_env.sim.get_property_value(prp.position_long_gc_deg)
        lat = self._gym_env.sim.get_property_value(prp.position_lat_geod_deg)
        alt = self._gym_env.sim.get_property_value(prp.position_h_sl_ft)
        psi = self._gym_env.sim.get_property_value(prp.attitude_psi_rad)
        theta = self._gym_env.sim.get_property_value(prp.attitude_theta_rad)
        phi = self._gym_env.sim.get_property_value(prp.attitude_phi_rad)
        if (self._map is None):
            self._map = folium.Map(location=[lat, lon], zoom_start=18)
            self._map.get_root().header.add_child(folium.Element('<script type="text/javascript" src="http://livejs.com/live.js"></script>'))
            self._current_point = (lat, lon)
            with open(PATH_FILE) as json_file:
                data = json.load(json_file)
                points = []
                for p, c in data.items():
                    folium.Marker((c[1], c[0]), popup=p).add_to(self._map)
                    points.append((c[1], c[0]))
                folium.PolyLine(points, color='blue', weight=2.5, opacity=1).add_to(self._map)
        else:
            folium.PolyLine([self._current_point, (lat, lon)], color='red', weight=2.5, opacity=1).add_to(self._map)
            self._current_point = (lat, lon)
        self._map.save('gym_jsbsim_map.html')


domain_factory = lambda: GymRIWDomain(gym_env=gym_env,
                                      set_state=lambda e, s: e.set_state(s),
                                      get_state=lambda e: e.get_state(),
                                      continuous_feature_fidelity=3,
                                      discretization_factor=5)
domain = domain_factory()

def state_features(s, d):
    f = d.state_features(s)
    # f.append(s._context[5])
    # print('features:', str(f))
    return f

# TODO: understand why gscore ordering with negative costs (positive rewards) is much more efficient (than with negative rewards and gscore ordering?!?)

if RIW.check_domain(domain):
    solver_factory = lambda: RIW(state_features=lambda s, d: state_features(s, d),
                                 use_state_feature_hash=False,
                                 use_simulation_domain=False,
                                 time_budget=2000,
                                 rollout_budget=1000,
                                 max_depth=HORIZON-1,
                                 max_cost=1,
                                 exploration=0.25,
                                 parallel=False,
                                 debug_logs=False)
    solver = GymRIWDomain.solve_with(solver_factory, domain_factory)
    initial_state = solver._domain.reset()
    rollout(domain, solver, from_memory=initial_state, num_episodes=1, max_steps=HORIZON, max_framerate=30,
            outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
