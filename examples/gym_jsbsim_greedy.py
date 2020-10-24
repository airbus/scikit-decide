# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import gym_jsbsim
import numpy as np
import folium

from typing import Callable, Any

from skdecide import TransitionOutcome, TransitionValue, Solver, Space, Domain
from skdecide.builders.domain import SingleAgent, Rewards, Sequential, UnrestrictedActions, Initializable, Memoryless, FullyObservable, Renderable
from skdecide.hub.domain.gym import DeterministicGymDomain, GymDiscreteActionDomain
from skdecide.builders.solver import DeterministicPolicies, Utilities
from skdecide.hub.space.gym import GymSpace
from skdecide.utils import rollout

from gym_jsbsim.catalogs.catalog import Catalog as prp
from gym_jsbsim.envs.taxi_utils import *

# ENV_NAME = 'GymJsbsim-HeadingControlTask-v0'
ENV_NAME = 'GymJsbsim-TaxiapControlTask-v0'
HORIZON = 1000


class D(DeterministicGymDomain, GymDiscreteActionDomain):
    pass


class GymGreedyDomain(D):
    """This class wraps a cost-based deterministic OpenAI Gym environment as a domain
        usable by a width-based planner

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       set_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
                       get_state: Callable[[gym.Env], D.T_memory[D.T_state]] = None,
                       discretization_factor: int = 10,
                       branching_factor: int = None,
                       horizon: int = 1000) -> None:
        """Initialize GymRIWDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        set_state: Function to call to set the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        get_state: Function to call to get the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        discretization_factor: Number of discretized action variable values per continuous action variable
        branching_factor: if not None, sample branching_factor actions from the resulting list of discretized actions
        horizon: maximum number of steps allowed for the gym environment
        """
        DeterministicGymDomain.__init__(self,
                                        gym_env=gym_env,
                                        set_state=set_state,
                                        get_state=get_state)
        GymDiscreteActionDomain.__init__(self,
                                         discretization_factor=discretization_factor,
                                         branching_factor=branching_factor)
        gym_env._max_episode_steps = horizon
        self._map = None
        self._path = None
    
    def _render_from(self, memory: D.T_memory[D.T_state], **kwargs: Any) -> Any:
        # Get rid of the current state and just look at the gym env's current internal state
        lon = self._gym_env.sim.get_property_value(prp.position_long_gc_deg)
        lat = self._gym_env.sim.get_property_value(prp.position_lat_geod_deg)
        if (self._map is None):
            self._map = folium.Map(location=[lat, lon], zoom_start=18)
            taxiPath = taxi_path()
            for p in taxiPath.centerlinepoints:
                folium.Marker((p[1], p[0]), popup=p).add_to(self._map)
            folium.PolyLine(taxiPath.centerlinepoints, color='blue', weight=2.5, opacity=1).add_to(self._map)
            self._path = folium.PolyLine([(lat, lon)], color='red', weight=2.5, opacity=1)
            self._path.add_to(self._map)
            f = open('gym_jsbsim_map.html', 'w')
            f.write('<!DOCTYPE html>\n' +
                    '<HTML>\n' +
                    '<HEAD>\n' +
                    '<META http-equiv="refresh" content="5">\n' +
                    '</HEAD>\n' +
                    '<FRAMESET>\n' +
                    '<FRAME src="gym_jsbsim_map_update.html">\n' +
                    '</FRAMESET>\n' +
                    '</HTML>')
            f.close()
        else:
            self._path.locations.append(folium.utilities.validate_location((lat, lon)))
            self._map.location = folium.utilities.validate_location((lat, lon))
        self._map.save('gym_jsbsim_map_update.html')


class GreedyPlanner(Solver, DeterministicPolicies, Utilities):
    T_domain = D

    def __init__(self):
        self._domain = None
        self._best_action = None
        self._best_reward = None
        self._current_pos = None

    def _init_solve(self, domain_factory: Callable[[], D]) -> None:
        self._domain = domain_factory()
        self._domain.reset()
        lon = self._domain._gym_env.sim.get_property_value(prp.position_long_gc_deg)
        lat = self._domain._gym_env.sim.get_property_value(prp.position_lat_geod_deg)
        self._current_pos = (lat, lon)

    def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
        self._init_solve(domain_factory)

    def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
        self._best_action = None
        self._best_reward = -float('inf')
        self._current_pos = None
        for a in self._domain.get_applicable_actions(memory).get_elements():
            o = self._domain.get_next_state(memory, a)
            v = self._domain.get_transition_value(memory, a, o)
            if v.reward > self._best_reward:
                self._best_reward = v.reward
                self._best_action = a
                lon = self._domain._gym_env.sim.get_property_value(prp.position_long_gc_deg)
                lat = self._domain._gym_env.sim.get_property_value(prp.position_lat_geod_deg)
                self._current_pos = (lat, lon)
    
    def _is_solution_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return False  # for to recompute the best action at each step greedily
    
    def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        if not self._is_solution_defined_for(observation):
            self._solve_from(observation)
        return self._best_action
    
    def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
        return self._best_cost
    
    def get_current_position(self):
        return self._current_pos


domain_factory = lambda: GymGreedyDomain(gym_env=gym.make(ENV_NAME),
                                         set_state=lambda e, s: e.set_state(s),
                                         get_state=lambda e: e.get_state(),
                                         discretization_factor=5,
                                         horizon=HORIZON)
domain = domain_factory()
domain.reset()

if GreedyPlanner.check_domain(domain):
    with GreedyPlanner() as solver:
        GymGreedyDomain.solve_with(solver, domain_factory)
        initial_state = solver._domain.reset()
        rollout(domain, solver, from_memory=initial_state, num_episodes=1, max_steps=HORIZON, max_framerate=30,
                outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
