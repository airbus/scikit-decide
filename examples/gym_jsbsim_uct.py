# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import gym_jsbsim
import numpy as np
import folium
import json
import sys

from math import sqrt
from typing import Any

from skdecide import EnvironmentOutcome
from skdecide.core import DiscreteDistribution
from skdecide.builders.domain import  DeterministicInitialized
from skdecide.hub.domain.gym import GymDomain, GymDomainHashable, \
    GymDiscreteActionDomain, GymDomainStateProxy
from skdecide.hub.solver.mcts import UCT
from skdecide.utils import rollout

from gym_jsbsim.catalogs.catalog import Catalog as prp
from gym_jsbsim.envs.taxi_utils import *

# ENV_NAME = 'GymJsbsim-HeadingControlTask-v0'
ENV_NAME = 'GymJsbsim-TaxiapControlTask-v0'
HORIZON = 5000


def normalize_and_round(state):
    ns = np.array([s[0] for s in state])
    # ns = ns / np.linalg.norm(ns) if np.linalg.norm(ns) != 0 else ns
    scale = np.array([10.0, 0.1, 100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0])
    ns = 1/(1+np.exp(-ns/scale))
    np.around(ns, decimals=3, out=ns)
    return ns


class D(GymDomainHashable, GymDiscreteActionDomain, DeterministicInitialized):
    pass

class GymUCTRawDomain(D):
    """This class wraps a cost-based deterministic OpenAI Gym environment as a domain
        usable by a width-based planner

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       discretization_factor: int = 10,
                       branching_factor: int = None,
                       max_depth: int = None) -> None:
        """Initialize GymRIWDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        discretization_factor: Number of discretized action variable values per continuous action variable
        branching_factor: if not None, sample branching_factor actions from the resulting list of discretized actions
        max_depth: maximum depth of states to explore from the initial state
        """
        GymDomainHashable.__init__(self, gym_env=gym_env)
        GymDiscreteActionDomain.__init__(self,
                                         discretization_factor=discretization_factor,
                                         branching_factor=branching_factor)
        gym_env._max_episode_steps = max_depth
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
                    '<META http-equiv="refresh" content="60">\n' +
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


class D(GymDomain, GymDiscreteActionDomain):
    pass

class GymUCTStepDomain(D):
    """This class wraps a cost-based deterministic OpenAI Gym environment as a domain
        usable by a width-based planner

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       discretization_factor: int = 10,
                       branching_factor: int = None,
                       max_depth: int = None) -> None:
        """Initialize GymRIWDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        discretization_factor: Number of discretized action variable values per continuous action variable
        branching_factor: if not None, sample branching_factor actions from the resulting list of discretized actions
        max_depth: maximum depth of states to explore from the initial state
        """
        GymDomain.__init__(self, gym_env=gym_env)
        GymDiscreteActionDomain.__init__(self,
                                         discretization_factor=discretization_factor,
                                         branching_factor=branching_factor)
        gym_env._max_episode_steps = max_depth
        self.current_outcome = None
        self._map = None
        self._path = None

    def reset(self):
        self.current_outcome = EnvironmentOutcome(observation=GymDomainStateProxy(state=super().reset(), context=[]),
                                                  value=None,
                                                  termination=False,
                                                  info=None)
        return self.current_outcome.observation
    
    def get_next_state_distribution(self, state, action):
        if state != self.current_outcome.observation:
            self.reset()
            for a in state._context:
                self.step(a)
        outcome = self.step(action)
        observation = GymDomainStateProxy(state=outcome.observation._state, context=state._context + [action])
        self.current_outcome = EnvironmentOutcome(observation=observation, value=outcome.value, termination=outcome.termination, info=outcome.info)
        return DiscreteDistribution([(observation, 1.0)])
    
    def step(self, action):
        outcome = super().step(action)
        observation = GymDomainStateProxy(state=outcome.observation, context=None)
        return EnvironmentOutcome(observation=observation, value=outcome.value, termination=outcome.termination, info=outcome.info)

    def sample(self, state, action):
        if state != self.current_outcome.observation:
            self.reset()
            for a in state._context:
                self.step(a)
        outcome = self.step(action)
        observation = GymDomainStateProxy(state=outcome.observation._state, context=state._context + [action])
        self.current_outcome = EnvironmentOutcome(observation=observation, value=outcome.value, termination=outcome.termination, info=outcome.info)
        return self.current_outcome
    
    def get_transition_value(self, state, action, next_state):
        return self.current_outcome.value

    def is_terminal(self, state):
        return self.current_outcome.termination
    
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
                    '<META http-equiv="refresh" content="60">\n' +
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


class GymUCTSampleDomain(D):
    """This class wraps a cost-based deterministic OpenAI Gym environment as a domain
        usable by a width-based planner

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       discretization_factor: int = 10,
                       branching_factor: int = None,
                       max_depth: int = None) -> None:
        """Initialize GymRIWDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        discretization_factor: Number of discretized action variable values per continuous action variable
        branching_factor: if not None, sample branching_factor actions from the resulting list of discretized actions
        max_depth: maximum depth of states to explore from the initial state
        """
        GymDomain.__init__(self, gym_env=gym_env)
        GymDiscreteActionDomain.__init__(self,
                                         discretization_factor=discretization_factor,
                                         branching_factor=branching_factor)
        gym_env._max_episode_steps = max_depth
        self.current_outcome = None
        self._map = None
        self._path = None

    def reset(self):
        self.current_outcome = EnvironmentOutcome(observation=GymDomainStateProxy(state=normalize_and_round(super().reset()), context=self._gym_env.get_state()),
                                                  value=None,
                                                  termination=False,
                                                  info=None)
        return self.current_outcome.observation
    
    def get_next_state_distribution(self, state, action):
        if state._context != self.current_outcome.observation._context:
            self._gym_env.set_state(state._context)
        outcome = super().step(action)
        observation = GymDomainStateProxy(state=normalize_and_round(outcome.observation), context=self._gym_env.get_state())
        self.current_outcome = EnvironmentOutcome(observation=observation, value=outcome.value, termination=outcome.termination, info=outcome.info)
        return DiscreteDistribution([(observation, 1.0)])
    
    def step(self, action):
        outcome = super().step(action)
        observation = GymDomainStateProxy(state=normalize_and_round(outcome.observation), context=self._gym_env.get_state())
        return EnvironmentOutcome(observation=observation, value=outcome.value, termination=outcome.termination, info=outcome.info)

    def sample(self, state, action):
        if state._context != self.current_outcome.observation._context:
            self._gym_env.set_state(state._context)
        outcome = super().step(action)
        observation = GymDomainStateProxy(state=normalize_and_round(outcome.observation), context=self._gym_env.get_state())
        self.current_outcome = EnvironmentOutcome(observation=observation, value=outcome.value, termination=outcome.termination, info=outcome.info)
        return self.current_outcome
    
    def get_transition_value(self, state, action, next_state):
        return self.current_outcome.value

    def is_terminal(self, state):
        return self.current_outcome.termination
    
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
                    '<META http-equiv="refresh" content="60">\n' +
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


domain_factory = lambda: GymUCTRawDomain(gym_env=gym.make(ENV_NAME),
                                            discretization_factor=9,
                                            max_depth=HORIZON)

if UCT.check_domain(domain_factory()):
    solver_factory = lambda: UCT(domain_factory=domain_factory,
                                 time_budget = 1000,
                                 rollout_budget = 10,
                                 max_depth = 500,#HORIZON+1,
                                 discount = 1.0,
                                 ucb_constant = 1.0 / sqrt(2.0),
                                 transition_mode=UCT.Options.TransitionMode.Step,
                                 continuous_planning=True,
                                 online_node_garbage=True,
                                 parallel=False,
                                 debug_logs=False)
    with solver_factory() as solver:
        GymUCTRawDomain.solve_with(solver, domain_factory)
        solver._domain.reset()
        rollout(domain_factory(), solver, num_episodes=1, max_steps=HORIZON, max_framerate=30,
                outcome_formatter=lambda o: f'{o.observation} - reward: {o.value.reward:.2f}',
                action_formatter=lambda a: f'{a}', verbose=True)
        with open('gym_jsbsim_uct.json', 'w') as myfile:
            mydict = solver.get_policy()
            json.dump({str(s): (str(v[0]), v[1]) for s, v in mydict.items()}, myfile)
