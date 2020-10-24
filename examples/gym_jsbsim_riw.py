# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import gym_jsbsim
import numpy as np
import folium
import bisect
import math
import json

from typing import Callable, Any

from skdecide import TransitionOutcome, TransitionValue, Domain
from skdecide.builders.domain import DeterministicInitialized
from skdecide.hub.domain.gym import GymWidthDomain, \
    GymDiscreteActionDomain, GymDomainStateProxy, GymDomainHashable
from skdecide.hub.solver.riw import RIW
from skdecide.utils import rollout

from gym_jsbsim.catalogs.catalog import Catalog as prp
from gym_jsbsim.envs.taxi_utils import *

# ENV_NAME = 'GymJsbsim-HeadingControlTask-v0'
ENV_NAME = 'GymJsbsim-TaxiapControlTask-v0'
HORIZON = 1000


class D(GymDomainHashable, GymDiscreteActionDomain, GymWidthDomain, DeterministicInitialized):
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
        GymDomainHashable.__init__(self, gym_env=gym_env)
        GymDiscreteActionDomain.__init__(self,
                                         discretization_factor=discretization_factor,
                                         branching_factor=branching_factor)
        GymWidthDomain.__init__(self, continuous_feature_fidelity=continuous_feature_fidelity)
        gym_env._max_episode_steps = max_depth
        self._max_depth = max_depth
        self._current_depth = 0
        self._cumulated_reward = 0
        self._continuous_feature_fidelity = continuous_feature_fidelity
        self._map = None
        self._path = None
        self._must_reset_features = True
    
    def _state_reset(self) -> D.T_state:
        s = super()._state_reset()
        self._current_depth = 0
        self._cumulated_reward = 0
        self._cumulated_dist_to_start = 0
        self._cumulated_dist_to_line = 0
        return GymDomainStateProxy(state=s._state, context=(0, 0, 0, 0))
    
    def _state_step(self, action: D.T_agent[D.T_concurrency[D.T_event]]) -> TransitionOutcome[
            D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]:
        o = super()._state_step(action)
        self._current_depth += 1
        self._cumulated_reward += o.value.reward
        # self._cumulated_dist_to_start += math.exp(-math.fabs(self._gym_env.sim.get_property_value(prp.position_distance_from_start_mag_mt)))
        self._cumulated_dist_to_start = self._gym_env.sim.get_property_value(prp.position_distance_from_start_mag_mt)
        self._cumulated_dist_to_line += math.exp(-math.fabs(self._gym_env.sim.get_property_value(prp.shortest_dist)))
        return TransitionOutcome(state=GymDomainStateProxy(state=o.state._state, context=(self._current_depth,
                                                                                          self._cumulated_reward,
                                                                                          self._cumulated_dist_to_start,
                                                                                          self._cumulated_dist_to_line)),
                                 value=o.value,
                                 termination=o.termination, info=o.info)
    
    def reset_features(self):
        self._must_reset_features = True
    
    def reset_reward_features(self):
        self._feature_increments = []
        # positive increments list for each fidelity level
        self._feature_increments.append([[[0] for f in range(self._continuous_feature_fidelity)] for d in range(self._max_depth)])
        # negative increments list for each fidelity level
        self._feature_increments.append([[[0] for f in range(self._continuous_feature_fidelity)] for d in range(self._max_depth)])
    
    def bee_reward_features(self, state):
        if self._must_reset_features:
            self.reset_reward_features()
            self._must_reset_features = False
        features = []
        ref_val = 0
        depth = state._context[0]
        reward = state._context[1]
        if reward > 0:
            for f in range(self._continuous_feature_fidelity):
                i = bisect.bisect_left(self._feature_increments[0][depth][f], reward - ref_val)
                features.append(i)
                if i >= len(self._feature_increments[0][depth][f]):
                    self._feature_increments[0][depth][f].append(reward - ref_val)
                if i > 0:
                    ref_val = ref_val + self._feature_increments[0][depth][f][i-1]
        else:
            for f in range(self._continuous_feature_fidelity):
                i = bisect.bisect_left(self._feature_increments[1][depth][f], ref_val - reward)
                features.append(-i)
                if i >= len(self._feature_increments[1][depth][f]):
                    self._feature_increments[1][depth][f].append(ref_val - reward)
                if i > 0:
                    ref_val = ref_val - self._feature_increments[1][depth][f][i-1]
        return features
    
    def reset_distance_features(self):
        self._feature_increments = []
        for i in range(2):
            # positive increments list for each fidelity level
            self._feature_increments.append([[[0] for f in range(self._continuous_feature_fidelity)] for d in range(self._max_depth)])
            # negative increments list for each fidelity level
            self._feature_increments.append([[[0] for f in range(self._continuous_feature_fidelity)] for d in range(self._max_depth)])
    
    def bee_distance_features(self, state):
        if self._must_reset_features:
            self.reset_distance_features()
            self._must_reset_features = False
        features = []
        depth = state._context[0]
        index = 0
        for feature in state._context[2:]:
            ref_val = 0
            cf = []
            if feature > 0:
                for f in range(self._continuous_feature_fidelity):
                    i = bisect.bisect_left(self._feature_increments[2*index][depth][f], feature - ref_val)
                    cf.append(i)
                    if i >= len(self._feature_increments[2*index][depth][f]):
                        self._feature_increments[2*index][depth][f].append(feature - ref_val)
                    if i > 0:
                        ref_val = ref_val + self._feature_increments[2*index][depth][f][i-1]
            else:
                for f in range(self._continuous_feature_fidelity):
                    i = bisect.bisect_left(self._feature_increments[2*index+1][depth][f], ref_val - feature)
                    cf.append(-i)
                    if i >= len(self._feature_increments[2*index+1][depth][f]):
                        self._feature_increments[2*index+1][depth][f].append(ref_val - feature)
                    if i > 0:
                        ref_val = ref_val - self._feature_increments[2*index+1][depth][f][i-1]
            features += cf
            index += 1
        return features
    
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


class GymRIW(RIW):
    def __init__(self,
                 domain_factory: Callable[[], Domain],
                 state_features: Callable[[Domain, D.T_state], Any],
                 use_state_feature_hash: bool = False,
                 use_simulation_domain = False,
                 time_budget: int = 3600000,
                 rollout_budget: int = 100000,
                 max_depth: int = 1000,
                 exploration: float = 0.25,
                 online_node_garbage: bool = False,
                 continuous_planning: bool = True,
                 parallel: bool = True,
                 debug_logs: bool = False) -> None:
        super().__init__(domain_factory=domain_factory,
                         state_features=state_features,
                         use_state_feature_hash=use_state_feature_hash,
                         use_simulation_domain=use_simulation_domain,
                         time_budget=time_budget,
                         rollout_budget=rollout_budget,
                         max_depth=max_depth,
                         exploration=exploration,
                         online_node_garbage=online_node_garbage,
                         continuous_planning=continuous_planning,
                         parallel=parallel,
                         debug_logs=debug_logs)
    
    def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        if self._continuous_planning or not self._is_solution_defined_for(observation):
            self._solve_from(observation)
        action = self._solver.get_next_action(observation)
        self._domain.reset_features()
        if action is None:
            print('\x1b[3;33;40m' + 'No best action found in observation ' +
                    str(observation) + ', applying random action' + '\x1b[0m')
            return self._domain.get_action_space().sample()
        else:
            return action


domain_factory = lambda: GymRIWDomain(gym_env=gym.make(ENV_NAME),
                                      set_state=lambda e, s: e.set_state(s),
                                      get_state=lambda e: e.get_state(),
                                      continuous_feature_fidelity=1,
                                      discretization_factor=9,
                                      max_depth=HORIZON)
domain = domain_factory()

if RIW.check_domain(domain):
    solver_factory = lambda: GymRIW(domain_factory=domain_factory,
                                    state_features=lambda d, s: d.bee1_features(s),
                                    use_state_feature_hash=False,
                                    use_simulation_domain=False,
                                    continuous_planning=True,
                                    online_node_garbage=True,
                                    time_budget=10000,
                                    rollout_budget=30,
                                    max_depth=100,
                                    exploration=0.5,
                                    parallel=False,
                                    debug_logs=False)
    with solver_factory() as solver:
        GymRIWDomain.solve_with(solver, domain_factory)
        rollout(domain, solver, num_episodes=1, max_steps=HORIZON, max_framerate=30, verbose=True,
                outcome_formatter=lambda o: f'{o.observation} - reward: {o.value.reward:.2f}',
                action_formatter=lambda a: f'{a}')
        with open('gym_jsbsim_riw.json', 'w') as myfile:
            mydict = solver.get_policy()
            json.dump({str(s): (str(v[0]), v[1]) for s, v in mydict.items()}, myfile)
