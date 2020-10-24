# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import gym_jsbsim
import numpy as np
import folium

from typing import Callable, Any, Optional

from skdecide import TransitionOutcome, TransitionValue, EnvironmentOutcome, Domain
from skdecide.builders.domain import SingleAgent, Sequential, Environment, Actions, \
    DeterministicInitialized, Markovian, FullyObservable, Rewards
from skdecide.hub.domain.gym import GymPlanningDomain, GymWidthDomain, \
    GymDiscreteActionDomain, GymDomainStateProxy
from skdecide.hub.solver.iw import IW
from skdecide.utils import rollout

from gym_jsbsim.catalogs.catalog import Catalog as prp
from gym_jsbsim.envs.taxi_utils import *

# ENV_NAME = 'GymJsbsim-HeadingControlTask-v0'
ENV_NAME = 'GymJsbsim-TaxiapControlTask-v0'
HORIZON = 5000


def normalize_and_round(state):
    ns = np.array([s[0] for s in state])
    ns = ns / np.linalg.norm(ns) if np.linalg.norm(ns) != 0 else ns
    # scale = np.array([10.0, 0.1, 100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0])
    # ns = 1/(1+np.exp(-ns/scale))
    np.around(ns, decimals=4, out=ns)
    return ns


class D(GymPlanningDomain, GymWidthDomain, GymDiscreteActionDomain):
    pass


class GymIWDomain(D):
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
        GymPlanningDomain.__init__(self,
                                   gym_env=gym_env,
                                   set_state=set_state,
                                   get_state=get_state,
                                   termination_is_goal=False,
                                   max_depth=max_depth)
        GymDiscreteActionDomain.__init__(self,
                                         discretization_factor=discretization_factor,
                                         branching_factor=branching_factor)
        GymWidthDomain.__init__(self, continuous_feature_fidelity=continuous_feature_fidelity)
        gym_env._max_episode_steps = max_depth
    
    def _get_initial_state_(self) -> D.T_state:
        state_proxy = super()._get_initial_state_()
        return GymDomainStateProxy(state=normalize_and_round(state_proxy._state), context=state_proxy._context)
    
    def _state_step(self, action: D.T_agent[D.T_concurrency[D.T_event]]) -> TransitionOutcome[
            D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]:
        o = super()._state_step(action)
        return TransitionOutcome(state=GymDomainStateProxy(state=normalize_and_round(o.state._state), context=o.state._context),
                                 value=TransitionValue(reward=o.value.reward - 1), termination=o.termination, info=o.info)

    def _sample(self, memory: D.T_memory[D.T_state], action: D.T_agent[D.T_concurrency[D.T_event]]) -> \
        EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]:
        o = super()._sample(memory, action)
        return EnvironmentOutcome(observation=GymDomainStateProxy(state=normalize_and_round(o.observation._state), context=o.observation._context),
                                  value=TransitionValue(reward=o.value.reward - 1), termination=o.termination, info=o.info)
    
    def _get_next_state(self, memory: D.T_memory[D.T_state],
                              action: D.T_agent[D.T_concurrency[D.T_event]]) -> D.T_state:
        o = super()._get_next_state(memory, action)
        return GymDomainStateProxy(state=normalize_and_round(o._state), context=o._context)
    
    def _get_transition_value(self, memory: D.T_memory[D.T_state], action: D.T_agent[D.T_concurrency[D.T_event]],
                              next_state: Optional[D.T_state] = None) -> D.T_agent[TransitionValue[D.T_value]]:
        v = super()._get_transition_value(memory, action, next_state)
        return TransitionValue(reward=v.reward - 1)


class EvaluationDomain(GymPlanningDomain):
    def __init__(self, planning_domain: GymIWDomain) -> None:
        GymPlanningDomain.__init__(self,
                                   gym_env=planning_domain._gym_env,
                                   set_state=planning_domain._set_state,
                                   get_state=planning_domain._get_state,
                                   termination_is_goal=False,
                                   max_depth=HORIZON)
        self._current_state = None
        self._map = None
        self._path = None
    
    def _state_reset(self) -> D.T_state:
        s = super()._state_reset()
        self._current_state = self._gym_env.get_state()
        return s
    
    def _state_step(self, action: D.T_agent[D.T_concurrency[D.T_event]]) -> TransitionOutcome[
            D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]:
        self._gym_env.set_state(self._current_state)
        o = super()._state_step(action)
        self._current_state = self._gym_env.get_state()
        return TransitionOutcome(state=o.state, value=TransitionValue(reward=o.value.reward - 1), termination=o.termination, info=o.info)
    
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
    
    def set_memory(self, memory: D.T_memory[D.T_state]) -> None:
        self._current_state = memory._context[4]
        self._gym_env.set_state(self._current_state)


class D(Domain, SingleAgent, Sequential, Environment, Actions, DeterministicInitialized, Markovian,
            FullyObservable, Rewards):  # TODO: check why DeterministicInitialized & PositiveCosts/Rewards?
        pass


class GymIW(IW):
    def __init__(self,
                 domain_factory: Callable[[], Domain],
                 state_features: Callable[[Domain, D.T_state], Any],
                 use_state_feature_hash: bool = False,
                 node_ordering: Callable[[float, int, int, float, int, int], bool] = None,
                 parallel: bool = True,
                 debug_logs: bool = False) -> None:
        super().__init__(domain_factory=domain_factory,
                         state_features=state_features,
                         use_state_feature_hash=use_state_feature_hash,
                         node_ordering=node_ordering,
                         parallel=parallel,
                         debug_logs=debug_logs)
    
    def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        state = GymDomainStateProxy(state=normalize_and_round(observation._state), context=observation._context)
        action = super()._get_next_action(state)
        if action is None:
            state._context[5] = 0
            self.reset()
            self._domain._reset_features()
            self._domain._current_depth = 0
            return super()._get_next_action(state)
        else:
            return action


domain_factory = lambda: GymIWDomain(gym_env=gym.make(ENV_NAME),
                                      set_state=lambda e, s: e.set_state(s),
                                      get_state=lambda e: e.get_state(),
                                      continuous_feature_fidelity=3,
                                      discretization_factor=5,
                                      max_depth=50)

if IW.check_domain(domain_factory()):
    solver_factory = lambda: GymIW(domain_factory=domain_factory,
                                   state_features=lambda d, s: d.bee1_features(s),
                                   use_state_feature_hash=False,
                                #    node_ordering=lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: True if a_novelty > b_novelty else False if a_novelty < b_novelty else a_gscore < b_gscore,
                                #    node_ordering=lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: True if a_gscore < b_gscore else False if a_gscore > b_gscore else a_novelty > b_novelty,
                                   parallel=False,
                                   debug_logs=False)
    with solver_factory() as solver:
        GymIWDomain.solve_with(solver, domain_factory)
        evaluation_domain = EvaluationDomain(solver._domain)
        evaluation_domain.reset()
        rollout(evaluation_domain, solver, num_episodes=1, max_steps=HORIZON, max_framerate=30,
                outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
