import gym
import gym_jsbsim
import numpy as np

from typing import Callable

from airlaps import TransitionOutcome, TransitionValue
from airlaps.hub.domain.gym import DeterministicInitializedGymDomain, GymWidthDomain, GymDiscreteActionDomain
from airlaps.hub.solver.iw import IW
from airlaps.hub.solver.riw import RIW
from airlaps.utils import rollout

from gym_jsbsim.catalogs.catalog import Catalog as prp

# ENV_NAME = 'GymJsbsim-HeadingControlTask-v0'
ENV_NAME = 'GymJsbsim-TaxiapControlTask-v0'
HORIZON = 200

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
                       discretization_factor: int = 10,
                       branching_factor: int = None) -> None:
        """Initialize GymRIWDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        set_state: Function to call to set the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        get_state: Function to call to get the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        """
        DeterministicInitializedGymDomain.__init__(self,
                                                   gym_env=gym_env,
                                                   set_state=set_state,
                                                   get_state=get_state)
        GymDiscreteActionDomain.__init__(self,
                                         discretization_factor=discretization_factor,
                                         branching_factor=branching_factor)
        GymWidthDomain.__init__(self)
    
    def _state_step(self, action: D.T_agent[D.T_concurrency[D.T_event]]) -> TransitionOutcome[
            D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]:
        o = super()._state_step(action)
        return TransitionOutcome(state=o.state, value=TransitionValue(reward=-o.value.reward), termination=o.termination, info=o.info)


domain_factory = lambda: GymRIWDomain(gym_env=gym_env,
                                      set_state=lambda e, s: e.set_state(s),
                                      get_state=lambda e: e.get_state(),
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
                                 time_budget=5000,
                                 rollout_budget=1000,
                                 max_depth=HORIZON-1,
                                 max_cost=1,
                                 exploration=0.25,
                                 parallel=False,
                                 debug_logs=False)
    solver = GymRIWDomain.solve_with(solver_factory, domain_factory)
    initial_state = solver._domain.reset()
    rollout(domain, solver, from_memory=initial_state, num_episodes=1, max_steps=HORIZON, max_framerate=30, outcome_formatter=None, action_formatter=None)

    # init_state = domain.get_initial_state()
    # solver.solve_from(init_state)
    # plan = []
    # current_state = init_state

    # while True:
    #     print('current state:', str(current_state))
    #     if solver._is_solution_defined_for(current_state):
    #         plan.append(solver.get_next_action(current_state))
    #         current_state = domain.get_next_state(current_state, plan[-1])
    #         # lon = domain._gym_env.sim.get_property_value(prp.position_long_gc_deg)
    #         # lat = domain._gym_env.sim.get_property_value(prp.position_lat_geod_deg)
    #         # print(str(lon), str(lat))
    #     else:
    #         break
    # print('PLAN:', str(plan))

    # traj_string = []
    # attitude_string = []

    # while not done:
    #     ...
    #     lon = self.env.sim.get_property_value(prp.position_long_gc_deg)
    #     lat = self.env.sim.get_property_value(prp.position_lat_geod_deg)
    #     alt = self.env.sim.get_property_value(prp.position_h_sl_ft)
    #     psi = self.env.sim.get_property_value(prp.attitude_psi_rad)
    #     theta = self.env.sim.get_property_value(prp.attitude_theta_rad)
    #     phi = self.env.sim.get_property_value(prp.attitude_phi_rad)

    #     sim_time = self.env.sim.get_property_value(prp.simulation_sim_time_sec)

    #     traj_string += [sim_time, lon, lat, alt]
    #     attitude_string += [sim_time, psi, theta, phi]

    #     print("\n==========\nTRAJECTOIRE\n==========\n")
    #     for t in range(0, len(traj_string) - 5, +4):
    #         print(str(traj_string[t]) + "," + str(traj_string[t + 1]) + "," + str(traj_string[t + 2]) + "," + str(traj_string[t + 3]) + ",")
    #     print("\n==========\nATTITUDE\n==========\n")
    #     for t in range(0, len(attitude_string) - 5, +4):
    #         print(str(attitude_string[t]) + "," + str(attitude_string[t + 1]) + "," + str(attitude_string[t + 2]) + "," + str(attitude_string[t + 3]) + ",")


    #         f = open('cart3D_taxi_CGP.json', 'w')
    #         f.write('cart3D_taxi_CGP=[')
    #         for t in range(0, len(traj_string) - 5, +4):
    #             f.write(str(traj_string[t]) + "," + str(traj_string[t + 1]) + "," + str(traj_string[t + 2]) + "," + str(traj_string[t + 3]))
    #             if t == len(traj_string) - 8:
    #                 f.write(']')
    #             else:
    #                 f.write(',')
    #         f.close()


    #         f = open('angles_taxi_CGP.json', 'w')
    #         f.write('angles_taxi_CGP=[')
    #         for t in range(0, len(attitude_string) - 5, +4):
    #             f.write(str(attitude_string[t]) + "," + str(attitude_string[t + 1]) + "," + str(
    #                 attitude_string[t + 2]) + "," + str(attitude_string[t + 3]))
    #             if t == len(traj_string) - 8:
    #                 f.write(']')
    #             else:
    #                 f.write(',')
    #         f.close()
