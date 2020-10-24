# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import numpy as np
from stable_baselines3 import PPO
from typing import Any, Callable
from math import sqrt

from skdecide.utils import load_registered_domain, load_registered_solver, match_solvers, rollout
from skdecide.hub.domain.gym import GymPlanningDomain, GymWidthDomain, GymDiscreteActionDomain


class D(GymPlanningDomain, GymWidthDomain, GymDiscreteActionDomain):
    pass


class GymDomainForWidthSolvers(D):
    def __init__(self, gym_env: gym.Env,
                 set_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
                 get_state: Callable[[gym.Env], D.T_memory[D.T_state]] = None,
                 termination_is_goal: bool = True,
                 continuous_feature_fidelity: int = 1,
                 discretization_factor: int = 3,
                 branching_factor: int = None,
                 max_depth: int = 50) -> None:
        GymPlanningDomain.__init__(self,
                                   gym_env=gym_env,
                                   set_state=set_state,
                                   get_state=get_state,
                                   termination_is_goal=termination_is_goal,
                                   max_depth=max_depth)
        GymDiscreteActionDomain.__init__(self,
                                         discretization_factor=discretization_factor,
                                         branching_factor=branching_factor)
        GymWidthDomain.__init__(self, continuous_feature_fidelity=continuous_feature_fidelity)
        gym_env._max_episode_steps = max_depth
    
    def state_features(self, s):
        return self.bee1_features(np.append(s._state,
                                            s._context[3].value.reward if s._context[3] is not None else 0))


if __name__ == '__main__':

    try_domains = [

        # Simple Grid World
        {'name': 'Simple Grid World',
         'entry': 'SimpleGridWorld',
         'config': {},
         'rollout': {'max_steps': 1000, 'outcome_formatter': lambda o: f'{o.observation} - cost: {o.value.cost:.2f}'}},

        # Maze
        {'name': 'Maze',
         'entry': 'Maze',
         'config': {},
         'rollout': {'max_steps': 1000, 'max_framerate': 30,
                     'outcome_formatter': lambda o: f'{o.observation} - cost: {o.value.cost:.2f}'}},

        # Mastermind
        {'name': 'Mastermind',
         'entry': 'MasterMind',
         'config': {'n_colours': 3, 'n_positions': 3},
         'rollout': {'max_steps': 1000, 'outcome_formatter': lambda o: f'{o.observation} - cost: {o.value.cost:.2f}'}},

        # Cart Pole (OpenAI Gym)
        {'name': 'Cart Pole (OpenAI Gym)',
         'entry': 'GymDomain',
         'config': {'gym_env': gym.make('CartPole-v1')},
         'rollout': {'num_episodes': 3, 'max_steps': 1000, 'max_framerate': 30, 'outcome_formatter': None}},

        # Mountain Car continuous (OpenAI Gym)
        {'name': 'Mountain Car continuous (OpenAI Gym)',
         'entry': 'GymDomain',
         'config': {'gym_env': gym.make('MountainCarContinuous-v0')},
         'rollout': {'num_episodes': 3, 'max_steps': 1000, 'max_framerate': 30, 'outcome_formatter': None}},

        # ATARI Pacman (OpenAI Gym)
        {'name': 'ATARI Pacman (OpenAI Gym)',
         'entry': 'GymDomain',
         'config': {'gym_env': gym.make('MsPacman-v4')},
         'rollout': {'num_episodes': 3, 'max_steps': 1000, 'max_framerate': 30, 'outcome_formatter': None}}

    ]

    try_solvers = [

        # Simple greedy
        {'name': 'Simple greedy',
         'entry': 'SimpleGreedy',
         'need_domain_factory': False,
         'config': {}},

        # Lazy A* (classical planning)
        {'name': 'Lazy A* (classical planning)',
         'entry': 'LazyAstar',
         'need_domain_factory': False,
         'config': {'heuristic': lambda d, s: d.heuristic(s), 'verbose': False}},

        # A* (planning)
        {'name': 'A* (planning)',
         'entry': 'Astar',
         'need_domain_factory': True,
         'config': {'heuristic': lambda d, s: d.heuristic(s),
                    'parallel': False, 'debug_logs': False}},

        # UCT (reinforcement learning / search)
        {'name': 'UCT (reinforcement learning / search)',
         'entry': 'UCT',
         'need_domain_factory': True,
         'config': {'time_budget': 1000, 'rollout_budget': 100,
                    'heuristic': lambda d, s: (-d.heuristic(s), 10000),
                    'online_node_garbage': True,
                    'max_depth': 50, 'ucb_constant': 1.0 / sqrt(2.0),
                    'parallel': False, 'debug_logs': False}},

        # PPO: Proximal Policy Optimization (deep reinforcement learning)
        {'name': 'PPO: Proximal Policy Optimization (deep reinforcement learning)',
         'entry': 'StableBaseline',
         'need_domain_factory': False,
         'config': {'algo_class': PPO, 'baselines_policy': 'MlpPolicy',
                    'learn_config': {'total_timesteps': 30000},
                    'verbose': 1}},

        # POMCP: Partially Observable Monte-Carlo Planning (online planning for POMDP)
        {'name': 'POMCP: Partially Observable Monte-Carlo Planning (online planning for POMDP)',
         'entry': 'POMCP',
         'need_domain_factory': False,
         'config': {}},

        # CGP: Cartesian Genetic Programming (evolution strategy)
        {'name': 'CGP: Cartesian Genetic Programming (evolution strategy)',
         'entry': 'CGP',
         'need_domain_factory': False,
         'config': {'folder_name': 'TEMP', 'n_it': 25}},

        # Rollout-IW (classical planning)
        {'name': 'Rollout-IW (classical planning)',
         'entry': 'RIW',
         'need_domain_factory': True,
         'config': {'state_features': lambda d, s: d.state_features(s),
                    'time_budget': 1000, 'rollout_budget': 100,
                    'max_depth': 50, 'exploration': 0.25,
                    'use_simulation_domain': True, 'online_node_garbage': True,
                    'continuous_planning': False,
                    'parallel': False, 'debug_logs': False}},
        
        # IW (classical planning)
        {'name': 'IW (classical planning)',
         'entry': 'IW',
         'need_domain_factory': True,
         'config': {'state_features': lambda d, s: d.state_features(s),
                    'parallel': False, 'debug_logs': False}},
        
        # BFWS (classical planning)
        {'name': 'BFWS (planning) - (num_rows * num_cols) binary encoding (1 binary variable <=> 1 cell)',
         'entry': 'BFWS',
         'need_domain_factory': True,
         'config': {'state_features': lambda d, s: d.state_features(s),
                    'heuristic': lambda d, s: d.heuristic(s),
                    'termination_checker': lambda d, s: d.is_goal(s),
                    'parallel': False, 'debug_logs': False}}
    ]

    # Load domains (filtering out badly installed ones)
    domains = map(lambda d: dict(d, entry=load_registered_domain(d['entry'])), try_domains)
    domains = list(filter(lambda d: d['entry'] is not None, domains))

    # Load solvers (filtering out badly installed ones)
    solvers = map(lambda s: dict(s, entry=load_registered_solver(s['entry'])), try_solvers)
    solvers = list(filter(lambda s: s['entry'] is not None, solvers))
    solvers.insert(0, {'name': 'Random Walk', 'entry': None})  # Add Random Walk as option

    # Run loop to ask user input
    solver_candidates = [s['entry'] for s in solvers if s['entry'] is not None]
    while True:
        # Ask user input to select domain
        domain_choice = int(input('\nChoose a domain:\n{domains}\n'.format(
            domains='\n'.join([f'{i + 1}. {d["name"]}' for i, d in enumerate(domains)]))))
        selected_domain = domains[domain_choice - 1]
        domain_type = selected_domain['entry']
        domain = domain_type(**selected_domain['config'])

        while True:
            # Match solvers compatible with selected domain
            compatible = [None] + match_solvers(domain, candidates=solver_candidates)
            if selected_domain['name'] == 'Cart Pole (OpenAI Gym)' or \
               selected_domain['name'] == 'Mountain Car continuous (OpenAI Gym)':
                # Those gym domain actually have more capabilities than they pretend,
                # so we will transform them later to GymDomainForWidthSolvers (which
                # includes planning domains that UCT can solve)
                compatible += [load_registered_solver('IW'),
                               load_registered_solver('RIW'),
                               load_registered_solver('UCT')]

            # Ask user input to select compatible solver
            solver_choice = int(input('\nChoose a compatible solver:\n{solvers}\n'.format(solvers='\n'.join(
                ['0. [Change domain]'] + [f'{i + 1}. {s["name"]}' for i, s in enumerate(solvers) if
                                          s['entry'] in compatible]))))

            if solver_choice == 0:  # the user wants to change domain
                break
            else:
                selected_solver = solvers[solver_choice - 1]
                solver_type = selected_solver['entry']
                # Set the domain-dependent heuristic for search algorithms
                if selected_domain['name'] == 'Simple Grid World':
                    setattr(domain_type, 'heuristic',
                            lambda self, s: sqrt((self.num_cols - 1 - s.x)**2 + (self.num_rows - 1 - s.y)**2))
                elif selected_domain['name'] == 'Maze':
                    setattr(domain_type, 'heuristic',
                            lambda self, s: sqrt((self._goal.x - s.x)**2 + (self._goal.y - s.y)**2))
                else:
                    setattr(domain_type, 'heuristic',
                            lambda self, s: 0)
                # Set the domain-dependent stat features for width-based algorithms
                if selected_domain['name'] == 'Simple Grid World':
                    setattr(domain_type, 'state_features',
                            lambda self, s: [s.x, s.y])
                elif selected_domain['name'] == 'Maze':
                    setattr(domain_type, 'state_features',
                            lambda self, s: [s.x, s.y])
                elif selected_domain['entry'].__name__ == 'GymDomain':
                    setattr(domain_type, 'state_features',
                            lambda self, s: self.bee1_features(s))
                else:
                    setattr(domain_type, 'state_features',
                            lambda self, s: s)
                # Test solver solution on domain
                print('==================== TEST SOLVER ====================')
                # Check if Random Walk selected or other
                if solver_type is None:
                    rollout(domain, solver=None, **selected_domain['rollout'])
                else:
                    # Solve with selected solver
                    actual_domain_type = domain_type
                    actual_domain = domain
                    if selected_solver['need_domain_factory']:
                        if selected_domain['entry'].__name__ == 'GymDomain' and \
                            (selected_solver['entry'].__name__ == 'IW' or selected_solver['entry'].__name__ == 'BFWS'):
                            actual_domain_type = GymDomainForWidthSolvers
                            actual_domain = actual_domain_type(**selected_domain['config'])
                            selected_solver['config']['node_ordering'] = lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: a_novelty > b_novelty
                        selected_solver['config']['domain_factory'] = lambda: actual_domain_type(**selected_domain['config'])
                    with solver_type(**selected_solver['config']) as solver:
                        actual_domain_type.solve_with(solver, lambda: actual_domain_type(**selected_domain['config']))
                        rollout(actual_domain, solver, **selected_domain['rollout'])
                if hasattr(domain, 'close'):
                    domain.close()
