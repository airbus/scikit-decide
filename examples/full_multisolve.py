# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
from stable_baselines3 import PPO

from skdecide.utils import load_registered_domain, load_registered_solver, match_solvers, rollout


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
         'config': {}},

        # Lazy A* (classical planning)
        {'name': 'Lazy A* (classical planning)',
         'entry': 'LazyAstar',
         'config': {'verbose': True}},

        # PPO: Proximal Policy Optimization (deep reinforcement learning)
        {'name': 'PPO: Proximal Policy Optimization (deep reinforcement learning)',
         'entry': 'StableBaseline',
         'config': {'algo_class': PPO, 'baselines_policy': 'MlpPolicy', 'learn_config': {'total_timesteps': 30000},
                    'verbose': 1}},

        # POMCP: Partially Observable Monte-Carlo Planning (online planning for POMDP)
        {'name': 'POMCP: Partially Observable Monte-Carlo Planning (online planning for POMDP)',
         'entry': 'POMCP',
         'config': {}},

        # CGP: Cartesian Genetic Programming (evolution strategy)
        {'name': 'CGP: Cartesian Genetic Programming (evolution strategy)',
         'entry': 'CGP',
         'config': {'folder_name': 'TEMP', 'n_it': 25}},

        # IW: Iterated Width search (width-based planning)
        {'name': 'IW: Iterated Width search (width-based planning)',
         'entry': 'IW',
         'config': {}}
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

            # Ask user input to select compatible solver
            solver_choice = int(input('\nChoose a compatible solver:\n{solvers}\n'.format(solvers='\n'.join(
                ['0. [Change domain]'] + [f'{i + 1}. {s["name"]}' for i, s in enumerate(solvers) if
                                          s['entry'] in compatible]))))

            if solver_choice == 0:  # the user wants to change domain
                break
            else:
                selected_solver = solvers[solver_choice - 1]
                solver_type = selected_solver['entry']
                # Test solver solution on domain
                print('==================== TEST SOLVER ====================')
                # Check if Random Walk selected or other
                if solver_type is None:
                    rollout(domain, solver=None, **selected_domain['rollout'])
                else:
                    # Solve with selected solver
                    with solver_type(**selected_solver['config']) as solver:
                        domain_type.solve_with(solver, lambda: domain_type(**selected_domain['config']))
                        rollout(domain, solver, **selected_domain['rollout'])
                if hasattr(domain, 'close'):
                    domain.close()
