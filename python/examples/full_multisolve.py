import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from airlaps import hub
from airlaps.utils import rollout, match_solvers


if __name__ == '__main__':

    try_domains = [

        # Simple Grid World
        {'name': 'Simple Grid World',
         'type': {'entry': 'SimpleGridWorld', 'folder': 'hub/domain/simple_grid_world'},
         'config': {},
         'rollout': {'max_steps': 1000, 'outcome_formatter': lambda o: f'{o.observation} - cost: {o.value.cost:.2f}'}},

        # Maze
        {'name': 'Maze',
         'type': {'entry': 'Maze', 'folder': 'hub/domain/maze'},
         'config': {},
         'rollout': {'max_steps': 1000, 'max_framerate': 30,
                     'outcome_formatter': lambda o: f'{o.observation} - cost: {o.value.cost:.2f}'}},

        # Mastermind
        {'name': 'Mastermind',
         'type': {'entry': 'MasterMind', 'folder': 'hub/domain/mastermind'},
         'config': {'n_colours': 3, 'n_positions': 3},
         'rollout': {'max_steps': 1000, 'outcome_formatter': lambda o: f'{o.observation} - cost: {o.value.cost:.2f}'}},

        # Cart Pole (OpenAI Gym)
        {'name': 'Cart Pole (OpenAI Gym)',
         'type': {'entry': 'GymDomain', 'folder': 'hub/domain/gym'},
         'config': {'gym_env': gym.make('CartPole-v1')},
         'rollout': {'num_episodes': 3, 'max_steps': 1000, 'max_framerate': 30, 'outcome_formatter': None}},

        # Mountain Car continuous (OpenAI Gym)
        {'name': 'Mountain Car continuous (OpenAI Gym)',
         'type': {'entry': 'GymDomain', 'folder': 'hub/domain/gym'},
         'config': {'gym_env': gym.make('MountainCarContinuous-v0')},
         'rollout': {'num_episodes': 3, 'max_steps': 1000, 'max_framerate': 30, 'outcome_formatter': None}},

        # ATARI Pacman (OpenAI Gym)
        {'name': 'ATARI Pacman (OpenAI Gym)',
         'type': {'entry': 'GymDomain', 'folder': 'hub/domain/gym'},
         'config': {'gym_env': gym.make('MsPacman-v4')},
         'rollout': {'num_episodes': 3, 'max_steps': 1000, 'max_framerate': 30, 'outcome_formatter': None}}

    ]

    try_solvers = [

        # Random walk
        {'name': 'Random walk', 'type': None},

        # Simple greedy
        {'name': 'Simple greedy', 'type': {'entry': 'SimpleGreedy', 'folder': 'hub/solver/simple_greedy'},
         'config': {}},

        # Lazy A* (classical planning)
        {'name': 'Lazy A* (classical planning)', 'type': {'entry': 'LazyAstar', 'folder': 'hub/solver/lazy_astar'},
         'config': {'verbose': True}},

        # PPO: Proximal Policy Optimization (deep reinforcement learning)
        {'name': 'PPO: Proximal Policy Optimization (deep reinforcement learning)',
         'type': {'entry': 'StableBaselines', 'folder': 'hub/solver/stable_baselines'},
         'config': {'algo_class': PPO2, 'baselines_policy': MlpPolicy, 'learn_config': {'total_timesteps': 25000},
                    'verbose': 1}},

        # POMCP: Partially Observable Monte-Carlo Planning (online planning for POMDP)
        {'name': 'POMCP: Partially Observable Monte-Carlo Planning (online planning for POMDP)',
         'type': {'entry': 'POMCP', 'folder': 'hub/solver/pomcp'}, 'config': {}},

        # CGP: Cartesian Genetic Programming (evolution strategy)
        {'name': 'CGP: Cartesian Genetic Programming (evolution strategy)',
         'type': {'entry': 'CGP', 'folder': 'hub/solver/cgp'}, 'config': {'folder_name': 'TEMP', 'n_it': 25}},

        # IW: Iterated Width search (width-based planning)
        {'name': 'IW: Iterated Width search (width-based planning)', 'type': {'entry': 'IW', 'folder': 'hub/solver/iw'},
         'config': {}}
    ]

    # Load domains (if installed)
    domains = []
    for d in try_domains:
        try:
            if d['type'] is not None:
                d['type'] = hub.load(**d['type'])
            domains.append(d)
        except Exception:
            print(rf'/!\ Could not load {d["name"]} from hub: check installation & missing dependencies')

    # Load solvers (if installed)
    solvers = []
    for s in try_solvers:
        try:
            if s['type'] is not None:
                s['type'] = hub.load(**s['type'])
            solvers.append(s)
        except Exception:
            print(rf'/!\ Could not load {s["name"]} from hub: check installation & missing dependencies')

    # Run loop to ask user input
    solver_candidates = [s['type'] for s in solvers if s['type'] is not None]
    while True:
        # Ask user input to select domain
        domain_choice = int(input('\nChoose a domain:\n{domains}\n'.format(
            domains='\n'.join([f'{i + 1}. {d["name"]}' for i, d in enumerate(domains)]))))
        selected_domain = domains[domain_choice - 1]
        domain_type = selected_domain['type']
        domain = domain_type(**selected_domain['config'])

        while True:
            # Match solvers compatible with selected domain
            compatible = [None] + match_solvers(domain, candidates=solver_candidates, add_local_hub=False)

            # Ask user input to select compatible solver
            solver_choice = int(input('\nChoose a compatible solver:\n{solvers}\n'.format(solvers='\n'.join(
                ['0. [Change domain]'] + [f'{i + 1}. {s["name"]}' for i, s in enumerate(solvers) if
                                          s['type'] in compatible]))))

            if solver_choice == 0:  # the user wants to change domain
                break
            else:
                selected_solver = solvers[solver_choice - 1]
                solver_type = selected_solver['type']
                if solver_type is None:
                    solver = None
                else:
                    # Check that the solver is compatible with the domain
                    assert solver_type.check_domain(domain)
                    # Solve with selected solver
                    solver = domain_type.solve_with(lambda: solver_type(**selected_solver['config']),
                                                    lambda: domain_type(**selected_domain['config']))
                # Test solver solution on domain
                print('==================== TEST SOLVER ====================')
                rollout(domain, solver, **selected_domain['rollout'])
                if hasattr(domain, 'close'):
                    domain.close()
