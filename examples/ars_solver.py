import gym
import gym_jsbsim
from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.solver.ars import ars
from skdecide.utils import load_registered_domain, rollout


if __name__ == '__main__':

    try_domains = [

        {'name': 'Heading Control Task',
         'entry': 'GymDomain',
         'config': {'gym_env': gym.make('GymJsbsim-HeadingControlTask-v0')},
         'rollout': {'num_episodes': 3, 'max_steps': 500, 'max_framerate': 30, 'outcome_formatter': None}},

        {'name': 'Approach Control Task',
         'entry': 'GymDomain',
         'config': {'gym_env': gym.make('GymJsbsim-ApproachControlTask-v0')},
         'rollout': {'num_episodes': 3, 'max_steps': 500, 'max_framerate': 30, 'outcome_formatter': None}},

        {'name': 'Taxi Ap Control Task',
         'entry': 'GymDomain',
         'config': {'gym_env': gym.make('GymJsbsim-TaxiapControlTask-v0')},
         'rollout': {'num_episodes': 3, 'max_steps': 200, 'max_framerate': 30, 'outcome_formatter': None}},

        {'name': 'Taxi Control Task',
         'entry': 'GymDomain',
         'config': {'gym_env': gym.make('GymJsbsim-TaxiControlTask-v0')},
         'rollout': {'num_episodes': 3, 'max_steps': 200, 'max_framerate': 30, 'outcome_formatter': None}},

        # Mountain Car Continuous
        {'name': 'Mountain car continuous',
         'entry': 'GymDomain',
         'config': {'gym_env': gym.make('MountainCarContinuous-v0')},
         'rollout': {'num_episodes': 3, 'max_steps': 200, 'max_framerate': 30, 'outcome_formatter': None}},

        # Mountain Car Discrete
        {'name': 'Mountain car discrete',
         'entry': 'GymDomain',
         'config': {'gym_env': gym.make('MountainCar-v0')},
         'rollout': {'num_episodes': 3, 'max_steps': 200, 'max_framerate': 30, 'outcome_formatter': None}},

        # Cart pole Discrete
        {'name': 'Cart Pole',
         'entry': 'GymDomain',
         'config': {'gym_env': gym.make('CartPole-v1')},
         'rollout': {'num_episodes': 3, 'max_steps': 200, 'max_framerate': 30, 'outcome_formatter': None}}

    ]

    # Load domains (filtering out badly installed ones)
    domains = map(lambda d: dict(d, entry=load_registered_domain(d['entry'])), try_domains)
    domains = list(filter(lambda d: d['entry'] is not None, domains))

    while True:
        # Ask user input to select domain
        domain_choice = int(input('\nChoose a domain:\n{domains}\n'.format(
            domains='\n'.join([f'{i + 1}. {d["name"]}' for i, d in enumerate(domains)]))))

        # for learning to fly
        if domain_choice <=4:
            n_epochs = 500
            epoch_size = 200
            directions = 10
            top_directions = 3
            learning_rate = 0.02
            policy_noise = 0.03
            reward_maximization = True
        else:
            n_epochs = 300
            epoch_size = 200
            directions = 25
            top_directions = 3
            learning_rate = 1
            policy_noise = 1
            reward_maximization = True

        selected_domain = domains[domain_choice - 1]
        domain_type = selected_domain['entry']
        domain = domain_type(**selected_domain['config'])

        solver_factory = lambda: ars.AugmentedRandomSearch(n_epochs=n_epochs, epoch_size=epoch_size, directions=directions,
                                                              top_directions=top_directions, learning_rate=learning_rate, policy_noise=policy_noise, reward_maximization=reward_maximization)
        with solver_factory() as solver:
            GymDomain.solve_with(solver, lambda: domain_type(**selected_domain['config']))
            # Test solver solution on domain
            print('==================== TEST SOLVER ====================')
            print(domain.get_observation_space().unwrapped(), '===>', domain.get_action_space().unwrapped())
            rollout(domain, solver, **selected_domain['rollout'])
        if hasattr(domain, 'close'):
            domain.close()
