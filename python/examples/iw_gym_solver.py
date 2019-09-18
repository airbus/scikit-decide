import gym

from airlaps import hub
from airlaps.utils import rollout

GymWidthPlanningDomain = hub.load('GymWidthPlanningDomain', folder='hub/domain/gym')
IW = hub.load('IW', folder='hub/solver/iw')

ENV_NAME = 'CartPole-v0'
HORIZON = 20

domain_factory = lambda: GymWidthPlanningDomain(gym_env=gym.make(ENV_NAME),
                                                discretization_factor=10,
                                                max_depth=HORIZON)
domain = domain_factory()

if IW.check_domain(domain):
    solver_factory = lambda: IW(nb_of_binary_features=lambda d: d.nb_of_binary_features(),
                                state_binarizer=lambda s, d, f: d.binarize(s, f),
                                use_state_feature_hash=True,
                                parallel=False, debug_logs=False)
    solver = GymWidthPlanningDomain.solve_with(solver_factory, domain_factory)
    rollout(domain, solver, num_episodes=5, max_steps=HORIZON, max_framerate=30, outcome_formatter=None)

# s = domain.reset()
# bv = []
# domain.binarize(s, lambda i: bv.append(i))
# print('nb of binary features: ', str(domain.nb_of_binary_features()))
# print('Bit vector:', str(bv))
# print('applicable actions:', str(domain.get_applicable_actions(s).get_elements()))
