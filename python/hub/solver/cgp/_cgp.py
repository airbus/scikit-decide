import gym
import numpy as np
from typing import Callable

from pycgp.cgpes import *
from pycgp.cgpfunctions import *

from airlaps import *
from airlaps.builders.domain import *
from airlaps.builders.solver import *

GymSpace = hub.load('GymSpace', folder='hub/space/gym')


class D(Domain, SingleAgent, Sequential, Environment, UnrestrictedActions, Initializable, History, PartiallyObservable,
        Rewards):
    pass


class CGPWrapper(Solver, DeterministicPolicies, Restorable):
    T_domain = D

    def __init__(self, folder_name, library=None, col=100, row=1, nb_ind=4, mutation_rate_nodes=0.1,
                 mutation_rate_outputs=0.3, n_cpus=1, n_it=1000000, genome=None, verbose=True):

        if library is None:
            library = self._get_default_function_lib()

        self._library = library
        self._folder_name = folder_name
        self._col = col
        self._row = row
        self._nb_ind = nb_ind
        self._mutation_rate_nodes = mutation_rate_nodes
        self._mutation_rate_outputs = mutation_rate_outputs
        self._n_cpus = n_cpus
        self._n_it = n_it
        self._genome = genome
        self._verbose = verbose

    @classmethod
    def _check_domain_additional(cls, domain: D) -> bool:
        # TODO: add space conditions (only Box or Tuple of one or more Box from gym spaces)?
        return isinstance(domain.get_action_space(), GymSpace) and isinstance(domain.get_observation_space(), GymSpace)

    def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
        domain = domain_factory()

        evaluator = AirlapsEvaluator(domain)
        if self._genome is None:
            cgpFather = CGP.random(len(domain.get_observation_space().sample()),
                                   len(domain.get_action_space().sample()), self._col, self._row, self._library, 1.0)
        else:
            cgpFather = CGP.load_from_file(self._genome, self._library)

        if self._verbose:
            print(cgpFather.genome)

        es = CGPES(self._nb_ind, self._mutation_rate_nodes, self._mutation_rate_outputs, cgpFather, evaluator,
                   self._folder_name, self._n_cpus)
        es.run(self._n_it)

        self._domain = domain
        self._es = es
        self._evaluator = evaluator

    def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        # TODO: encapsulate in a evaluator function (to avoid redundancy)?
        observation = np.array(list(flatten(observation)))
        observation = 2.0 * (observation - self._evaluator.obs_mins) / (self._evaluator.obs_maxs - self._evaluator.obs_mins) - 1.0
        action = self._es.father.run(observation)
        action = (action + 1.0) * (self._evaluator.act_maxs - self._evaluator.act_mins) / 2 + self._evaluator.act_mins
        return action

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _save(self, path: str) -> None:
        pass  # TODO

    def _load(self, path: str) -> None:
        pass  # TODO

    def _get_default_function_lib(self):
        return [CGP.CGPFunc(f_sum, 'sum', 2),
                CGP.CGPFunc(f_aminus, 'aminus', 2),
                CGP.CGPFunc(f_mult, 'mult', 2),
                CGP.CGPFunc(f_exp, 'exp', 2),
                CGP.CGPFunc(f_abs, 'abs', 1),
                CGP.CGPFunc(f_sqrt, 'sqrt', 1),
                CGP.CGPFunc(f_sqrtxy, 'sqrtxy', 2),
                CGP.CGPFunc(f_squared, 'squared', 1),
                CGP.CGPFunc(f_pow, 'pow', 2),
                CGP.CGPFunc(f_one, 'one', 0),
                CGP.CGPFunc(f_zero, 'zero', 0),
                CGP.CGPFunc(f_inv, 'inv', 1),
                CGP.CGPFunc(f_gt, 'gt', 2),
                CGP.CGPFunc(f_asin, 'asin', 1),
                CGP.CGPFunc(f_acos, 'acos', 1),
                CGP.CGPFunc(f_atan, 'atan', 1),
                CGP.CGPFunc(f_min, 'min', 2),
                CGP.CGPFunc(f_max, 'max', 2),
                CGP.CGPFunc(f_round, 'round', 1),
                CGP.CGPFunc(f_floor, 'floor', 1),
                CGP.CGPFunc(f_ceil, 'ceil', 1)
                ]


def flatten(c):
    """
    Generator flattening the structure

    >>> list(flatten([2, [2, "test", (4, 5, [7], [2, [6, 2, 6, [6], 4]], 6)]]))
    [2, 2, "test", 4, 5, 7, 2, 6, 2, 6, 6, 4, 6]
    """
    from collections.abc import Iterable
    for x in c:
        if isinstance(x, str) or not isinstance(x, Iterable):
            yield x
        else:
            yield from flatten(x)


class AirlapsEvaluator(Evaluator):

    def __init__(self, domain, it_max=10000, ep_max=1):
        super().__init__()
        self.it_max = it_max
        self.ep_max = ep_max
        self.domain = domain

        def get_mins_maxs(space):
            if not isinstance(space, gym.spaces.Tuple):
                space = tuple([space])
            mins = []
            maxs = []
            for box in space:
                mins += list(box.low)
                maxs += list(box.high)
            return np.array(mins), np.array(maxs)

        self.obs_mins, self.obs_maxs = get_mins_maxs(domain.get_observation_space().unwrapped())
        self.act_mins, self.act_maxs = get_mins_maxs(domain.get_action_space().unwrapped())

    def evaluate(self, cgp, it, verbose=False):
        fitnesses = np.zeros(self.ep_max)
        for e in range(self.ep_max):
            end = False
            fit = 0
            states = self.domain.reset()
            step = 0
            while not end and step < self.it_max:
                states = np.array(list(flatten(states)))
                # states = np.minimum(self.obs_maxs, np.maximum(self.obs_mins, states))
                states = 2.0 * (states - self.obs_mins) / (self.obs_maxs - self.obs_mins) - 1.0
                actions = cgp.run(states)
                actions = (actions + 1.0)*(self.act_maxs - self.act_mins)/2 + self.act_mins

                states, transition_value, end, _ = self.domain.step(actions).astuple()
                reward = transition_value[0]  # TODO: correct Gym wrapper

                if verbose:
                    print(states, '=>', actions)

                fit += reward
                step += 1
            fitnesses[e] = fit
        np.sort(fitnesses)
        fit = 0
        sum_e = 0
        for e in range(self.ep_max):
            fit += fitnesses[e] * (e + 1)
            sum_e += e + 1

        return fit / sum_e

    def clone(self):
        return AirlapsEvaluator(self.domain, self.it_max, self.ep_max)
