# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable
from collections.abc import Iterable

import gym
import numpy as np

from skdecide import Domain, Solver
from skdecide.builders.domain import SingleAgent, Sequential, Environment, UnrestrictedActions, Initializable, History, \
    PartiallyObservable, Rewards
from skdecide.builders.solver import DeterministicPolicies, Restorable
from skdecide.hub.space.gym import GymSpace
from .pycgp.cgpes import CGP, CGPES, Evaluator
from .pycgp.cgpfunctions import f_sum, f_aminus, f_mult, f_exp, f_abs, f_sqrt, f_sqrtxy, f_squared, f_pow, f_one, \
    f_zero, f_inv, f_gt, f_acos, f_asin, f_atan, f_min, f_max, f_round, f_floor, f_ceil


class D(Domain, SingleAgent, Sequential, Environment, UnrestrictedActions, Initializable, History, PartiallyObservable,
        Rewards):
    pass


def change_interval(x, inmin, inmax, outmin, outmax):
    # making sure x is in the interval
    x = max(inmin, min(inmax, x))
    # normalizing x between 0 and 1
    x = (x - inmin) / (inmax - inmin)
    # denormalizing between outmin and outmax
    return x * (outmax - outmin) + outmin


def change_float_to_int_interval(x, inmin, inmax, outdiscmin, outdiscmax):
    x = change_interval(x, inmin, inmax, 0, 1)
    if x == 1:
        return outdiscmax
    else:
        return int(x * (outdiscmax - outdiscmin + 1) + outdiscmin)


def flatten(c):
    """
    Generator flattening the structure

    >>> list(flatten([2, [2, "test", [4,5, [7], [2, [6, 2, 6, [6], 4]], 6]]]))
    [2, 2, "test", 4, 5, 7, 2, 6, 2, 6, 6, 4, 6]
    """
    for x in c:
        if isinstance(x, str) or not isinstance(x, Iterable):
            yield x
        else:
            yield from flatten(x)


def norm_and_flatten(vals, types):
    """
    Flatten and normalise according to AIGYM type (BOX, DISCRETE, TUPLE)
    :param vals: a np array structure
    :param types: the gym type corresponding to the vals arrays
    :return: a flatten array with normalised vals
    """

    if not isinstance(types, Iterable) and \
            not isinstance(types, gym.spaces.Tuple):
        types = [types]
    if not isinstance(vals, Iterable) and \
            not isinstance(vals, gym.spaces.Tuple):
        vals = [vals]

    flat_vals = list(flatten(vals))
    index = 0
    for i in range(len(types)):
        t = types[i]
        if isinstance(t, gym.spaces.Box):
            lows = list(flatten(t.low))
            highs = list(flatten(t.high))
            for j in range(len(lows)):
                flat_vals[index] = change_interval(flat_vals[index], lows[j], highs[j], -1, 1)
                index += 1
        elif isinstance(t, gym.spaces.Discrete):
            flat_vals[index] = change_interval(flat_vals[index], 0, t.n - 1, -1, 1)
            index += 1
        else:
            raise ValueError("Unsupported type ", str(t))

    return flat_vals


def norm(vals, types):
    """
    Normalise according to AIGYM type (BOX, DISCRETE, TUPLE)
    :param vals: a np array structure
    :param types: the gym type corresponding to the vals arrays
    :return: array with normalised vals
    """

    temp_vals = list(vals)
    temp_types = types
    if not isinstance(types, gym.spaces.Tuple):
        temp_types = [types]
        temp_vals = [temp_vals]

    for i in range(len(temp_types)):
        t = temp_types[i]
        if isinstance(t, gym.spaces.Box):
            lows = list(flatten(t.low))
            highs = list(flatten(t.high))
            for j in range(len(lows)):
                temp_vals[i][j] = change_interval(temp_vals[i][j], lows[j], highs[j], -1, 1)
        elif isinstance(t, gym.spaces.Discrete):
            temp_vals[i] = change_interval(temp_vals[i], 0, t.n - 1, -1, 1)
        else:
            raise ValueError("Unsupported type ", str(t))

    if not isinstance(types, gym.spaces.Tuple):
        temp_vals = temp_vals[0]

    return temp_vals


def denorm(vals, types):
    """
    Denormalize values according to AIGYM types (BOX, DISCRETE, TUPLE)
    :param vals: an array of [-1,1] normalised values
    :param types: the gym types corresponding to vals
    :return: the same vals array with denormalised values
    """
    if not isinstance(types, Iterable) and \
            not isinstance(types, gym.spaces.Tuple):
        types = [types]
    if not isinstance(vals, Iterable) and \
            not isinstance(vals, gym.spaces.Tuple):
        vals = [vals]
    out = []
    index = 0
    for i in range(len(types)):
        t = types[i]
        if isinstance(t, gym.spaces.Box):
            out_temp = []
            for j in range(len(t.low)):
                out_temp += [change_interval(vals[index], -1, 1, t.low[j], t.high[j])]
                index += 1
            out += out_temp
        elif isinstance(t, gym.spaces.Discrete):
            out += [change_float_to_int_interval(vals[index], -1, 1, 0, t.n - 1)]
            index += 1
        else:
            raise ValueError("Unsupported type ", str(t))
    # burk
    if len(types) == 1 and not isinstance(types[0], gym.spaces.Box):
        return out[0]
    else:
        return out


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
        """
        CGP manage all kind of gym types, BOX, DISCRETE and TUPLE as well
        """
        return isinstance(domain.get_action_space(), GymSpace) and isinstance(domain.get_observation_space(), GymSpace)

    def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
        domain = domain_factory()

        evaluator = SkDecideEvaluator(domain)
        if self._genome is None:
            a = domain.get_action_space().sample()
            if isinstance(a, Iterable) or isinstance(a, gym.spaces.Tuple):
                num_outputs = len(a)
            else:
                num_outputs = 1
            cgpFather = CGP.random(len(domain.get_observation_space().sample()),
                                   num_outputs, self._col, self._row, self._library, 1.0)
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

        return denorm(
            self._es.father.run(norm_and_flatten(observation, self._domain.get_observation_space().unwrapped())),
            self._domain.get_action_space().unwrapped())

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


class SkDecideEvaluator(Evaluator):

    def __init__(self, domain, it_max=10000, ep_max=1):
        super().__init__()
        self.it_max = it_max
        self.ep_max = ep_max
        self.domain = domain

        # def get_mins_maxs(space):
        #     if not isinstance(space, gym.spaces.Tuple):
        #         space = tuple([space])
        #     mins = []
        #     maxs = []
        #     for box in space:
        #         mins += list(box.low)
        #         maxs += list(box.high)
        #     return np.array(mins), np.array(maxs)

        # self.obs_mins, self.obs_maxs = get_mins_maxs(domain.get_observation_space().unwrapped())
        # #self.act_mins, self.act_maxs = 0,2#get_mins_maxs(domain.get_action_space().unwrapped())

    def evaluate(self, cgp, it, verbose=False):
        fitnesses = np.zeros(self.ep_max)
        for e in range(self.ep_max):
            end = False
            fit = 0
            states = self.domain.reset()
            step = 0
            while not end and step < self.it_max:

                actions = denorm(cgp.run(norm_and_flatten(states, self.domain.get_observation_space().unwrapped())),
                                 self.domain.get_action_space().unwrapped())
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
        return SkDecideEvaluator(self.domain, self.it_max, self.ep_max)
