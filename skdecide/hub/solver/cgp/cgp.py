# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import gymnasium as gym
import numpy as np
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
    IntegerHyperparameter,
)

from skdecide import Domain, Solver
from skdecide.builders.domain import (
    Environment,
    History,
    Initializable,
    PartiallyObservable,
    Rewards,
    Sequential,
    SingleAgent,
    UnrestrictedActions,
)
from skdecide.builders.solver import DeterministicPolicies

from .pycgp.cgpes import CGP, CGPES, Evaluator
from .pycgp.cgpfunctions import (
    f_abs,
    f_acos,
    f_aminus,
    f_asin,
    f_atan,
    f_ceil,
    f_exp,
    f_floor,
    f_gt,
    f_inv,
    f_max,
    f_min,
    f_mult,
    f_one,
    f_pow,
    f_round,
    f_sqrt,
    f_sqrtxy,
    f_squared,
    f_sum,
    f_zero,
)


class D(
    Domain,
    SingleAgent,
    Sequential,
    Environment,
    UnrestrictedActions,
    Initializable,
    History,
    PartiallyObservable,
    Rewards,
):
    pass


def change_interval(x, inmin, inmax, outmin, outmax):
    # redefine interval if min, max are set to +-infinity by the GYM environment
    # TODO: maybe we could reject those environments in the future.
    if inmin == -np.inf:
        inmin = -1
    if inmax == np.inf:
        inmax = 1
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

    if not isinstance(types, Iterable) and not isinstance(types, gym.spaces.Tuple):
        types = [types]
    if not isinstance(vals, Iterable) and not isinstance(vals, gym.spaces.Tuple):
        vals = [vals]

    flat_vals = list(flatten(vals))
    index = 0
    for i in range(len(types)):
        t = types[i]
        if isinstance(t, gym.spaces.Box):
            lows = list(flatten(t.low))
            highs = list(flatten(t.high))
            for j in range(len(lows)):
                flat_vals[index] = change_interval(
                    flat_vals[index], lows[j], highs[j], -1, 1
                )
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
                temp_vals[i][j] = change_interval(
                    temp_vals[i][j], lows[j], highs[j], -1, 1
                )
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
    if not isinstance(types, Iterable) and not isinstance(types, gym.spaces.Tuple):
        types = [types]
    if not isinstance(vals, Iterable) and not isinstance(vals, gym.spaces.Tuple):
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


class CGPWrapper(Solver, DeterministicPolicies):
    """Cartesian Genetic Programming solver."""

    T_domain = D

    hyperparameters = [
        IntegerHyperparameter(name="col"),
        IntegerHyperparameter(name="row"),
        IntegerHyperparameter(name="nb_ind"),
        FloatHyperparameter(name="mutation_rate_nodes"),
        FloatHyperparameter(name="mutation_rate_outputs"),
        IntegerHyperparameter(name="n_it"),
    ]

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        folder_name,
        library=None,
        col=100,
        row=1,
        nb_ind=4,
        mutation_rate_nodes=0.1,
        mutation_rate_outputs=0.3,
        n_cpus=1,
        n_it=1000000,
        genome=None,
        verbose=True,
        callback: Callable[[CGPWrapper], bool] = lambda solver: False,
    ):
        """

        # Parameters
        domain_factory
        folder_name
        library
        col
        row
        nb_ind
        mutation_rate_nodes
        mutation_rate_outputs
        n_cpus
        n_it
        genome
        verbose
        callback: function called at each solver iteration. If returning true, the solve process stops.

        """
        Solver.__init__(self, domain_factory=domain_factory)
        self.callback = callback
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
        action_space = domain.get_action_space().unwrapped()
        observation_space = domain.get_observation_space().unwrapped()

        if not isinstance(action_space, Iterable) and not isinstance(
            action_space, gym.spaces.Tuple
        ):
            action_space = [action_space]
        if not isinstance(observation_space, Iterable) and not isinstance(
            observation_space, gym.spaces.Tuple
        ):
            observation_space = [observation_space]

        flat_action_space = list(flatten(action_space))
        flat_observation_space = list(flatten(observation_space))

        print(flat_action_space)
        print(flat_observation_space)

        valide_action_space = True
        for x in flat_action_space:
            valide_action_space = isinstance(
                x, (gym.spaces.Tuple, gym.spaces.Discrete, gym.spaces.Box)
            )

        validate_observation_space = True
        for x in flat_observation_space:
            validate_observation_space = isinstance(
                x, (gym.spaces.Tuple, gym.spaces.Discrete, gym.spaces.Box)
            )

        return valide_action_space and validate_observation_space

    def _solve(self) -> None:
        domain = self._domain_factory()

        evaluator = SkDecideEvaluator(domain)
        if self._genome is None:
            a = domain.get_action_space().sample()
            b = domain.get_observation_space().sample()
            if isinstance(a, Iterable) or isinstance(a, gym.spaces.Tuple):
                num_outputs = len(a)
            else:
                num_outputs = 1
            if isinstance(b, Iterable) or isinstance(b, gym.spaces.Tuple):
                num_inputs = len(b)
            else:
                num_inputs = 1
            cgpFather = CGP.random(
                num_inputs, num_outputs, self._col, self._row, self._library, 1.0
            )
        else:
            cgpFather = CGP.load_from_file(self._genome, self._library)

        if self._verbose:
            print(cgpFather.genome)

        es = CGPES(
            num_offsprings=self._nb_ind,
            mutation_rate_nodes=self._mutation_rate_nodes,
            mutation_rate_outputs=self._mutation_rate_outputs,
            father=cgpFather,
            evaluator=evaluator,
            folder=self._folder_name,
            num_cpus=self._n_cpus,
            verbose=self._verbose,
            callback=self.callback,
            cgpwrapper=self,
        )
        self._domain = domain
        self._es = es
        self._evaluator = evaluator

        es.run(self._n_it)

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:

        return denorm(
            self._es.father.run(
                norm_and_flatten(
                    observation, self._domain.get_observation_space().unwrapped()
                )
            ),
            self._domain.get_action_space().unwrapped(),
        )

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _get_default_function_lib(self):
        return [
            CGP.CGPFunc(f_sum, "sum", 2),
            CGP.CGPFunc(f_aminus, "aminus", 2),
            CGP.CGPFunc(f_mult, "mult", 2),
            CGP.CGPFunc(f_exp, "exp", 2),
            CGP.CGPFunc(f_abs, "abs", 1),
            CGP.CGPFunc(f_sqrt, "sqrt", 1),
            CGP.CGPFunc(f_sqrtxy, "sqrtxy", 2),
            CGP.CGPFunc(f_squared, "squared", 1),
            CGP.CGPFunc(f_pow, "pow", 2),
            CGP.CGPFunc(f_one, "one", 0),
            CGP.CGPFunc(f_zero, "zero", 0),
            CGP.CGPFunc(f_inv, "inv", 1),
            CGP.CGPFunc(f_gt, "gt", 2),
            CGP.CGPFunc(f_asin, "asin", 1),
            CGP.CGPFunc(f_acos, "acos", 1),
            CGP.CGPFunc(f_atan, "atan", 1),
            CGP.CGPFunc(f_min, "min", 2),
            CGP.CGPFunc(f_max, "max", 2),
            CGP.CGPFunc(f_round, "round", 1),
            CGP.CGPFunc(f_floor, "floor", 1),
            CGP.CGPFunc(f_ceil, "ceil", 1),
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
                actions = denorm(
                    cgp.run(
                        norm_and_flatten(
                            states, self.domain.get_observation_space().unwrapped()
                        )
                    ),
                    self.domain.get_action_space().unwrapped(),
                )
                states, transition_value, end, _ = self.domain.step(actions).astuple()
                reward = transition_value[0]  # TODO: correct Gym wrapper

                if verbose:
                    print(states, "=>", actions)

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
