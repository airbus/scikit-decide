# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
import random
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
from deap import algorithms, creator, gp, tools
from deap.base import Fitness, Toolbox
from deap.gp import PrimitiveSet, PrimitiveTree, genHalfAndHalf
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
    SubBrickKwargsHyperparameter,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import (
    Hyperparametrizable,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution
from discrete_optimization.rcpsp.solver.cpm import CPM
from scipy import stats
from scipy.spatial import distance

from skdecide import Solver
from skdecide.builders.domain.scheduling.modes import SingleMode
from skdecide.builders.domain.scheduling.scheduling_domains import D, SchedulingDomain
from skdecide.builders.domain.scheduling.scheduling_domains_modelling import (
    SchedulingAction,
    State,
    rebuild_all_tasks_dict,
    rebuild_tasks_complete_details_dict,
)
from skdecide.builders.solver.policy import DeterministicPolicies
from skdecide.hub.solver.do_solver.do_solver_scheduling import (
    DOSolver,
    PolicyMethodParams,
    PolicyRCPSP,
    SolvingMethod,
)
from skdecide.hub.solver.do_solver.sgs_policies import BasePolicyMethod
from skdecide.hub.solver.do_solver.sk_to_do_binding import build_do_domain


def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2


def protected_div(left, right):
    if right != 0.0:
        return left / right
    else:
        return 1.0


def max_operator(left, right):
    return max(left, right)


def min_operator(left, right):
    return min(left, right)


def feature_task_duration(
    domain: SchedulingDomain, cpm, cpm_esd, task_id: int, **kwargs
):
    return domain.sample_task_duration(task_id)


def feature_total_n_res(domain: SchedulingDomain, cpm, cpm_esd, task_id: int, **kwargs):
    val = 0
    mode_consumption = domain.get_task_modes(task_id)[1]
    for res in mode_consumption.get_ressource_names():
        val += mode_consumption.get_resource_need(res)
    return val


def feature_n_successors(
    domain: SchedulingDomain, cpm, cpm_esd, task_id: int, **kwargs
):
    return len(domain.get_successors_task(task_id)) / len(domain.get_tasks_ids())


def feature_n_predecessors(
    domain: SchedulingDomain, cpm, cpm_esd, task_id: int, **kwargs
):
    return len(domain.get_predecessors_task(task_id)) / len(domain.get_tasks_ids())


def get_resource_requirements_across_duration(
    domain: SchedulingDomain, task_id: int, **kwargs
):
    values = []
    mode_consumption = domain.get_task_modes(task_id)[1]
    duration = domain.get_latest_sampled_duration(task_id, 1, 0.0)
    if duration > 0:
        for res in mode_consumption.get_ressource_names():
            tmp = 0
            for t in range(duration):
                need = mode_consumption.get_resource_need_at_time(res, t)
                total = domain.sample_quantity_resource(res, t)
                tmp += need / total
            values.append(tmp / duration)
    else:
        values = [0.0]
    # print(task_id,':', values)
    return values


def feature_average_resource_requirements(
    domain: SchedulingDomain, cpm, cpm_esd, task_id: int, **kwargs
):
    values = get_resource_requirements_across_duration(domain=domain, task_id=task_id)
    val = np.mean(values)
    return val


def feature_minimum_resource_requirements(
    domain: SchedulingDomain, cpm, cpm_esd, task_id: int, **kwargs
):
    values = get_resource_requirements_across_duration(domain=domain, task_id=task_id)
    val = np.min(values)
    return val


def feature_non_zero_minimum_resource_requirements(
    domain: SchedulingDomain, cpm, cpm_esd, task_id: int, **kwargs
):
    values = get_resource_requirements_across_duration(domain=domain, task_id=task_id)
    if np.sum(values) > 0.0:
        val = np.min([x for x in values if x > 0.0])
    else:
        val = np.min(values)
    return val


def feature_maximum_resource_requirements(
    domain: SchedulingDomain, cpm, cpm_esd, task_id: int, **kwargs
):
    values = get_resource_requirements_across_duration(domain=domain, task_id=task_id)
    val = np.max(values)
    return val


def feature_resource_requirements(
    domain: SchedulingDomain, cpm, cpm_esd, task_id: int, **kwargs
):
    values = get_resource_requirements_across_duration(domain=domain, task_id=task_id)
    val = len([x for x in values if x > 0.0]) / len(values)
    return val


def feature_all_descendants(
    domain: SchedulingDomain, cpm, cpm_esd, task_id: int, **kwargs
):
    return len(domain.full_successors[task_id]) / len(domain.get_tasks_ids())


def feature_precedence_done(
    domain: SchedulingDomain, cpm, cpm_esd, task_id: int, state: State, **kwargs
):
    return task_id in domain.task_possible_to_launch_precedence(state=state)


def compute_cpm(do_domain):
    cpm_solver = CPM(do_domain)
    path = cpm_solver.run_classic_cpm()
    cpm = cpm_solver.map_node
    cpm_esd = cpm[path[-1]]._ESD  # to normalize...
    return cpm, cpm_esd


def feature_esd(domain: SchedulingDomain, cpm, cpm_esd, task_id: int, **kwargs):
    """Will only work if you store cpm results into the object"""
    return cpm[task_id]._ESD / cpm_esd


def feature_lsd(domain: SchedulingDomain, cpm, cpm_esd, task_id: int, **kwargs):
    """Will only work if you store cpm results into the object"""
    return cpm[task_id]._LSD / cpm_esd


def feature_efd(domain: SchedulingDomain, cpm, cpm_esd, task_id: int, **kwargs):
    """Will only work if you store cpm results into the object"""
    return cpm[task_id]._EFD / cpm_esd


def feature_lfd(domain: SchedulingDomain, cpm, cpm_esd, task_id: int, **kwargs):
    """Will only work if you store cpm results into the object"""
    return cpm[task_id]._LFD / cpm_esd


class D(SchedulingDomain, SingleMode):
    pass


class FeatureEnum(Enum):
    TASK_DURATION = "task_duration"
    RESSOURCE_TOTAL = "total_nres"
    N_SUCCESSORS = "n_successors"
    N_PREDECESSORS = "n_predecessors"
    RESSOURCE_REQUIRED = "res_requ"
    RESSOURCE_AVG = "avg_res_requ"
    RESSOURCE_MIN = "min_res_requ"
    RESSOURCE_NZ_MIN = "nz_min_res_requ"
    RESSOURCE_MAX = "max_res_requ"
    ALL_DESCENDANTS = "all_descendants"
    PRECEDENCE_DONE = "precedence_done"
    EARLIEST_START_DATE = "ESD"
    LATEST_START_DATE = "LSD"
    EARLIEST_FINISH_DATE = "EFD"
    LATEST_FINISH_DATE = "LFD"


feature_function_map = {
    FeatureEnum.TASK_DURATION: feature_task_duration,
    FeatureEnum.RESSOURCE_TOTAL: feature_total_n_res,
    FeatureEnum.N_SUCCESSORS: feature_n_successors,
    FeatureEnum.N_PREDECESSORS: feature_n_predecessors,  #
    FeatureEnum.RESSOURCE_REQUIRED: feature_resource_requirements,  #
    FeatureEnum.RESSOURCE_AVG: feature_average_resource_requirements,  #
    FeatureEnum.RESSOURCE_MIN: feature_minimum_resource_requirements,  #
    FeatureEnum.RESSOURCE_NZ_MIN: feature_non_zero_minimum_resource_requirements,  #
    FeatureEnum.RESSOURCE_MAX: feature_maximum_resource_requirements,  #
    FeatureEnum.ALL_DESCENDANTS: feature_all_descendants,  #
    FeatureEnum.PRECEDENCE_DONE: feature_precedence_done,
    FeatureEnum.EARLIEST_START_DATE: feature_esd,  #
    FeatureEnum.EARLIEST_FINISH_DATE: feature_efd,  #
    FeatureEnum.LATEST_START_DATE: feature_lsd,  #
    FeatureEnum.LATEST_FINISH_DATE: feature_lfd,
}  #

feature_static_map = {
    FeatureEnum.TASK_DURATION: True,
    FeatureEnum.RESSOURCE_TOTAL: True,
    FeatureEnum.N_SUCCESSORS: True,
    FeatureEnum.N_PREDECESSORS: True,  #
    FeatureEnum.RESSOURCE_REQUIRED: True,  #
    FeatureEnum.RESSOURCE_AVG: True,  #
    FeatureEnum.RESSOURCE_MIN: True,  #
    FeatureEnum.RESSOURCE_NZ_MIN: True,  #
    FeatureEnum.RESSOURCE_MAX: True,  #
    FeatureEnum.ALL_DESCENDANTS: True,  #
    FeatureEnum.PRECEDENCE_DONE: False,
    FeatureEnum.EARLIEST_START_DATE: True,  #
    FeatureEnum.EARLIEST_FINISH_DATE: True,  #
    FeatureEnum.LATEST_START_DATE: True,  #
    FeatureEnum.LATEST_FINISH_DATE: True,
}  #


class EvaluationGPHH(Enum):
    SGS = 0
    PERMUTATION_DISTANCE = 1
    # SGS_DEVIATION = 2


class PermutationDistance(Enum):
    KTD = 0
    HAMMING = 1
    KTD_HAMMING = 2


class ParametersGPHH(Hyperparametrizable):

    hyperparameters = [
        FloatHyperparameter(name="tournament_ratio"),
        IntegerHyperparameter(name="pop_size", low=1, high=100),
        IntegerHyperparameter(name="min_tree_depth", low=1, high=4),
        IntegerHyperparameter(name="max_tree_depth", low=1, high=20),
        FloatHyperparameter(name="crossover_rate", low=0.0, high=1.0),
        FloatHyperparameter(name="mutation_rate", low=0.0, high=1.0),
        EnumHyperparameter(name="evaluation", enum=EvaluationGPHH),
        EnumHyperparameter(
            name="permutation_distance",
            enum=PermutationDistance,
        ),
    ] + PolicyMethodParams.hyperparameters

    set_feature: Set[FeatureEnum] = None
    set_primitves: PrimitiveSet = None
    tournament_ratio: float = None
    pop_size: int = None
    n_gen: int = None
    min_tree_depth: int = None
    max_tree_depth: int = None
    crossover_rate: float = None
    mutation_rate: float = None
    base_policy_method = None
    delta_index_freedom: int = None
    delta_time_freedom: int = None
    deap_verbose: bool = None
    evaluation: EvaluationGPHH = None
    permutation_distance = PermutationDistance.KTD

    def __init__(
        self,
        set_feature,
        set_primitves,
        tournament_ratio,
        pop_size,
        n_gen,
        min_tree_depth,
        max_tree_depth,
        crossover_rate,
        mutation_rate,
        base_policy_method,
        delta_index_freedom,
        delta_time_freedom,
        deap_verbose,
        evaluation,
        permutation_distance,
    ):
        self.set_feature = set_feature
        self.set_primitves = set_primitves
        self.tournament_ratio = tournament_ratio
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.min_tree_depth = min_tree_depth
        self.max_tree_depth = max_tree_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.base_policy_method = base_policy_method
        self.delta_index_freedom = delta_index_freedom
        self.delta_time_freedom = delta_time_freedom
        self.deap_verbose = deap_verbose
        self.evaluation = evaluation
        self.permutation_distance = permutation_distance

    @staticmethod
    def default():
        set_feature = {
            FeatureEnum.EARLIEST_FINISH_DATE,
            FeatureEnum.EARLIEST_START_DATE,
            FeatureEnum.LATEST_FINISH_DATE,
            FeatureEnum.LATEST_START_DATE,
            FeatureEnum.N_PREDECESSORS,
            FeatureEnum.N_SUCCESSORS,
            FeatureEnum.ALL_DESCENDANTS,
            FeatureEnum.RESSOURCE_REQUIRED,
            FeatureEnum.RESSOURCE_AVG,
            FeatureEnum.RESSOURCE_MAX,
            # FeatureEnum.RESSOURCE_MIN
            FeatureEnum.RESSOURCE_NZ_MIN,
        }

        pset = PrimitiveSet("main", len(set_feature))
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protected_div, 2)
        pset.addPrimitive(max_operator, 2)
        pset.addPrimitive(min_operator, 2)
        pset.addPrimitive(operator.neg, 1)

        return ParametersGPHH(
            set_feature=set_feature,
            set_primitves=pset,
            tournament_ratio=0.1,
            pop_size=40,
            n_gen=100,
            min_tree_depth=1,
            max_tree_depth=4,
            crossover_rate=0.7,
            mutation_rate=0.3,
            base_policy_method=BasePolicyMethod.FOLLOW_GANTT,
            delta_index_freedom=0,
            delta_time_freedom=0,
            deap_verbose=True,
            # evaluation=EvaluationGPHH.PERMUTATION_DISTANCE,
            evaluation=EvaluationGPHH.SGS,
            permutation_distance=PermutationDistance.KTD,
        )

    @staticmethod
    def fast_test():
        set_feature = {
            FeatureEnum.EARLIEST_FINISH_DATE,
            FeatureEnum.EARLIEST_START_DATE,
            FeatureEnum.LATEST_FINISH_DATE,
            FeatureEnum.LATEST_START_DATE,
            FeatureEnum.N_PREDECESSORS,
            FeatureEnum.N_SUCCESSORS,
            FeatureEnum.ALL_DESCENDANTS,
            FeatureEnum.RESSOURCE_REQUIRED,
            FeatureEnum.RESSOURCE_AVG,
            FeatureEnum.RESSOURCE_MAX,
            # FeatureEnum.RESSOURCE_MIN
            FeatureEnum.RESSOURCE_NZ_MIN,
        }

        pset = PrimitiveSet("main", len(set_feature))
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protected_div, 2)
        pset.addPrimitive(max_operator, 2)
        pset.addPrimitive(min_operator, 2)
        pset.addPrimitive(operator.neg, 1)

        return ParametersGPHH(
            set_feature=set_feature,
            set_primitves=pset,
            tournament_ratio=0.1,
            pop_size=10,
            n_gen=2,
            min_tree_depth=1,
            max_tree_depth=4,
            crossover_rate=0.7,
            mutation_rate=0.3,
            base_policy_method=BasePolicyMethod.FOLLOW_GANTT,
            delta_index_freedom=0,
            delta_time_freedom=0,
            deap_verbose=True,
            evaluation=EvaluationGPHH.SGS,
            # evaluation=EvaluationGPHH.PERMUTATION_DISTANCE,
            permutation_distance=PermutationDistance.KTD,
        )

    @staticmethod
    def default_for_set_features(set_feature: Set[FeatureEnum]):
        pset = PrimitiveSet("main", len(set_feature))
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        # pset.addPrimitive(protected_div, 2)
        pset.addPrimitive(max_operator, 2)
        pset.addPrimitive(min_operator, 2)
        pset.addPrimitive(operator.neg, 1)

        return ParametersGPHH(
            set_feature=set_feature,
            set_primitves=pset,
            tournament_ratio=0.25,
            pop_size=20,
            n_gen=20,
            min_tree_depth=1,
            max_tree_depth=4,
            crossover_rate=0.7,
            mutation_rate=0.1,
            base_policy_method=BasePolicyMethod.SGS_READY,
            delta_index_freedom=0,
            delta_time_freedom=0,
            deap_verbose=True,
            evaluation=EvaluationGPHH.PERMUTATION_DISTANCE,
            permutation_distance=PermutationDistance.KTD,
        )


class GPHH(Solver, DeterministicPolicies):
    T_domain = D

    hyperparameters = [
        SubBrickKwargsHyperparameter(
            name="params_gphh_kwargs", subbrick_cls=ParametersGPHH
        )
    ]

    training_domains: List[T_domain]
    verbose: bool
    weight: int
    pset: PrimitiveSet
    toolbox: Toolbox
    policy: DeterministicPolicies
    params_gphh: ParametersGPHH
    # policy: GPHHPolicy
    evaluation_method: EvaluationGPHH
    reference_permutations: Dict
    permutation_distance: PermutationDistance

    def __init__(
        self,
        domain_factory: Callable[[], SchedulingDomain],
        training_domains: List[T_domain],
        domain_model: SchedulingDomain,
        weight: int,
        # set_feature: Set[FeatureEnum]=None,
        params_gphh: Optional[ParametersGPHH] = None,
        reference_permutations=None,
        # reference_makespans=None,
        training_domains_names=None,
        verbose: bool = False,
        params_gphh_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Genetic Programming based Hyper-Heuristics

        # Parameters
        domain_factory
        training_domains
        domain_model
        weight
        params_gphh
        reference_permutations
        training_domains_names
        verbose
        params_gphh_kwargs: can be used as kwargs to create params_gphh if not specified.
            Useful when using optuna to tune GPHH's hyperparameters. Normal users should
            rather use directly params_gphh.

        """
        Solver.__init__(self, domain_factory=domain_factory)
        self.training_domains = training_domains
        self.domain_model = domain_model
        # default params_gphh
        if params_gphh is None:
            self.params_gphh = ParametersGPHH.default()
        else:
            self.params_gphh = params_gphh
        # update params_gphh (for optuna tuning)
        if params_gphh_kwargs is not None:
            for k, v in params_gphh_kwargs.items():
                setattr(self.params_gphh, k, v)
        self.set_feature = self.params_gphh.set_feature
        print("self.set_feature: ", self.set_feature)
        print("Evaluation: ", self.params_gphh.evaluation)
        # if set_feature is None:
        #     self.set_feature = {FeatureEnum.RESSOURCE_TOTAL,
        #                         FeatureEnum.TASK_DURATION,
        #                         FeatureEnum.N_SUCCESSORS,
        #                         FeatureEnum.N_SUCCESSORS,
        #                         FeatureEnum.RESSOURCE_AVG}
        self.list_feature = list(self.set_feature)
        self.verbose = verbose
        self.pset = self.init_primitives(self.params_gphh.set_primitves)
        self.weight = weight
        self.evaluation_method = self.params_gphh.evaluation
        self.initialize_cpm_data_for_training()
        if self.evaluation_method == EvaluationGPHH.PERMUTATION_DISTANCE:
            self.init_reference_permutations(
                reference_permutations, training_domains_names
            )
            self.permutation_distance = self.params_gphh.permutation_distance
        # if self.evaluation_method == EvaluationGPHH.SGS_DEVIATION:
        #     self.init_reference_makespans(reference_makespans, training_domains_names)

    def init_reference_permutations(
        self, reference_permutations={}, training_domains_names=[]
    ) -> None:
        self.reference_permutations = {}
        for i in range(len(self.training_domains)):
            td = self.training_domains[i]
            td_name = training_domains_names[i]
            if td_name not in reference_permutations.keys():
                # Run CP
                td.set_inplace_environment(False)
                solver = DOSolver(
                    domain_factory=lambda: td,
                    policy_method_params=PolicyMethodParams(
                        base_policy_method=BasePolicyMethod.SGS_PRECEDENCE,
                        delta_index_freedom=0,
                        delta_time_freedom=0,
                    ),
                    method=SolvingMethod.CP,
                )
                solver.solve()
                raw_permutation = solver.best_solution.rcpsp_permutation
                full_permutation = [x + 2 for x in raw_permutation]
                full_permutation.insert(0, 1)
                full_permutation.append(np.max(full_permutation) + 1)
                print("full_perm: ", full_permutation)
                self.reference_permutations[td] = full_permutation
            else:
                self.reference_permutations[td] = reference_permutations[td_name]

    def _solve(self) -> None:
        self.domain = self._domain_factory()

        tournament_ratio = self.params_gphh.tournament_ratio
        pop_size = self.params_gphh.pop_size
        n_gen = self.params_gphh.n_gen
        min_tree_depth = self.params_gphh.min_tree_depth
        max_tree_depth = self.params_gphh.max_tree_depth
        crossover_rate = self.params_gphh.crossover_rate
        mutation_rate = self.params_gphh.mutation_rate

        creator.create("FitnessMin", Fitness, weights=(self.weight,))
        creator.create("Individual", PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = Toolbox()
        self.toolbox.register(
            "expr",
            genHalfAndHalf,
            pset=self.pset,
            min_=min_tree_depth,
            max_=max_tree_depth,
        )
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        if self.evaluation_method == EvaluationGPHH.SGS:
            self.toolbox.register(
                "evaluate", self.evaluate_heuristic, domains=self.training_domains
            )
        # if self.evaluation_method == EvaluationGPHH.SGS_DEVIATION:
        #     self.toolbox.register("evaluate", self.evaluate_heuristic_sgs_deviation, domains=self.training_domains)
        elif self.evaluation_method == EvaluationGPHH.PERMUTATION_DISTANCE:
            self.toolbox.register(
                "evaluate",
                self.evaluate_heuristic_permutation,
                domains=self.training_domains,
            )
        # self.toolbox.register("evaluate", self.evaluate_heuristic, domains=[self.training_domains[1]])

        self.toolbox.register(
            "select", tools.selTournament, tournsize=int(tournament_ratio * pop_size)
        )
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=max_tree_depth)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset
        )

        self.toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )
        self.toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        self.hof = hof
        pop, log = algorithms.eaSimple(
            pop,
            self.toolbox,
            crossover_rate,
            mutation_rate,
            n_gen,
            stats=mstats,
            halloffame=hof,
            verbose=True,
        )

        self.best_heuristic = hof[0]
        print("best_heuristic: ", self.best_heuristic)

        self.func_heuristic = self.toolbox.compile(expr=self.best_heuristic)
        self.policy = GPHHPolicy(
            self.domain,
            self.domain_model,
            self.func_heuristic,
            features=self.list_feature,
            params_gphh=self.params_gphh,
            recompute_cpm=True,
        )

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        action = self.policy.sample_action(observation)
        # print('action_1: ', action.action)
        return action

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def init_primitives(self, pset) -> PrimitiveSet:
        for i in range(len(self.list_feature)):
            pset.renameArguments(**{"ARG" + str(i): self.list_feature[i].value})
        return pset

    def evaluate_heuristic(self, individual, domains) -> float:
        vals = []
        func_heuristic = self.toolbox.compile(expr=individual)
        # print('individual', individual)
        for domain in domains:

            ###
            initial_state = domain.get_initial_state()

            do_model = build_do_domain(domain)
            modes = [
                initial_state.tasks_mode.get(j, 1)
                for j in sorted(domain.get_tasks_ids())
            ]
            modes = modes[1:-1]

            cpm = self.cpm_data[domain]["cpm"]
            cpm_esd = self.cpm_data[domain]["cpm_esd"]

            raw_values = []
            for task_id in domain.get_available_tasks(initial_state):
                input_features = [
                    feature_function_map[lf](
                        domain=domain,
                        cpm=cpm,
                        cpm_esd=cpm_esd,
                        task_id=task_id,
                        state=initial_state,
                    )
                    for lf in self.list_feature
                ]
                output_value = func_heuristic(*input_features)
                raw_values.append(output_value)

            normalized_values = [
                x + 1
                for x in sorted(
                    range(len(raw_values)), key=lambda k: raw_values[k], reverse=False
                )
            ]
            normalized_values_for_do = [
                normalized_values[i] - 2
                for i in range(len(normalized_values))
                if normalized_values[i] not in {1, len(normalized_values)}
            ]

            solution = RCPSPSolution(
                problem=do_model,
                rcpsp_permutation=normalized_values_for_do,
                rcpsp_modes=modes,
            )
            last_activity = max(list(solution.rcpsp_schedule.keys()))
            do_makespan = solution.rcpsp_schedule[last_activity]["end_time"]
            vals.append(do_makespan)

        fitness = [np.mean(vals)]
        return fitness

    def initialize_cpm_data_for_training(self):
        self.cpm_data = {}
        for domain in self.training_domains:
            do_model = build_do_domain(domain)
            cpm, cpm_esd = compute_cpm(do_model)
            self.cpm_data[domain] = {"cpm": cpm, "cpm_esd": cpm_esd}

    def evaluate_heuristic_permutation(self, individual, domains) -> float:
        vals = []
        func_heuristic = self.toolbox.compile(expr=individual)
        # print('individual', individual)

        for domain in domains:

            raw_values = []
            initial_state = domain.get_initial_state()

            regenerate_cpm = False
            if regenerate_cpm:
                do_model = build_do_domain(domain)
                cpm, cpm_esd = compute_cpm(do_model)
            else:
                cpm = self.cpm_data[domain]["cpm"]
                cpm_esd = self.cpm_data[domain]["cpm_esd"]

            for task_id in domain.get_available_tasks(state=initial_state):

                input_features = [
                    feature_function_map[lf](
                        domain=domain,
                        cpm=cpm,
                        cpm_esd=cpm_esd,
                        task_id=task_id,
                        state=initial_state,
                    )
                    for lf in self.list_feature
                ]
                output_value = func_heuristic(*input_features)
                raw_values.append(output_value)

            most_common_raw_val = max(raw_values, key=raw_values.count)
            most_common_count = raw_values.count(most_common_raw_val)

            heuristic_permutation = [
                x + 1
                for x in sorted(
                    range(len(raw_values)), key=lambda k: raw_values[k], reverse=False
                )
            ]

            if self.permutation_distance == PermutationDistance.KTD:
                dist, p_value = stats.kendalltau(
                    heuristic_permutation, self.reference_permutations[domain]
                )
                dist = -dist

            if self.permutation_distance == PermutationDistance.HAMMING:
                dist = distance.hamming(
                    heuristic_permutation, self.reference_permutations[domain]
                )

            if self.permutation_distance == PermutationDistance.KTD_HAMMING:
                ktd, _ = stats.kendalltau(
                    heuristic_permutation, self.reference_permutations[domain]
                )
                dist = -ktd + distance.hamming(
                    heuristic_permutation, self.reference_permutations[domain]
                )

            penalty = most_common_count / len(raw_values)
            # penalty = 0.
            penalized_distance = dist + penalty
            vals.append(penalized_distance)
        fitness = [np.mean(vals)]
        # fitness = [np.max(vals)]
        return fitness

    def test_features(self, domain, task_id, observation):
        for f in FeatureEnum:
            print("feature: ", f)
            calculated_feature = feature_function_map[f](
                domain=domain, task_id=task_id, state=observation
            )
            print("\tcalculated_feature: ", calculated_feature)

    def set_domain(self, domain):
        self.domain = domain
        if self.policy is not None:
            self.policy.domain = domain


class GPHHPolicy(DeterministicPolicies):
    def __init__(
        self,
        domain: SchedulingDomain,
        domain_model: SchedulingDomain,
        func_heuristic,
        features: List[FeatureEnum] = None,
        params_gphh=None,
        recompute_cpm=True,
        cpm_data=None,
    ):
        self.domain = domain
        self.domain_model = domain_model
        self.func_heuristic = func_heuristic
        self.list_feature = features
        self.params_gphh = params_gphh
        self.recompute_cpm = recompute_cpm
        self.cpm_data = cpm_data

    def reset(self):
        pass

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        run_sgs = True
        cheat_mode = False

        do_model = build_do_domain(self.domain_model)
        modes = [
            observation.tasks_mode.get(j, 1)
            for j in sorted(self.domain.get_tasks_ids())
        ]
        modes = modes[1:-1]
        scheduled_tasks_start_times = {}
        if run_sgs:
            tasks_details = rebuild_all_tasks_dict(observation)
            for j in tasks_details:
                if tasks_details[j].start is not None:
                    scheduled_tasks_start_times[j] = tasks_details[j].start
                    do_model.mode_details[j][1]["duration"] = tasks_details[
                        j
                    ].sampled_duration
            # needed to update the numba functions to take into account
            # the sampled durations of the current scheduling state.
            do_model.update_functions()

        # do_model = build_do_domain(self.domain)
        # modes = [observation.tasks_mode.get(j, 1) for j in sorted(self.domain.get_tasks_ids())]
        # modes = modes[1:-1]
        #
        # if run_sgs:
        #     scheduled_tasks_start_times = {}
        #     for j in observation.tasks_details.keys():
        #         if observation.tasks_details[j].start is not None:
        #             scheduled_tasks_start_times[j] = observation.tasks_details[j].start
        #         else:
        #             if not cheat_mode:
        #                 do_model.mode_details[j][1]['duration'] = self.domain_model.sample_task_duration(j, 1, 0.)

        if self.recompute_cpm:
            cpm, cpm_esd = compute_cpm(do_model)
        else:
            cpm = self.cpm_data[self.domain]["cpm"]
            cpm_esd = self.cpm_data[self.domain]["cpm_esd"]

        t = observation.t
        raw_values = []
        for task_id in self.domain.get_available_tasks(observation):
            input_features = [
                feature_function_map[lf](
                    domain=self.domain,
                    cpm=cpm,
                    cpm_esd=cpm_esd,
                    task_id=task_id,
                    state=observation,
                )
                for lf in self.list_feature
            ]
            output_value = self.func_heuristic(*input_features)
            raw_values.append(output_value)

        normalized_values = [
            x + 1
            for x in sorted(
                range(len(raw_values)), key=lambda k: raw_values[k], reverse=False
            )
        ]
        normalized_values_for_do = [
            normalized_values[i] - 2
            for i in range(len(normalized_values))
            if normalized_values[i] not in {1, len(normalized_values)}
        ]

        # print(t, ': ', normalized_values)
        # print('normalized_values_for_do: ', normalized_values_for_do)

        modes_dictionnary = {}
        for i in range(len(normalized_values)):
            modes_dictionnary[i + 1] = 1

        if run_sgs:
            tasks_complete_dict = rebuild_tasks_complete_details_dict(observation)
            # Force the scheduler_tasks_start_times to be disjoint with the complete tasks,
            # this avoid a corner effect giving wrong results in the numba SGS implementation
            # (sgs_fast_partial_schedule_incomplete_permutation_tasks)
            scheduled_tasks_start_times = {
                j: scheduled_tasks_start_times[j]
                for j in scheduled_tasks_start_times
                if j not in tasks_complete_dict
            }
            solution = RCPSPSolution(
                problem=do_model,
                rcpsp_permutation=normalized_values_for_do,
                rcpsp_modes=modes,
            )
            solution.generate_schedule_from_permutation_serial_sgs_2(
                current_t=t,
                completed_tasks=tasks_complete_dict,
                scheduled_tasks_start_times=scheduled_tasks_start_times,
            )

            schedule = solution.rcpsp_schedule
        else:
            schedule = None

        sgs_policy = PolicyRCPSP(
            domain=self.domain,
            schedule=schedule,
            policy_method_params=PolicyMethodParams(
                # base_policy_method=BasePolicyMethod.SGS_PRECEDENCE,
                # base_policy_method=BasePolicyMethod.SGS_READY,
                base_policy_method=self.params_gphh.base_policy_method,
                delta_index_freedom=self.params_gphh.delta_index_freedom,
                delta_time_freedom=self.params_gphh.delta_time_freedom,
            ),
            permutation_task=normalized_values,
            modes_dictionnary=modes_dictionnary,
        )
        action: SchedulingAction = sgs_policy.sample_action(observation)
        # print('action_2: ', action.action)
        return action

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True


class PoolAggregationMethod(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    RANDOM = "random"


class PooledGPHHPolicy(DeterministicPolicies):
    def __init__(
        self,
        domain: SchedulingDomain,
        domain_model: SchedulingDomain,
        func_heuristics,
        pool_aggregation_method: PoolAggregationMethod = PoolAggregationMethod.MEAN,
        remove_extremes_values: int = 0,
        features: List[FeatureEnum] = None,
        params_gphh=None,
    ):
        self.domain = domain
        self.domain_model = domain_model
        self.func_heuristics = func_heuristics
        self.list_feature = features
        self.params_gphh = params_gphh
        self.pool_aggregation_method = pool_aggregation_method
        self.remove_extremes_values = remove_extremes_values

    def reset(self):
        pass

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:

        run_sgs = True
        cheat_mode = False
        regenerate_cpm = True

        do_model = build_do_domain(self.domain_model)
        modes = [
            observation.tasks_mode.get(j, 1)
            for j in sorted(self.domain.get_tasks_ids())
        ]
        modes = modes[1:-1]

        if run_sgs:
            scheduled_tasks_start_times = {}
            tasks_details = rebuild_all_tasks_dict(observation)
            for j in tasks_details:
                if tasks_details[j].start is not None:
                    scheduled_tasks_start_times[j] = tasks_details[j].start
                    do_model.mode_details[j][1]["duration"] = tasks_details[
                        j
                    ].sampled_duration

        # do_model = build_do_domain(self.domain)
        # modes = [observation.tasks_mode.get(j, 1) for j in sorted(self.domain.get_tasks_ids())]
        # modes = modes[1:-1]
        #
        # if run_sgs:
        #     scheduled_tasks_start_times = {}
        #     for j in observation.tasks_details.keys():
        #         # schedule[j] = {}
        #         if observation.tasks_details[j].start is not None:
        #             # schedule[j]["start_time"] = observation.tasks_details[j].start
        #             scheduled_tasks_start_times[j] = observation.tasks_details[j].start
        #         # if observation.tasks_details[j].end is not None:
        #         #     schedule[j]["end_time"] = observation.tasks_details[j].end
        #         else:
        #             if not cheat_mode:
        #                 do_model.mode_details[j][1]['duration'] = self.domain_model.sample_task_duration(j, 1, 0.)

        if regenerate_cpm:
            cpm, cpm_esd = compute_cpm(do_model)

        t = observation.t
        raw_values = []
        for task_id in self.domain.get_available_tasks(observation):
            input_features = [
                feature_function_map[lf](
                    domain=self.domain,
                    cpm=cpm,
                    cpm_esd=cpm_esd,
                    task_id=task_id,
                    state=observation,
                )
                for lf in self.list_feature
            ]
            output_values = []
            for f in self.func_heuristics:
                output_value = f(*input_features)
                output_values.append(output_value)

            # print('output_values: ', output_values)
            if self.remove_extremes_values > 0:
                the_median = float(np.median(output_values))
                tmp = {}
                for i in range(len(output_values)):
                    tmp[i] = abs(output_values[i] - the_median)
                tmp = sorted(tmp.items(), key=lambda x: x[1], reverse=True)
                to_remove = [tmp[i][0] for i in range(self.remove_extremes_values)]
                output_values = list(np.delete(output_values, to_remove))

            # print('output_values filtered: ', output_values)
            if self.pool_aggregation_method == PoolAggregationMethod.MEAN:
                agg_value = np.mean(output_values)
            elif self.pool_aggregation_method == PoolAggregationMethod.MEDIAN:
                agg_value = np.median(output_values)
            elif self.pool_aggregation_method == PoolAggregationMethod.RANDOM:
                index = random.randint(len(output_values))
                agg_value = output_values[index]

            # print('agg_value: ', agg_value)
            raw_values.append(agg_value)

        normalized_values = [
            x + 1
            for x in sorted(
                range(len(raw_values)), key=lambda k: raw_values[k], reverse=False
            )
        ]
        normalized_values_for_do = [
            normalized_values[i] - 2
            for i in range(len(normalized_values))
            if normalized_values[i] not in {1, len(normalized_values)}
        ]

        # print('normalized_values: ', normalized_values)
        # print('normalized_values_for_do: ', normalized_values_for_do)

        modes_dictionnary = {}
        for i in range(len(normalized_values)):
            modes_dictionnary[i + 1] = 1

        if run_sgs:

            solution = RCPSPSolution(
                problem=do_model,
                rcpsp_permutation=normalized_values_for_do,
                rcpsp_modes=modes,
            )
            tasks_complete_dict = rebuild_tasks_complete_details_dict(observation)
            solution.generate_schedule_from_permutation_serial_sgs_2(
                current_t=t,
                completed_tasks=tasks_complete_dict,
                scheduled_tasks_start_times=scheduled_tasks_start_times,
            )

            schedule = solution.rcpsp_schedule
        else:
            schedule = None

        sgs_policy = PolicyRCPSP(
            domain=self.domain,
            schedule=schedule,
            policy_method_params=PolicyMethodParams(
                # base_policy_method=BasePolicyMethod.SGS_PRECEDENCE,
                # base_policy_method=BasePolicyMethod.SGS_READY,
                base_policy_method=self.params_gphh.base_policy_method,
                delta_index_freedom=self.params_gphh.delta_index_freedom,
                delta_time_freedom=self.params_gphh.delta_time_freedom,
            ),
            permutation_task=normalized_values,
            modes_dictionnary=modes_dictionnary,
        )
        action: SchedulingAction = sgs_policy.sample_action(observation)
        # print('action_2: ', action.action)
        return action

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True


class FixedPermutationPolicy(DeterministicPolicies):
    def __init__(
        self, domain: SchedulingDomain, domain_model: SchedulingDomain, fixed_perm
    ):
        self.domain = domain
        self.domain_model = domain_model
        self.fixed_perm = fixed_perm

    def reset(self):
        pass

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        run_sgs = True
        cheat_mode = False

        do_model = build_do_domain(self.domain_model)
        modes = [
            observation.tasks_mode.get(j, 1)
            for j in sorted(self.domain.get_tasks_ids())
        ]
        modes = modes[1:-1]

        if run_sgs:
            scheduled_tasks_start_times = {}
            tasks_details = rebuild_all_tasks_dict(observation)
            for j in tasks_details.keys():
                if tasks_details[j].start is not None:
                    scheduled_tasks_start_times[j] = tasks_details[j].start
                    do_model.mode_details[j][1]["duration"] = tasks_details[
                        j
                    ].sampled_duration

        # do_model = build_do_domain(self.domain)
        # modes = [observation.tasks_mode.get(j, 1) for j in sorted(self.domain.get_tasks_ids())]
        # modes = modes[1:-1]
        #
        # if run_sgs:
        #     scheduled_tasks_start_times = {}
        #     for j in observation.tasks_details.keys():
        #         # schedule[j] = {}
        #         if observation.tasks_details[j].start is not None:
        #             # schedule[j]["start_time"] = observation.tasks_details[j].start
        #             scheduled_tasks_start_times[j] = observation.tasks_details[j].start
        #         # if observation.tasks_details[j].end is not None:
        #         #     schedule[j]["end_time"] = observation.tasks_details[j].end
        #         else:
        #             if not cheat_mode:
        #                 # print('do_model: ', do_model)
        #                 do_model.mode_details[j][1]['duration'] = self.domain_model.sample_task_duration(j, 1, 0.)

        normalized_values = self.fixed_perm

        normalized_values_for_do = [
            normalized_values[i] - 2
            for i in range(len(normalized_values))
            if normalized_values[i] not in {1, len(normalized_values)}
        ]

        # print('normalized_values: ', normalized_values)
        # print('normalized_values_for_do: ', normalized_values_for_do)
        t = observation.t

        modes_dictionnary = {}
        for i in range(len(normalized_values)):
            modes_dictionnary[i + 1] = 1

        if run_sgs:

            solution = RCPSPSolution(
                problem=do_model,
                rcpsp_permutation=normalized_values_for_do,
                rcpsp_modes=modes,
            )
            tasks_details_complete = rebuild_tasks_complete_details_dict(observation)
            solution.generate_schedule_from_permutation_serial_sgs_2(
                current_t=t,
                completed_tasks=tasks_details_complete,
                scheduled_tasks_start_times=scheduled_tasks_start_times,
            )

            schedule = solution.rcpsp_schedule
        else:
            schedule = None

        sgs_policy = PolicyRCPSP(
            domain=self.domain,
            schedule=schedule,
            policy_method_params=PolicyMethodParams(
                # base_policy_method=BasePolicyMethod.SGS_PRECEDENCE,
                # base_policy_method=BasePolicyMethod.SGS_READY,
                base_policy_method=BasePolicyMethod.FOLLOW_GANTT,
                # delta_index_freedom=self.params_gphh.delta_index_freedom,
                # delta_time_freedom=self.params_gphh.delta_time_freedom
            ),
            permutation_task=normalized_values,
            modes_dictionnary=modes_dictionnary,
        )
        action: SchedulingAction = sgs_policy.sample_action(observation)
        # print('action_2: ', action.action)
        return action

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True
