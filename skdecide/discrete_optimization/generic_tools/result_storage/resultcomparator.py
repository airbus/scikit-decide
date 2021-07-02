# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from skdecide.discrete_optimization.generic_tools.result_storage import ResultStorage, result_storage_to_pareto_front
from typing import List
from skdecide.discrete_optimization.generic_tools.do_problem import Problem


class ResultComparator:
    list_result_storage: List[ResultStorage]
    result_storage_names: List[str]
    objectives_str: List[str]
    objective_weights: List[int]
    test_problems: List[Problem]
    super_pareto: ResultStorage

    # If test problem is None, then we use the fitnesses from the ResultStorage
    def __init__(self, list_result_storage: List[ResultStorage],
                 result_storage_names: List[str],
                 objectives_str: List[str],
                 objective_weights: List[int],
                 test_problems=None):
        self.list_result_storage = list_result_storage
        self.result_storage_names = result_storage_names
        self.objectives_str = objectives_str
        self.objective_weights = objective_weights
        self.test_problems = test_problems
        self.reevaluated_results = {}

        if self.test_problems is not None:
            self.reevaluate_result_storages()

    def reevaluate_result_storages(self):
        for res in self.list_result_storage:
            self.reevaluated_results[self.list_result_storage.index(res)] = {}
            for obj in self.objectives_str:
                self.reevaluated_results[self.list_result_storage.index(res)][obj] = []
                for scenario in self.test_problems:
                    # res.list_solution_fits[0][0].change_problem(scenario)
                    # val = scenario.evaluate(res.list_solution_fits[0][0])[obj]
                    res.get_best_solution().change_problem(scenario)
                    val = scenario.evaluate(res.get_best_solution())[obj]
                    self.reevaluated_results[self.list_result_storage.index(res)][obj].append(val)
        print('reevaluated_results: ', self.reevaluated_results)

    def print_test_distribution(self):
        ...

    def get_best_by_objective_by_result_storage(self, objectif_str: str):
        obj_index = self.objectives_str.index(objectif_str)
        # print('obj_index: ', obj_index)
        val = {}
        for i in range(len(self.list_result_storage)):
            fit_array = [self.list_result_storage[i].list_solution_fits[j][1].vector_fitness[obj_index]
                         for j in range(len(self.list_result_storage[i].list_solution_fits))] # create fit array
            # self.objective_weights[obj_index] > 0:
            if self.list_result_storage[i].maximize:
                best_fit = max(fit_array)
            else:
                best_fit = min(fit_array)
            # best_fit = max(fit_array)

            best_index = fit_array.index(best_fit)
            best_sol = self.list_result_storage[i].list_solution_fits[best_index]
            # print('fit_array:', fit_array)
            # print('best_sol:', best_sol)
            val[self.result_storage_names[i]] = best_sol
        return val

    def generate_super_pareto(self):
        sols = []
        for rs in self.list_result_storage:
            for s in rs.list_solution_fits:
                sols.append(s)
        rs = ResultStorage(list_solution_fits=sols, best_solution=None)
        pareto_store = result_storage_to_pareto_front(result_storage=rs, problem=None)
        return pareto_store



