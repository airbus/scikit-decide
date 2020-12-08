from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage, result_storage_to_pareto_front
from typing import List, Tuple, Dict, Union, Set, Any, Optional
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Problem
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ParetoFront, plot_pareto_2d, plot_storage_2d

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

import numpy as np
import math

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

    def plot_distribution_for_objective(self, objective_str: str):
        obj_index = self.objectives_str.index(objective_str)
        fig, ax = plt.subplots(1, figsize=(10, 10))
        for i in range(len(self.result_storage_names)):
            sns.distplot(self.reevaluated_results[i][objective_str],
                         rug=True,
                         bins=max(1, len(self.reevaluated_results[i][objective_str]) // 10),
                         label=self.result_storage_names[i],
                         ax=ax)
        ax.legend()
        ax.set_title(objective_str.upper()+" distribution over test instances, for different optimisation approaches")
        return fig

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
        # print('len(rs): ', len(rs.list_solution_fits))
        pareto_store = result_storage_to_pareto_front(result_storage=rs, problem=None)
        # print('len(pareto_store): ', len(pareto_store.list_solution_fits))
        # print('hhhh: ', [x[1].vector_fitness for x in pareto_store.list_solution_fits])
        return pareto_store

    def plot_all_2d_paretos_single_plot(self,
                                        objectives_str=None):

        if objectives_str is None:
            objecives_names = self.objectives_str[:2]
            objectives_index = [0,1]
        else:
            objecives_names = objectives_str
            objectives_index = []
            for obj in objectives_str:
                obj_index = self.objectives_str.index(obj)
                objectives_index.append(obj_index)

        colors = cm.rainbow(np.linspace(0, 1, len(self.list_result_storage)))
        fig, ax = plt.subplots(1)
        ax.set_xlabel(objecives_names[0])
        ax.set_ylabel(objecives_names[1])

        for i in range(len(self.list_result_storage)):
            ax.scatter(x=[p[1].vector_fitness[objectives_index[0]]
                          for p in self.list_result_storage[i].list_solution_fits],
                       y=[p[1].vector_fitness[objectives_index[1]]
                          for p in self.list_result_storage[i].list_solution_fits],
                       color=colors[i])
        ax.legend(self.result_storage_names)
        return ax

    def plot_all_2d_paretos_subplots(self, objectives_str=None):

        if objectives_str is None:
            objecives_names = self.objectives_str[:2]
            objectives_index = [0, 1]
        else:
            objecives_names = objectives_str
            objectives_index = []
            for obj in objectives_str:
                obj_index = self.objectives_str.index(obj)
                objectives_index.append(obj_index)

        cols = 2
        rows = math.ceil(len(self.list_result_storage) / cols)  # I have to do this to ensure at least 2 rows or else it creates axs with only 1 diumension and it crashes
        fig, axs = plt.subplots(rows, cols)
        axis = axs.flatten()
        colors = cm.rainbow(np.linspace(0, 1, len(self.list_result_storage)))
        print(axs.shape)
        for i, ax in zip(range(len(self.list_result_storage)), axis[:len(self.list_result_storage)]):
            x = [p[1].vector_fitness[objectives_index[0]]
                 for p in self.list_result_storage[i].list_solution_fits]
            y = [p[1].vector_fitness[objectives_index[1]]
                 for p in self.list_result_storage[i].list_solution_fits]
            ax.scatter(x=x,
                       y=y,
                       color=colors[i])
            ax.set_title(self.result_storage_names[i])
        fig.tight_layout(pad=3.0)

        return fig

    def plot_super_pareto(self):
        super_pareto = self.generate_super_pareto()
        # plot_storage_2d(result_storage=super_pareto, name_axis=self.objectives_str)
        plot_pareto_2d(pareto_front=super_pareto, name_axis=self.objectives_str)
        # TODO: This one is not working ! Need to check why
        plt.title('Pareto front obtained by merging solutions from all result stores')

    def plot_all_best_by_objective(self, objectif_str):
        obj_index = self.objectives_str.index(objectif_str)
        data = self.get_best_by_objective_by_result_storage(objectif_str)
        x = list(data.keys())
        y = [data[key][1].vector_fitness[obj_index] for key in x]
        # print('x: ', x)
        # print('y: ', y)
        y_pos = np.arange(len(x))

        plt.bar(y_pos, y)
        plt.xticks(y_pos, x, rotation=45)
        plt.title('Comparison on ' + objectif_str)
        # plt.show()


