# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Plot utilities outside of the main library, needs additional dependencies like seaborn.

from __future__ import annotations

import math
from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # You'd need seaborn for this example.
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ParetoFront,
    ResultStorage,
    result_storage_to_pareto_front,
)
from discrete_optimization.generic_tools.result_storage.resultcomparator import (
    ResultComparator,
)


def plot_storage_2d(
    result_storage: ResultStorage, name_axis: List[str], ax=None, color="r"
):
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.scatter(
        x=[p[1].vector_fitness[0] for p in result_storage.list_solution_fits],
        y=[p[1].vector_fitness[1] for p in result_storage.list_solution_fits],
        color=color,
    )
    ax.set_xlabel(name_axis[0])
    ax.set_ylabel(name_axis[1])


def plot_pareto_2d(pareto_front: ParetoFront, name_axis: List[str], ax=None, color="b"):
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.scatter(
        x=[p[1].vector_fitness[0] for p in pareto_front.paretos],
        y=[p[1].vector_fitness[1] for p in pareto_front.paretos],
        color=color,
    )
    ax.set_xlabel(name_axis[0])
    ax.set_ylabel(name_axis[1])


class ResultComparatorPlot(ResultComparator):
    def plot_distribution_for_objective(self, objective_str: str):
        obj_index = self.objectives_str.index(objective_str)
        fig, ax = plt.subplots(1, figsize=(10, 10))
        for i in range(len(self.result_storage_names)):
            sns.distplot(
                self.reevaluated_results[i][objective_str],
                rug=True,
                bins=max(1, len(self.reevaluated_results[i][objective_str]) // 10),
                label=self.result_storage_names[i],
                ax=ax,
            )
        ax.legend()
        ax.set_title(
            objective_str.upper()
            + " distribution over test instances, for different optimisation approaches"
        )
        return fig

    def generate_super_pareto(self):
        sols = []
        for rs in self.list_result_storage:
            for s in rs.list_solution_fits:
                sols.append(s)
        rs = ResultStorage(list_solution_fits=sols, best_solution=None)
        pareto_store = result_storage_to_pareto_front(result_storage=rs, problem=None)
        return pareto_store

    def plot_all_2d_paretos_single_plot(self, objectives_str=None):

        if objectives_str is None:
            objecives_names = self.objectives_str[:2]
            objectives_index = [0, 1]
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
            ax.scatter(
                x=[
                    p[1].vector_fitness[objectives_index[0]]
                    for p in self.list_result_storage[i].list_solution_fits
                ],
                y=[
                    p[1].vector_fitness[objectives_index[1]]
                    for p in self.list_result_storage[i].list_solution_fits
                ],
                color=colors[i],
            )
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
        rows = math.ceil(len(self.list_result_storage) / cols)
        fig, axs = plt.subplots(rows, cols)
        axis = axs.flatten()
        colors = cm.rainbow(np.linspace(0, 1, len(self.list_result_storage)))
        print(axs.shape)
        for i, ax in zip(
            range(len(self.list_result_storage)), axis[: len(self.list_result_storage)]
        ):
            x = [
                p[1].vector_fitness[objectives_index[0]]
                for p in self.list_result_storage[i].list_solution_fits
            ]
            y = [
                p[1].vector_fitness[objectives_index[1]]
                for p in self.list_result_storage[i].list_solution_fits
            ]
            ax.scatter(x=x, y=y, color=colors[i])
            ax.set_title(self.result_storage_names[i])
        fig.tight_layout(pad=3.0)

        return fig

    def plot_super_pareto(self):
        super_pareto = self.generate_super_pareto()
        plot_pareto_2d(pareto_front=super_pareto, name_axis=self.objectives_str)
        # TODO: This one is not working ! Need to check why
        plt.title("Pareto front obtained by merging solutions from all result stores")

    def plot_all_best_by_objective(self, objectif_str):
        obj_index = self.objectives_str.index(objectif_str)
        data = self.get_best_by_objective_by_result_storage(objectif_str)
        x = list(data.keys())
        y = [data[key][1].vector_fitness[obj_index] for key in x]
        y_pos = np.arange(len(x))

        plt.bar(y_pos, y)
        plt.xticks(y_pos, x, rotation=45)
        plt.title("Comparison on " + objectif_str)
        # plt.show()
