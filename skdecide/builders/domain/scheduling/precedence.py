# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict, List

from skdecide.builders.domain.scheduling.graph_toolbox import Graph
from skdecide.builders.domain.scheduling.scheduling_domains_modelling import State

__all__ = ["WithPrecedence", "WithoutPrecedence"]


class WithPrecedence:
    """A domain must inherit this class if there exist some predecence constraints between tasks."""

    def _get_successors(self) -> Dict[int, List[int]]:
        """Return the successors of the tasks. Successors are given as a list for a task given as a key."""
        raise NotImplementedError

    def get_successors(self) -> Dict[int, List[int]]:
        """Return the successors of the tasks. Successors are given as a list for a task given as a key."""
        return self._get_successors()

    def _get_successors_task(self, task_id: int) -> List[int]:
        return self.get_successors()[task_id]

    def get_successors_task(self, task_id: int) -> List[int]:
        return self._get_successors_task(task_id=task_id)

    def _get_predecessors(self) -> Dict[int, List[int]]:
        """Return the predecessors of the task. Successors are given as a list for a task given as a key."""
        return self.graph.predecessors_map()

    def get_predecessors(self) -> Dict[int, List[int]]:
        """Return the predecessors of the task. Successors are given as a list for a task given as a key."""
        return self._get_predecessors()

    def _get_predecessors_task(self, task_id: int) -> List[int]:
        return self.get_predecessors()[task_id]

    def get_predecessors_task(self, task_id: int) -> List[int]:
        return self._get_predecessors_task(task_id=task_id)

    def compute_graph(self):
        task_ids = self.get_tasks_ids()
        successors = self.get_successors()
        mode_details = self.get_tasks_modes()
        nodes = [
            (
                n,
                {
                    mode: self.sample_task_duration(task=n, mode=mode)
                    for mode in mode_details[n]
                },
            )
            for n in task_ids
        ]
        edges = []
        for n in successors:
            for succ in successors[n]:
                edges += [(n, succ, {})]
        return Graph(nodes, edges, False)

    def _task_modes_possible_to_launch(self, state: State):
        mode_details = self.get_tasks_modes()
        return [
            (n, mode)
            for n in state.tasks_remaining
            for mode in mode_details[n]
            if all(m in state.tasks_complete for m in self.ancestors[n])
        ]

    def task_modes_possible_to_launch(self, state: State):
        return self._task_modes_possible_to_launch(state=state)

    def _task_possible_to_launch_precedence(self, state: State):
        return [
            n
            for n in state.tasks_remaining
            if all(m in state.tasks_complete for m in self.ancestors[n])
        ]

    def task_possible_to_launch_precedence(self, state: State):
        return self._task_possible_to_launch_precedence(state=state)


class WithoutPrecedence(WithPrecedence):
    """A domain must inherit this class if there are no predecence constraints between tasks."""

    def _get_successors(self) -> Dict[int, List[int]]:
        """Return the successors of the tasks. Successors are given as a list for a task given as a key."""
        ids = self.get_tasks_ids()
        succ = {}
        for id in ids:
            succ[id] = []
        return succ

    def _get_predecessors(self) -> Dict[int, List[int]]:
        """Return the successors of the tasks. Successors are given as a list for a task given as a key."""
        ids = self.get_tasks_ids()
        prec = {}
        for id in ids:
            prec[id] = []
        return prec
