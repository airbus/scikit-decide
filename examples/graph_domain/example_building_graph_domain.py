# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Example script to show how to build a GraphDomain from an example of deterministic planning domain.
A* is then run on both domain to check the computation time impact.
"""
import time

from skdecide.hub.domain.graph_domain.graph_domain_builders import (
    DFSExploration,
    FullSpaceExploration,
)
from skdecide.hub.domain.maze import Maze
from skdecide.hub.solver.lazy_astar.lazy_astar import LazyAstar


def example_maze():
    domain_maze = Maze()
    full_space_explorator = FullSpaceExploration(domain=domain_maze)
    graph = full_space_explorator.build_graph_domain(domain_maze.get_initial_state())
    print(graph.next_state_map)


def solve_with_astar_full_space_explo():
    domain_maze = Maze()
    full_space_explorator = FullSpaceExploration(domain=domain_maze)
    graph_maze = full_space_explorator.build_graph_domain(
        domain_maze.get_initial_state()
    )
    t = time.perf_counter()
    solver = LazyAstar(domain_factory=lambda: domain_maze)
    solver.solve()
    print(solver.get_plan())
    t1 = time.perf_counter()
    print(t1 - t, " sec to solve original maze")
    t = time.perf_counter()
    solver = LazyAstar(domain_factory=lambda: graph_maze)
    solver.solve(from_memory=domain_maze.get_initial_state())
    print(solver.get_plan())
    t2 = time.perf_counter()
    print(t2 - t, " sec to solve graph maze")


def solve_with_astar_dfs():
    domain_maze = Maze()
    dfs_explorator = DFSExploration(domain=domain_maze)
    graph_maze = dfs_explorator.build_graph_domain(domain_maze.get_initial_state())
    t = time.perf_counter()
    solver = LazyAstar(domain_factory=lambda: domain_maze)
    solver.solve()
    print(solver.get_plan())
    t1 = time.perf_counter()
    print(t1 - t, " sec to solve original maze")
    t = time.perf_counter()
    solver = LazyAstar(domain_factory=lambda: graph_maze)
    solver.solve(from_memory=domain_maze.get_initial_state())
    print(solver.get_plan())
    t2 = time.perf_counter()
    print(t2 - t, " sec to solve graph maze")
