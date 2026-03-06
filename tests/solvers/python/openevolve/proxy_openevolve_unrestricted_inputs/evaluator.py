import concurrent.futures
import importlib
import os
import traceback
from typing import Any, Callable, Optional

from openevolve.evaluation_result import EvaluationResult

from skdecide import Solver, rollout
from skdecide.builders.solver import Policies
from skdecide.hub.domain.maze import Maze
from skdecide.hub.domain.maze.maze import Action

EVALUATION_TIMEOUT = 60

with open(f"{os.path.dirname(__file__)}/../maze_maps.txt", "r") as f:
    LIST_MAZE_STR: list[str] = eval(f.read())


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """Run a function with a timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Solver timed out after {timeout_seconds} seconds")


def evaluate_solver_on_single_maze(
    solver_cls: type[Solver],
    solver_kwargs: dict[str, Any],
    maze_str: str,
    render: bool = False,
) -> float:
    domain_factory = lambda: Maze(maze_str=maze_str)
    domain = domain_factory()
    max_steps = 2 * domain._num_cols * domain._num_rows
    with solver_cls(domain_factory=domain_factory, **solver_kwargs) as solver:
        episodes = rollout(
            solver=solver,
            domain=domain,
            render=render,
            verbose=False,
            num_episodes=3,
            max_steps=max_steps,
            return_episodes=True,
        )
    total_cost = (
        sum(sum(c.cost for c in ep_costs) for ep_obs, ep_actions, ep_costs in episodes)
        / max_steps
    )  # normalized by maze size
    return total_cost


def evaluate_solver_on_mazes(
    solver_cls: type[Solver],
    solver_kwargs: dict[str, Any],
    list_maze_str: Optional[list[str]] = None,
    render: bool = False,
) -> float:
    if list_maze_str is None:
        list_maze_str = LIST_MAZE_STR
    return sum(
        evaluate_solver_on_single_maze(
            solver_cls=solver_cls,
            solver_kwargs=solver_kwargs,
            maze_str=maze_str,
            render=render,
        )
        for maze_str in list_maze_str
    )


def evaluate_solver_on_mazes_with_timeout(
    solver_cls: type[Solver],
    solver_kwargs: dict[str, Any],
    list_maze_str: Optional[list[str]] = None,
    timeout: int = 60,
    render: bool = False,
) -> float:
    return run_with_timeout(
        func=evaluate_solver_on_mazes,
        kwargs=dict(
            solver_cls=solver_cls,
            solver_kwargs=solver_kwargs,
            list_maze_str=list_maze_str,
            render=render,
        ),
        timeout_seconds=timeout,
    )


D = Maze


class EvolvedSolver(Solver, Policies):
    def _solve(self) -> None:
        pass

    T_domain = D

    def __init__(self, domain_factory: Callable[[], Maze], program_path: str):
        super().__init__(domain_factory)

        # retrieve the evolved sample_action()
        module_name = "evolved_program"
        spec = importlib.util.spec_from_file_location(module_name, program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.program_path = program_path
        evolved_planner_cls = module.Planner

        # get static arguments for sample_action
        domain: Maze = self.domain_factory()
        self._planner = evolved_planner_cls(goal=domain._goal, maze=domain._maze)

    def _sample_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        action_outside_hub = self._planner.sample_action(
            state=observation,
        )
        action = Action[action_outside_hub.name]
        return action

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True


def evaluate(program_path: str) -> EvaluationResult:
    try:
        cost = evaluate_solver_on_mazes_with_timeout(
            solver_cls=EvolvedSolver,
            solver_kwargs=dict(
                program_path=program_path,
            ),
            timeout=EVALUATION_TIMEOUT,
        )
    except Exception as e:
        return EvaluationResult(
            metrics=dict(combined_score=-float("inf")),
            artifacts=dict(
                error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()
            ),
        )
    return EvaluationResult(
        metrics=dict(
            cost=cost,
            combined_score=-cost,  # openevolve maximize the combined score
        )
    )


if __name__ == "__main__":
    program_paths = [
        "initial_program.py",
        "dfs_program.py",
        "initial_program_broken.py",
    ]
    for program_path in program_paths:
        total_cost = evaluate(program_path)
        print(program_path, total_cost)
