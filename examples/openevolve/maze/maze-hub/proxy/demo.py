"""Demo of proxy opensolve wrapper (no domain api awareness) on maze from hub

Usage:
    demo.py [--no-solve]

Arguments:
    --no-solve  does not solve, load previous solve history instead

Results can be visualized via

    OPENEVOLVE_VISUALIZER_PATH="path/to/openevolve-repo/scripts"
    python $OPENEVOLVE_VISUALIZER_PATH/visualizer.py --path output/demo

Prerequisites:
- Set OPENAI_API_KEY properly
    echo OPENAI_API_KEY="xxxxx" >> .env

"""

import os
from difflib import unified_diff
from enum import Enum

from docopt import docopt
from dotenv import load_dotenv
from evaluator import LIST_MAZE_STR

from skdecide import rollout
from skdecide.hub.domain.maze import Maze
from skdecide.hub.domain.maze.maze import Action
from skdecide.hub.solver.openevolve import (
    ProxyOpenEvolve,
)
from skdecide.hub.solver.openevolve.code_utils import check_diff_outside_evolveblocks

load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    # For demo purposes. Should be exported with proper value before running the script
    os.environ["OPENAI_API_KEY"] = ""

example_dir = os.path.dirname(__file__)

initial_program_path = f"{example_dir}/initial_program.py"
evaluator_path = f"{example_dir}/evaluator.py"
config = f"{example_dir}/config.yaml"
output_dir = f"{example_dir}/output/demo"

intermediate_save_dir = f"{output_dir}/intermediate"
final_save_dir = f"{output_dir}/final"

domain_factory = Maze


def planner_kwargs_factory(domain: Maze):
    return dict(goal=domain._goal, maze=domain._maze)


def planner_action_converter(planner_action: Enum) -> Action:
    return Action[planner_action.name]


solver_kwargs = dict(
    domain_factory=domain_factory,
    initial_program_path=initial_program_path,
    evaluator_path=evaluator_path,
    config=config,
    output_dir=output_dir,
    planner_kwargs_factory=planner_kwargs_factory,
    planner_action_converter=planner_action_converter,
)


def main(skip_solve: int = False):
    if not skip_solve:
        solver = ProxyOpenEvolve(**solver_kwargs)
        print("### INITIAL SOLVE ###")
        solver.solve()

        # store results
        solver.save(intermediate_save_dir)

        # load and go on for 5 more iterations
        print("### FURTHER SOLVE ###")
        solver = ProxyOpenEvolve(**solver_kwargs, nb_iterations=5)
        solver.load(intermediate_save_dir)
        solver.solve()
        solver.save(final_save_dir)

    # Evaluation on several mazes
    print("### FINAL ROLLOUT ON SEVERAL MAZES ###")
    total_cost = 0
    for maze_str in LIST_MAZE_STR:
        domain_factory = lambda: Maze(maze_str=maze_str)
        domain = domain_factory()
        max_steps = 2 * domain._num_cols * domain._num_rows
        render = False
        # update domain_factory in solver kwargs
        new_solver_kwargs = dict(solver_kwargs, domain_factory=domain_factory)
        with ProxyOpenEvolve(**new_solver_kwargs) as new_solver:
            new_solver.load(final_save_dir)
            episodes = rollout(
                solver=new_solver,
                domain=domain,
                render=render,
                verbose=False,
                num_episodes=3,
                max_steps=max_steps,
                return_episodes=True,
            )
            total_cost += (
                sum(
                    sum(c.cost for c in ep_costs)
                    for ep_obs, ep_actions, ep_costs in episodes
                )
                / max_steps
            )  # normalized by maze size
    print(total_cost)

    # print diff of generated code
    with ProxyOpenEvolve(**new_solver_kwargs) as new_solver:
        code_before_evolution = new_solver.get_best_program_code()

        new_solver.load(final_save_dir)
        best_code = new_solver.get_best_program_code()
    print("#### DIFF BETWEEN INITIAL CODE AND BEST CODE")
    print(
        "".join(
            unified_diff(
                code_before_evolution.splitlines(True),
                best_code.splitlines(True),
                fromfile="initial_program",
                tofile="best_program",
            )
        )
    )

    # check diff outside the evolve blocks
    print("#### DIFF OUTSIDE EVOLVE BLOCKS?")
    check = check_diff_outside_evolveblocks(
        code_before_evolution, best_code, path1="initial_program", path2="best_program"
    )
    if check:
        print("None.")


if __name__ == "__main__":
    arguments = docopt(__doc__)
    skip_solve = arguments["--no-solve"]
    main(skip_solve=skip_solve)
