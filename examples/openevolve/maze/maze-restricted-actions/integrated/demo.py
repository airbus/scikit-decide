"""Demo of integrated opensolve wrapper (aware of scikit-decide domain api) on maze from hub

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
import sys
from difflib import unified_diff

from docopt import docopt
from dotenv import load_dotenv

from skdecide import rollout
from skdecide.hub.solver.openevolve import (
    IntegratedOpenEvolve,
)
from skdecide.hub.solver.openevolve.code_utils import check_diff_outside_evolveblocks

load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    # For demo purposes. Should be exported with proper value before running the script
    os.environ["OPENAI_API_KEY"] = ""

example_dir = os.path.dirname(__file__)

with open(f"{example_dir}/../../maze_maps.txt", "r") as f:
    LIST_MAZE_STR: list[str] = eval(f.read())


config = f"{example_dir}/config.yaml"
output_dir = f"{example_dir}/output/demo"

intermediate_save_dir = f"{output_dir}/intermediate"
final_save_dir = f"{output_dir}/final"


sys.path.append(f"{example_dir}/..")
from maze_restricted_domain import Maze

domain_factory = Maze

# domain factories on which the evolved program will be evaluated
# NB: because how namespaces are cached in python,
# you have to put the parameters as default arguments of the lambda function
# to correctly create factories
evaluator_domain_factories = [
    lambda maze_str=maze_str: Maze(maze_str=maze_str) for maze_str in LIST_MAZE_STR
]


def evaluator_rollout_max_steps(domain: Maze) -> int:
    return 2 * domain.n_cells


solver_kwargs = dict(
    domain_factory=domain_factory,
    evaluator_domain_factories=evaluator_domain_factories,
    evaluator_rollout_max_steps=evaluator_rollout_max_steps,
    config=config,
    output_dir=output_dir,
    evaluator_timeout=60,
)


def main(skip_solve: int = False):
    print("### INITIAL PROGRAM GENERATED")
    solver = IntegratedOpenEvolve(**solver_kwargs)
    code = solver.get_best_program_code()
    print(code)

    print("### TESTING EVALUATOR")
    print(f"Initial program: {solver.evaluate_program_code(code)}")
    program_paths = [
        "program_examples/dfs_program.py",
        "program_examples/initial_program_broken.py",
    ]
    for program_path in program_paths:
        with open(program_path, "r") as f:
            code = f.read()
        print(f"{program_path}: {solver.evaluate_program_code(code)}")

    if not skip_solve:
        print("### INITIAL SOLVE ###")
        solver = IntegratedOpenEvolve(**solver_kwargs)
        solver.solve()

        # store results
        solver.save(intermediate_save_dir)

        # load and go on for 5 more iterations
        print("### FURTHER SOLVE ###")
        solver = IntegratedOpenEvolve(**solver_kwargs, nb_iterations=5)
        solver.load(intermediate_save_dir)
        solver.solve()
        solver.save(final_save_dir)

    # Evaluation on several mazes
    print("### FINAL ROLLOUT ON SEVERAL MAZES ###")
    total_cost = 0
    for maze_str in LIST_MAZE_STR:
        # update domain_factory in solver kwargs
        # WARNING: as the solver will autocast (hus modify) the domain,
        # the domain_factory must return a new instance or a copy of the domain
        # (cannot use directly lambda: domain)
        domain_factory = lambda: Maze(maze_str)
        rollout_domain = domain_factory()
        new_solver_kwargs = dict(solver_kwargs, domain_factory=domain_factory)
        max_steps = evaluator_rollout_max_steps(rollout_domain)
        render = False
        # Instantiate a new solver with the proper domain factory so that
        # the wrapped evolved planner works with the characteristics of the new domain
        with IntegratedOpenEvolve(**new_solver_kwargs) as new_solver:
            new_solver.load(final_save_dir)
            episodes = rollout(
                solver=new_solver,
                domain=rollout_domain,
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
    with IntegratedOpenEvolve(**new_solver_kwargs) as new_solver:
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
