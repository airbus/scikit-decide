"""Demo of integrated opensolve wrapper (aware of scikit-decide domain api) on maze from hub

Usage:
    demo.py [--no-solve]

Arguments:
    --no-solve  does not solve, load previous solve history instead


Results can be visualized via

    python path/to/openevolve-repo/scripts/visualizer.py --path output/demo

Prerequisites

- Download beluga challenge benchmarks and set the proper env variable



"""

import glob
import json
import os
import re
import sys

from docopt import docopt
from dotenv import load_dotenv

from skdecide import rollout
from skdecide.hub.solver.openevolve import (
    IntegratedOpenEvolve,
)

load_dotenv()

# PARAMS
MAX_STEPS_BY_JIG = 10  # rollout max step (to be mult by n_jigs)
N_PB_EVAL = 1  # nb of pb to use for evaluation
N_PB_VALID = 3  # nb of pbs to use for final validation
PICK_PB_RANDOM = False  # pick random pbs or sequentially


# "install" beluga toolkit
beluga_toolkit_repo = os.path.abspath(os.environ["BELUGA_TOOLKIT_REPO"])
sys.path.append(beluga_toolkit_repo)
from beluga_lib.beluga_problem import BelugaProblemDecoder
from skd_domains.skd_pddl_domain import SkdPDDLDomain


# list json def files
def extract_problem_id(problem_file: str) -> int:
    problem_name = os.path.basename(problem_file)
    return int(re.match("problem_([0-9]+)", problem_name).group(1))


beluga_benchmark_dir = os.environ["BELUGA_BENCHMARK_DIR"]
beluga_json_files = sorted(
    glob.glob(f"{beluga_benchmark_dir}/*.json"), key=extract_problem_id
)

beluga_json_main = beluga_json_files[0]
if PICK_PB_RANDOM:
    raise NotImplementedError()
else:
    beluga_jsons_eval = beluga_json_files[:N_PB_EVAL]
    beluga_jsons_valid = beluga_json_files[N_PB_EVAL : N_PB_EVAL + N_PB_VALID]


def create_domain(problem_json: str) -> SkdPDDLDomain:
    problem_folder = os.path.dirname(problem_json)
    problem_name = os.path.basename(problem_json)
    with open(problem_json, "r") as fp:
        inst = json.load(fp, cls=BelugaProblemDecoder)
    domain = SkdPDDLDomain(inst, problem_name, problem_folder)
    domain.n_jigs = len(inst.jigs)
    return domain


# config, output, save path
example_dir = os.path.dirname(os.path.abspath(__file__))
config = f"{example_dir}/config.yaml"
output_dir = f"{example_dir}/output/demo"

save_dir = f"{output_dir}/final"


domain_factory = lambda: create_domain(beluga_json_main)

# domain factories on which the evolved program will be evaluated
# NB: because how namespaces are cached in python,
# you have to put the parameters as default arguments of the lambda function
# to correctly create factories
evaluator_domain_factories = [
    lambda problem_json=problem_json: create_domain(problem_json)
    for problem_json in beluga_jsons_eval
]


def evaluator_rollout_max_steps(domain):
    return domain.n_jigs * MAX_STEPS_BY_JIG


solver_kwargs = dict(
    domain_factory=domain_factory,
    evaluator_domain_factories=evaluator_domain_factories,
    evaluator_rollout_max_steps=evaluator_rollout_max_steps,
    config=config,
    output_dir=output_dir,
    evaluator_timeout=120,
    evaluator_rollout_num_episodes=1,
    prompt_add_public_api_instruction=False,
)


def main(skip_solve: int = False):
    if not skip_solve:
        print("### INITIAL SOLVE ###")
        solver = IntegratedOpenEvolve(**solver_kwargs)
        solver.solve()
        solver.save(save_dir)

    # Evaluation on several domains
    print("### FINAL ROLLOUT ON SEVERAL PROBLEM ###")
    total_cost = 0
    for problem_json in beluga_jsons_valid:
        # update domain_factory in solver kwargs
        # WARNING: as the solver will autocast (hus modify) the domain,
        # the domain_factory must return a new instance or a copy of the domain
        # (cannot use directly lambda: domain)
        domain_factory = lambda: create_domain(problem_json)
        rollout_domain = domain_factory()
        new_solver_kwargs = dict(solver_kwargs, domain_factory=domain_factory)
        max_steps = evaluator_rollout_max_steps(rollout_domain)
        render = False
        # Instantiate a new solver with the proper domain factory so that
        # the wrapped evolved planner works with the characteristics of the new domain
        with IntegratedOpenEvolve(**new_solver_kwargs) as new_solver:
            new_solver.load(save_dir)
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


if __name__ == "__main__":
    arguments = docopt(__doc__)
    skip_solve = arguments["--no-solve"]
    main(skip_solve=skip_solve)
