import argparse
import time
from typing import Optional

from skdecide import DiscreteDistribution, rollout_episode
from skdecide.builders.domain.scheduling.scheduling_domains_modelling import (
    SchedulingAction,
    SchedulingActionEnum,
    State,
    timer,
)
from skdecide.hub.domain.rcpsp.rcpsp_sk import MSRCPSP, RCPSP
from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import (
    load_domain,
    load_multiskill_domain,
)
from skdecide.hub.solver.do_solver.do_solver_scheduling import DOSolver, SolvingMethod
from skdecide.hub.solver.sgs_policies.sgs_policies import (
    BasePolicyMethod,
    PolicyMethodParams,
)


def do_rollout_comparaison(domain: RCPSP, solver, inplace: bool = True):
    domain.set_inplace_environment(inplace)
    tic = time.perf_counter()
    states, actions, values = rollout_episode(
        domain=domain,
        solver=solver,
        from_memory=domain.get_initial_state(),
        max_steps=1000,
        outcome_formatter=None,
        action_formatter=None,
        verbose=False,
    )
    toc = time.perf_counter()
    print(
        f"{toc - tic:0.4f} seconds to rollout policy with inplace={inplace} environment, "
        f"Final time of the schedule : "
        f"{states[-1].t}"
    )
    return states


def run_expe(path: str, inplace, makespan: int, plot: bool, output: Optional[str]):
    do_solver = SolvingMethod.PILE  # Greedy solver.
    domain: RCPSP = load_domain(path)
    for task in domain.duration_dict:
        for mode in domain.duration_dict[task]:
            domain.duration_dict[task][mode] *= makespan
    if inplace is None:
        inplace = [0, 1]
    for flag in inplace:
        solver = DOSolver(
            policy_method_params=PolicyMethodParams(
                base_policy_method=BasePolicyMethod.FOLLOW_GANTT,
                # policy will just follow the output gantt of the greedy solver
                delta_index_freedom=0,
                delta_time_freedom=0,
            ),
            method=do_solver,
        )
        solver.solve(domain_factory=lambda: domain)
        states = do_rollout_comparaison(domain, solver, flag != 0)
        if output:
            with open(output, "wt") as f:
                for state in states:
                    f.write(f"{state.t}\n")

        # print([id(s) for s in states])
        print(f"Timers deepcopy {timer}")

        if plot:
            import matplotlib.pylab as plt

            from skdecide.hub.solver.do_solver.sk_to_do_binding import (
                from_last_state_to_solution,
            )

            do_sol = from_last_state_to_solution(states[-1], domain)
            from skdecide.discrete_optimization.rcpsp.rcpsp_plot_utils import (
                plot_ressource_view,
                plot_task_gantt,
            )

            # Each line of the plot is a task
            plot_task_gantt(do_sol.problem, do_sol)
            # Plot resource consumption plot
            plot_ressource_view(do_sol.problem, do_sol)
            plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Solve Graph problem using various algorithms"
    )
    parser.add_argument("--graph_file", type=str, required=True, help="SM file")
    parser.add_argument(
        "--inplace",
        type=int,
        action="append",
        required=False,
        help="set inplace (can be repeated)",
    )
    parser.add_argument(
        "--makespan", type=int, required=False, default=1, help="makespan"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        required=False,
        default=False,
        help="plot Gantt graph",
    )
    parser.add_argument(
        "--output", type=str, required=False, help="store solution in a text file"
    )
    args = parser.parse_args()

    run_expe(args.graph_file, args.inplace, args.makespan, args.plot, args.output)
