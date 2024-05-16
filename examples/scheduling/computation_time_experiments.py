import time

from rcpsp_datasets import get_complete_path

from skdecide import rollout_episode
from skdecide.hub.domain.rcpsp.rcpsp_sk import RCPSP
from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_domain
from skdecide.hub.solver.do_solver.do_solver_scheduling import DOSolver, SolvingMethod
from skdecide.hub.solver.do_solver.sgs_policies import (
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


def run_expe():
    do_solver = SolvingMethod.PILE  # Greedy solver.
    domain: RCPSP = load_domain(get_complete_path("j1201_1.sm"))
    solver = DOSolver(
        domain_factory=lambda: domain,
        policy_method_params=PolicyMethodParams(
            base_policy_method=BasePolicyMethod.FOLLOW_GANTT,
            # policy will just follow the output gantt of the greedy solver
            delta_index_freedom=0,
            delta_time_freedom=0,
        ),
        method=do_solver,
    )
    solver.solve()
    states_deepcopy = do_rollout_comparaison(domain, solver, False)
    states_inplace = do_rollout_comparaison(domain, solver, True)

    # main difference is that in the inplace environment, the object are overwritten as you can see
    print([id(s) for s in states_inplace])
    # Where as the deepcopy solution is creating each time a new object :
    print([id(s) for s in states_deepcopy])

    # There is a 20 factor speedup, and it's even more when the makespan is large, let's try :
    domain: RCPSP = load_domain(get_complete_path("j1201_1.sm"))
    for task in domain.duration_dict:
        for mode in domain.duration_dict[task]:
            domain.duration_dict[task][mode] *= 20
    domain.set_inplace_environment(False)
    solver = DOSolver(
        domain_factory=lambda: domain,
        policy_method_params=PolicyMethodParams(
            base_policy_method=BasePolicyMethod.FOLLOW_GANTT,
            delta_index_freedom=0,
            delta_time_freedom=0,
        ),
        method=do_solver,
    )
    solver.solve()
    states_deepcopy = do_rollout_comparaison(domain, solver, False)
    states_inplace = do_rollout_comparaison(domain, solver, True)


if __name__ == "__main__":
    run_expe()
