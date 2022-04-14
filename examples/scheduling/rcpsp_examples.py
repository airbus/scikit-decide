from examples.discrete_optimization.rcpsp_multiskill_parser_example import (
    get_complete_path_ms,
    get_data_available_ms,
)
from examples.discrete_optimization.rcpsp_parser_example import (
    get_complete_path,
    get_data_available,
)
from skdecide import DiscreteDistribution, rollout_episode
from skdecide.builders.domain.scheduling.scheduling_domains_modelling import (
    SchedulingAction,
    SchedulingActionEnum,
    State,
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


def random_walk():
    domain: RCPSP = load_domain(get_complete_path("j301_1.sm"))
    state = domain.get_initial_state()
    domain.set_inplace_environment(False)
    states, actions, values = rollout_episode(
        domain=domain,
        solver=None,
        from_memory=state,
        max_steps=500,
        outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
    )
    print(sum([v.cost for v in values]))
    print("rollout done")
    print("end times: ")
    for task_id in states[-1].tasks_details.keys():
        print("end task", task_id, ": ", states[-1].tasks_details[task_id].end)
    from discrete_optimization.rcpsp.rcpsp_plot_utils import (
        plot_resource_individual_gantt,
        plot_ressource_view,
        plot_task_gantt,
        plt,
    )

    from skdecide.hub.solver.do_solver.sk_to_do_binding import (
        from_last_state_to_solution,
    )

    do_sol = from_last_state_to_solution(states[-1], domain)
    plot_task_gantt(do_sol.problem, do_sol)
    plot_ressource_view(do_sol.problem, do_sol)
    plot_resource_individual_gantt(do_sol.problem, do_sol)
    plt.show()


def do_singlemode():
    do_solver = SolvingMethod.CP
    domain: RCPSP = load_domain(get_complete_path("j301_1.sm"))
    domain.set_inplace_environment(False)
    state = domain.get_initial_state()
    print("Initial state : ", state)
    solver = DOSolver(
        policy_method_params=PolicyMethodParams(
            base_policy_method=BasePolicyMethod.SGS_PRECEDENCE,
            delta_index_freedom=0,
            delta_time_freedom=0,
        ),
        method=do_solver,
    )
    solver.solve(domain_factory=lambda: domain)
    print(do_solver)

    states, actions, values = rollout_episode(
        domain=domain,
        solver=solver,
        from_memory=state,
        max_steps=500,
        outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
    )
    print(sum([v.cost for v in values]))
    print("rollout done")
    print("end times: ")
    for task_id in states[-1].tasks_details.keys():
        print("end task", task_id, ": ", states[-1].tasks_details[task_id].end)


def do_multimode():
    domain: RCPSP = load_domain(get_complete_path("j1010_2.mm"))
    state = domain.get_initial_state()
    solver = DOSolver(
        policy_method_params=PolicyMethodParams(
            base_policy_method=BasePolicyMethod.FOLLOW_GANTT,
            delta_index_freedom=0,
            delta_time_freedom=0,
        ),
        method=SolvingMethod.CP,
    )
    solver.solve(domain_factory=lambda: domain)
    states, actions, values = rollout_episode(
        domain=domain,
        solver=solver,
        from_memory=state,
        max_steps=1000,
        action_formatter=lambda a: f"{a}",
        outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
    )
    print("rollout done")
    print("end times: ")
    for task_id in states[-1].tasks_details.keys():
        print("end task", task_id, ": ", states[-1].tasks_details[task_id].end)


def random_walk_multiskill():
    domain: MSRCPSP = load_multiskill_domain(get_data_available_ms()[0])
    state = domain.get_initial_state()
    states, actions, values = rollout_episode(
        domain=domain,
        solver=None,
        from_memory=state,
        max_steps=1000,
        action_formatter=lambda a: f"{a}",
        outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
    )
    print("rollout done")
    print("end times: ")
    for task_id in states[-1].tasks_details.keys():
        print("end task", task_id, ": ", states[-1].tasks_details[task_id].end)


def do_multiskill():
    domain: MSRCPSP = load_multiskill_domain(get_data_available_ms()[0])
    domain.set_inplace_environment(False)
    state = domain.get_initial_state()
    solver = DOSolver(
        policy_method_params=PolicyMethodParams(
            base_policy_method=BasePolicyMethod.SGS_PRECEDENCE,
            delta_index_freedom=0,
            delta_time_freedom=0,
        ),
        method=SolvingMethod.LNS_CP,
    )
    solver.get_available_methods(domain)
    solver.solve(domain_factory=lambda: domain)
    states, actions, values = rollout_episode(
        domain=domain,
        solver=solver,
        from_memory=state,
        max_steps=3000,
        action_formatter=lambda a: f"{a}",
        outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
    )
    print(sum([v.cost for v in values]))
    print("rollout done")
    print("end times: ")
    for task_id in states[-1].tasks_details.keys():
        print("end task", task_id, ": ", states[-1].tasks_details[task_id].end)


if __name__ == "__main__":
    # Run Constraint programming based solver on single mode and multimode
    do_singlemode()
    do_multimode()

    # Random walk on rcpsp and multiskill rcpsp instances.
    random_walk()
    random_walk_multiskill()

    # Takes few minutes : run LNS-CP on multiskill instance
    do_multiskill()
