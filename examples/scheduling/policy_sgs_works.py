from examples.scheduling.rcpsp_datasets import get_complete_path
from skdecide.hub.domain.rcpsp.rcpsp_sk import (
    RCPSP,
    build_n_determinist_from_stochastic,
    build_stochastic_from_deterministic,
)
from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_domain
from skdecide.hub.solver.do_solver.do_solver_scheduling import (
    DOSolver,
    SolvingMethod,
    from_solution_to_policy,
)
from skdecide.hub.solver.meta_policy.meta_policies import MetaPolicy
from skdecide.hub.solver.policy_evaluators.policy_evaluator import (
    rollout_based_compute_expected_cost_for_policy_scheduling,
)
from skdecide.hub.solver.sgs_policies.sgs_policies import (
    BasePolicyMethod,
    PolicyMethodParams,
)


# Compare different online policies based on permutation on few sampled scenarios.
def run_and_compare_policies_sampled_scenarios():
    import random

    domain: RCPSP = load_domain(get_complete_path("j601_1.sm"))
    task_to_noise = set(
        random.sample(domain.get_tasks_ids(), min(30, len(domain.get_tasks_ids())))
    )
    stochastic_domain = build_stochastic_from_deterministic(
        domain, task_to_noise=task_to_noise
    )
    deterministic_domains = build_n_determinist_from_stochastic(
        stochastic_domain, nb_instance=5
    )
    for d in deterministic_domains:
        d.set_inplace_environment(True)
    stochastic_domain.set_inplace_environment(True)
    state = domain.get_initial_state()
    domain.set_inplace_environment(True)
    solver = DOSolver(
        policy_method_params=PolicyMethodParams(
            base_policy_method=BasePolicyMethod.FOLLOW_GANTT,
            delta_index_freedom=0,
            delta_time_freedom=0,
        ),
        method=SolvingMethod.LS,
        dict_params={"nb_iteration_max": 20},
    )
    solver.solve(domain_factory=lambda: domain)
    policy_methods = [
        PolicyMethodParams(
            base_policy_method=method, delta_time_freedom=0, delta_index_freedom=0
        )
        for method in [
            BasePolicyMethod.SGS_PRECEDENCE,  # ,
            BasePolicyMethod.SGS_READY,
            BasePolicyMethod.SGS_STRICT,
        ]
    ]
    # policy_methods += [PolicyMethodParams(base_policy_method=BasePolicyMethod.SGS_INDEX_FREEDOM,
    #                                       delta_time_freedom=0,
    #                                       delta_index_freedom=i)
    #                    for i in range(10)]
    # policy_methods += [PolicyMethodParams(base_policy_method=BasePolicyMethod.SGS_TIME_FREEDOM,
    #                                       delta_time_freedom=t,
    #                                       delta_index_freedom=0)
    #                    for t in range(0, 200, 5)]
    policies = {
        i: from_solution_to_policy(
            solution=solver.best_solution,
            domain=stochastic_domain,
            policy_method_params=policy_methods[i],
        )
        for i in range(len(policy_methods))
    }
    meta_policy = MetaPolicy(
        policies={k: policies[k] for k in policies},
        execution_domain=domain,
        known_domain=domain,
        nb_rollout_estimation=1,
        verbose=True,
    )
    policies["meta"] = meta_policy
    keys = list(policies.keys())[::-1]
    value_function_dict = {}
    for key in keys:
        value_function_dict[key] = 0.0
        for k, d in enumerate(deterministic_domains):
            (
                value_function_d,
                policy_dict,
                preds,
                succs,
            ) = rollout_based_compute_expected_cost_for_policy_scheduling(
                d, policies[key], nb_rollout=1
            )
            value_function_dict[key] += value_function_d[state]
        print("key : ", key, value_function_dict[key])


if __name__ == "__main__":
    run_and_compare_policies_sampled_scenarios()
