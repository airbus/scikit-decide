from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_domain
from skdecide.hub.domain.rcpsp.rcpsp_sk import RCPSP
from skdecide.hub.solver.do_solver.do_solver_scheduling import DOSolver, PolicyMethodParams, SolvingMethod, BasePolicyMethod
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='Solve RCPSP with CP.')
    parser.add_argument('--sm_file', type=str, required=True, help='SM file')
    args = parser.parse_args()
    solve(args.sm_file)

def solve(sm_file:str):
    """Solve j301_1 with CP (MiniZinc)"""
    big_domain: RCPSP = load_domain(sm_file)
    big_domain.set_inplace_environment(False)

    state = big_domain.get_initial_state()

    # base_policy_method=BasePolicyMethod.FOLLOW_GANTT
    base_policy_method=BasePolicyMethod.SGS_PRECEDENCE

    policy_method_params = PolicyMethodParams(base_policy_method=base_policy_method, delta_index_freedom=0, delta_time_freedom=0)

    # We want to focus on CP and LNS_CP
    method = SolvingMethod.CP
    # method = SolvingMethod.LNS_CP

    solver = DOSolver(policy_method_params=policy_method_params, method=method, dict_params={"verbose": False})

    tic = time.perf_counter()
    solver.solve(domain_factory=lambda: big_domain)
    toc = time.perf_counter()
    print(f"Time: {toc - tic:0.4f} seconds")

if __name__ == "__main__":
    main()
