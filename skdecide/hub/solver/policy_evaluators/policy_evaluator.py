from skdecide import GoalMDPDomain, DeterministicPolicySolver
from skdecide.builders.scheduling.scheduling_domains import SchedulingDomain
from skdecide.builders.solver.policy import DeterministicPolicies
from collections import defaultdict
from typing import Tuple, Dict, Any, Set, List


# Adapted from skdecide/hub/solver/cssp/utils/cost_shift_solver.py works.
def construct_dict_policy(domain: GoalMDPDomain, policy: DeterministicPolicies)->Tuple[Dict[Any, Any],
                                                                                       Dict[Any, Set[Any]],
                                                                                       Dict[Any,
                                                                                            List[Tuple[Any, float]]]]:
    stack = [domain.get_initial_state()]
    # We store predecessors to make it easier to retrieve the expected costs
    # later on
    preds = defaultdict(set)
    succs = defaultdict()
    policy_dict = {}
    while len(stack) > 0:
        s = stack.pop()
        if domain.is_terminal(s) or s in policy_dict:
            continue
        action = policy.get_next_action(s)
        policy_dict[s] = action
        successors = domain.get_next_state_distribution(s, action).get_values()
        succs[s] = successors
        for succ, prob in successors:
            if prob != 0:
                stack.append(succ)
                preds[succ].add(s)
    # Sanity check that successors and predecessors are equal
    for state, successor_pairs in succs.items():
        for succ, prob in successor_pairs:
            assert (state in preds[succ])
    return policy_dict, preds, succs


def expected_costs_for_policy(domain,
                              policy_dict,
                              preds,
                              succs):
    # Compute value function for states that are explored by the policy.
    opt_val = dict()
    # Initialize states where all successors have potentially known values
    stack = set()
    for s in preds.keys():
        if domain.is_goal(s):
            opt_val[s] = 0
            stack.update(preds[s])
    while len(stack) > 0:
        s = stack.pop()
        # Assert that all successors have known optimal values
        if s in opt_val or not successor_value_is_known(succs[s], opt_val):
            continue
        a = policy_dict[s]
        main_cost = 0
        for succ, prob in succs[s]:
            # evaluate objective function on transition
            val = domain.get_transition_value(s, a, succ)
            main_cost += prob * (val.cost + opt_val[succ])
        opt_val[s] = main_cost
        stack.update(preds[s])
    return opt_val


def compute_expected_cost_for_policy(domain, policy):
    policy_dict, preds, succs = construct_dict_policy(domain, policy)
    value_function_dict = expected_costs_for_policy(domain, policy_dict, preds, succs)
    return value_function_dict, policy_dict, preds, succs


def my_custom_rollout(domain: GoalMDPDomain, state, policy: DeterministicPolicies):
    states = [state]
    values = []
    summed_value = 0.
    actions = []
    while True:
        action = policy.get_next_action(states[-1])
        next_state = SchedulingDomain._state_sample(domain, states[-1], action).state
        value = domain.get_transition_value(states[-1], action, next_state)
        values += [value.cost]
        summed_value += value.cost
        states += [next_state]
        actions += [action]
        if domain.is_goal(states[-1]):
            break
        if domain.is_terminal(states[-1]):
            summed_value += 1000  # penalty
            break
    return states, summed_value, values, actions

# for uncertain domain leading to intractable number of states. which is the case for non trivial scheduling domains
# for example
import numpy as np
def rollout_based_policy_estimation(domain: GoalMDPDomain, policy: DeterministicPolicies, nb_rollout:int=1) \
        ->Tuple[Dict[Any,
                     Any],
                Dict[Any, Any],
                Dict[Any,
                     Set[Any]],
                Dict[Any,
                     Set[Any]]]:
    policy_dict = {}
    nb_visit_dict = {}
    summed_value = {}
    final_value = {}
    preds = {}
    succs = {}
    for rollout in range(nb_rollout):
        states, summed_value_rollout, values, actions = my_custom_rollout(domain, domain.get_initial_state(), policy)
        values_np = np.array(values[::-1]).cumsum()
        k = 0
        for j in range(len(states)-2, -1, -1):
            if states[j] not in nb_visit_dict:
                nb_visit_dict[states[j]] = 0
                summed_value[states[j]] = 0
                final_value[states[j]] = 0
                policy_dict[states[j]] = actions[j]
                preds[states[j]] = set()
                succs[states[j]] = set()
            summed_value[states[j]] += values_np[k]
            nb_visit_dict[states[j]] += 1
            if j > 0:
                preds[states[j]].add(states[j-1])
            succs[states[j]].add(states[j+1])
            k += 1
    final_value = {st: summed_value[st]/nb_visit_dict[st] for st in summed_value}
    return final_value, policy_dict, preds, succs


def rollout_based_policy_estimation_fast_scheduling(domain: SchedulingDomain,
                                                    policy: DeterministicPolicies,
                                                    nb_rollout:int=1) \
        ->Tuple[Dict[Any,
                     Any],
                Dict[Any, Any],
                Dict[Any,
                     Set[Any]],
                Dict[Any,
                     Set[Any]]]:
    policy_dict = {}
    nb_visit_dict = {}
    summed_value = {}
    final_value = {}
    preds = {}
    succs = {}
    s = domain.get_initial_state()
    summed_value[s] = 0
    nb_visit_dict[s] = 0
    domain.set_inplace_environment(True)
    for rollout in range(nb_rollout):
        states, summed_value_rollout, values, actions = my_custom_rollout(domain,
                                                                          s.copy(),
                                                                          policy)
        summed_value[s] += states[-1].t-s.t
        nb_visit_dict[s] += 1
    final_value = {st: summed_value[st]/nb_visit_dict[st] for st in summed_value}
    return final_value, policy_dict, preds, succs


def rollout_based_compute_expected_cost_for_policy(domain, policy, nb_rollout=100):
    final_value, policy_dict, preds, succs = rollout_based_policy_estimation(domain=domain,
                                                                             policy=policy,
                                                                             nb_rollout=nb_rollout)
    return final_value, policy_dict, preds, succs


def rollout_based_compute_expected_cost_for_policy_scheduling(domain, policy, nb_rollout=100):
    final_value, policy_dict, preds, succs = rollout_based_policy_estimation_fast_scheduling(domain=domain,
                                                                                             policy=policy,
                                                                                             nb_rollout=nb_rollout)
    return final_value, policy_dict, preds, succs

def successor_value_is_known(successors, opt_val):
    for succ, prob in successors:
        if prob != 0 and succ not in opt_val:
            return False
    return True
