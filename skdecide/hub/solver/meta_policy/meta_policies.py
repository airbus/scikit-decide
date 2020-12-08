from typing import Union, Dict, Any

from skdecide import rollout_episode
from skdecide.builders.solver import DeterministicPolicies, Policies
from skdecide.builders.scheduling.scheduling_domains import SchedulingDomain, D


class MetaPolicy(DeterministicPolicies):
    T_domain = D

    def __init__(self,
                 policies: Dict[Any, DeterministicPolicies],
                 execution_domain: SchedulingDomain,
                 known_domain: SchedulingDomain,
                 nb_rollout_estimation=1,
                 verbose=True):
        self.known_domain = known_domain
        self.known_domain.fast = True
        self.execution_domain = execution_domain
        self.policies = policies
        self.current_states = {method: None for method in policies}
        self.nb_rollout_estimation = nb_rollout_estimation
        self.verbose = verbose

    def reset(self):
        self.current_states = {method: None for method in self.policies}

    def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        results = {}
        actions_map = {}
        self.known_domain.set_inplace_environment(True)
        actions_c = [self.policies[method].get_next_action(observation)
                     for method in self.policies]
        if len(set(actions_c)) > 1:
            for method in self.policies:
                results[method] = 0.
                for j in range(self.nb_rollout_estimation):
                    states, actions, values = rollout_episode(domain=self.known_domain,
                                                              solver=self.policies[method],
                                                              outcome_formatter=None,
                                                              action_formatter=None,
                                                              verbose=False,
                                                              from_memory=observation.copy())
                    # cost = sum(v.cost for v in values)
                    results[method] += states[-1].t-observation.t # TODO, this is a trick...
                    actions_map[method] = actions[0]
            if self.verbose:
                # print(results)
                print(actions_map[min(results, key=lambda x: results[x])])
            return actions_map[min(results, key=lambda x: results[x])]
        else:
            return actions_c[0]

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True
