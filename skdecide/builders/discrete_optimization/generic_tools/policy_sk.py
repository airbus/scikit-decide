from skdecide.builders.solver.policy import DeterministicPolicies
from skdecide import Domain
from skdecide.domains import SingleAgent, Sequential, EnumerableTransitions,\
    Actions, Goals, Markovian, FullyObservable, PositiveCosts, GoalMDPDomain, DeterministicInitialized
from typing import Dict


class D(Domain, SingleAgent, Sequential, EnumerableTransitions, Actions,
        Goals, Markovian, FullyObservable,
        PositiveCosts, DeterministicInitialized):
    pass

class LookUpPolicy(DeterministicPolicies):
    """ A solver which is initialized with a stochastic policy but provides
    interfaces for deterministic policies.
    """
    T_domain = D
    def __init__(self, policy_dict):
        self.policy_dict = policy_dict
    
    def reset(self):
        pass

    def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        return self.policy_dict.get(observation, None)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return observation in self.policy_dict

    
