# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Any, Dict

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    IntegerHyperparameter,
)

from skdecide import rollout
from skdecide.builders.domain.scheduling.scheduling_domains import D, SchedulingDomain
from skdecide.builders.solver import DeterministicPolicies

logger = logging.getLogger(__name__)


class MetaPolicy(DeterministicPolicies):
    """
    Utility policy function that represents a meta policy :
    At a given state, it launches a rollout for each policy to evaluate each of them.
    Then the policy for the given state is obtained with the policy that is giving the lowest estimated cost.
    """

    T_domain = D

    hyperparameters = [
        IntegerHyperparameter(name="nb_rollout_estimation"),
    ]

    def __init__(
        self,
        policies: Dict[Any, DeterministicPolicies],
        domain: SchedulingDomain,
        nb_rollout_estimation=1,
        verbose=True,
    ):
        """
        # Parameters
        policies: dictionaries of different policies to evaluate
        domain: domain on which to evaluate the policies
        nb_rollout_estimation: relevant if the domain is stochastic,
        run nb_rollout_estimation time(s) the rollout to estimate the expected cost of the policy.

        """
        self.domain = domain
        self.domain.fast = True
        self.policies = policies
        self.current_states = {method: None for method in policies}
        self.nb_rollout_estimation = nb_rollout_estimation
        self.verbose = verbose

    def reset(self):
        self.current_states = {method: None for method in self.policies}

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        results = {}
        actions_map = {}
        self.domain.set_inplace_environment(True)
        actions_c = [
            self.policies[method].get_next_action(observation)
            for method in self.policies
        ]
        if len(set(actions_c)) > 1:
            for method in self.policies:
                results[method] = 0.0
                for j in range(self.nb_rollout_estimation):
                    states, actions, values = rollout(
                        domain=self.domain,
                        solver=self.policies[method],
                        outcome_formatter=None,
                        action_formatter=None,
                        verbose=False,
                        goal_logging_level=logging.DEBUG,
                        from_memory=observation.copy(),
                        return_episodes=True,
                    )[0]
                    results[method] += states[-1].t - observation.t
                    actions_map[method] = actions[0]
            if self.verbose:
                logger.debug(f"{actions_map[min(results, key=lambda x: results[x])]}")
            return actions_map[min(results, key=lambda x: results[x])]
        else:
            return actions_c[0]

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True
