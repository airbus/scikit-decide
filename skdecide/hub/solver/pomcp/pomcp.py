# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Original code by Patrik Haslum, based on POMCP from:
# Silver, D., & Veness, J. (2010). Monte-Carlo Planning in Large POMDPs.
# In Advances in neural information processing systems (pp. 2164–2172).
from __future__ import annotations

import math
import random
from typing import Callable

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    IntegerHyperparameter,
)

from skdecide import DiscreteDistribution, Domain, Memory, Solver
from skdecide.builders.domain import (
    Actions,
    EnumerableTransitions,
    Goals,
    Sequential,
    SingleAgent,
    UncertainInitialized,
)
from skdecide.builders.solver import DeterministicPolicies


class D(
    Domain,
    SingleAgent,
    Sequential,
    EnumerableTransitions,
    Actions,
    Goals,
    UncertainInitialized,
):
    pass


class POMCP(Solver, DeterministicPolicies):
    """Partially-Observable Monte Carlo Planning solver."""

    T_domain = D

    hyperparameters = [
        IntegerHyperparameter(name="max_iterations"),
        IntegerHyperparameter(name="max_depth"),
        IntegerHyperparameter(name="n_samples"),
    ]

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        max_iterations=5000,
        max_depth=50,
        n_samples=5000,
        callback: Callable[[POMCP], bool] = lambda solver: False,
    ) -> None:
        """

        # Parameters
        domain_factory
        max_iterations
        max_depth
        n_samples
        callback: function called at each solver iteration. If returning true, the solve process stops.

        """
        self.callback = callback
        Solver.__init__(self, domain_factory=domain_factory)
        self._max_iterations = max_iterations
        self._max_depth = max_depth
        self._n_samples = n_samples

    def _reset(self) -> None:
        # Reset whatever is needed on this solver before running a new episode
        self._obs_history = tuple()
        self._act_history = (None,)
        self._belief = self._initial_belief
        self._tree = dict()
        # VLV is the Very Large Value; this is supposed to be a value that
        # represents "infinite" cost (i.e., goal not reached within depth
        # bound). The approximation 2 * max_depth is ok if all actions have
        # cost 1. In general, there seems to be no way to query the domain
        # for the range of possible cost values.
        self._VLV = 100 * self._max_depth

    def _solve(self) -> None:
        self._domain = self._domain_factory()
        self._initial_belief = []
        d = self._domain.get_initial_state_distribution()
        for _ in range(self._n_samples):
            self._initial_belief.append(d.sample())
        # No further solving code required here since everything is computed online

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        # Get the next action from the solver's current policy:
        # this corresponds to the top-level Search procedure in the POMCP paper

        # Since we have now received a new observation, update our
        # belief state with this information; note that obs may
        # depend on the last action taken:
        self._belief = self._filter_belief_state(
            self._belief, self._act_history[-1], observation
        )

        # Record the added observation:
        self._obs_history = self._obs_history + (observation,)

        # Then, update each state in the filtered belief with the
        # effects of the last action taken:
        self._belief = self._update_belief_state(self._belief, self._act_history[-1])

        # Now, we can make a decision from the new belief state:
        iterations = 0
        while iterations < self._max_iterations and not self.callback(
            self
        ):  # or some other cut-off
            # sample a state from the current belief
            state = random.choice(self._belief)
            self._tree_search(state, self._act_history, self._obs_history, 0)
            iterations += 1

        # Select the best action from the successors of the current node:
        action = self._get_best_action(self._act_history, self._obs_history)

        # Record the last action, and then return it:
        self._act_history = self._act_history + (action,)

        return action

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _filter_belief_state(self, belief, action, obs):
        prob = [0] * len(belief)
        for i, state in enumerate(belief):
            d = self._domain.get_observation_distribution(state, action)
            prob[i] = get_probability(d, obs)
        new_belief = random.choices(belief, weights=prob, k=len(belief))
        return new_belief

    def _update_belief_state(self, belief, action):
        new_belief = []
        for state in belief:
            d = (
                self._domain.get_next_state_distribution(Memory([state]), action)
                if action is not None
                else self._domain.get_initial_state_distribution()
            )
            new_state = d.sample()
            new_belief.append(new_state)
        return new_belief

    def _get_best_action(self, h_act, h_obs, w=0):
        """Retrieve best action at (h_act, h_obs) from stored tree.

        If w > 0, best is determined using the UCT formula with weight w;
        else it's just the action with min expected cost.
        """
        best_action = None
        best_action_score = self._VLV
        if w > 0:
            parent = self._tree[(h_act, h_obs)]
        for action in self._domain.get_action_space().get_elements():
            key = (h_act + (action,), h_obs)
            if key in self._tree:
                node = self._tree[key]
                # node[0] is visit count (N); node[1] is average cost (V)
                if w > 0:
                    if node[0] == 0:
                        score = -self._VLV
                    else:
                        score = node[1] - (w * math.sqrt(math.log(parent[0]) / node[0]))
                else:
                    score = node[1]
                if score <= best_action_score:
                    best_action = action
                    best_action_score = score
        return best_action

    def _tree_search(self, state, h_act, h_obs, depth):
        """UCT search from a given state with act/obs history.

        This corresponds to the Simulate function in the POMCP paper.
        """
        # This must be a history that ends on an observation
        assert len(h_act) == len(h_obs)
        if depth > self._max_depth:
            return self._VLV
        if (h_act, h_obs) not in self._tree:
            # generate new child nodes
            for action in self._domain.get_applicable_actions(
                Memory([state])
            ).get_elements():
                assert action is not None
                self._tree[(h_act + (action,), h_obs)] = [0, 0, []]
            # but we must also store this node, or we'll never get out of this case!
            cost = self._rollout(state, h_act, h_obs, depth)
            self._tree[(h_act, h_obs)] = [1, cost, [state]]
            return cost
        else:
            # pick a successor node according to the UCT formula
            action = self._get_best_action(h_act, h_obs, w=self._max_depth)
            assert action is not None
            # simulate outcome of this action:
            new_state = self._domain.get_next_state_distribution(
                Memory([state]), action
            ).sample()
            TV = self._domain.get_transition_value(Memory([state]), action, new_state)
            new_obs = self._domain.get_observation_distribution(state, action).sample()
            if self._domain.is_goal(new_obs):
                s_cost = TV.cost
            else:
                s_cost = TV.cost + self._tree_search(
                    new_state, h_act + (action,), h_obs + (new_obs,), depth + 1
                )
                s_cost = min(s_cost, self._VLV)
            this_node = self._tree[(h_act, h_obs)]
            succ_node = self._tree[(h_act + (action,), h_obs)]
            # update average cost for succ node:
            succ_node[1] = ((succ_node[1] * succ_node[0]) + s_cost) / (succ_node[0] + 1)
            # increment visit counters for both this node and succ node:
            this_node[0] = this_node[0] + 1
            succ_node[0] = succ_node[0] + 1
            return s_cost

    def _rollout(self, state, h_act, h_obs, depth):
        if depth > self._max_depth:
            return self._VLV
        action = self._get_random_action(state, h_act, h_obs, depth)
        assert action is not None
        new_state = self._domain.get_next_state_distribution(
            Memory([state]), action
        ).sample()
        TV = self._domain.get_transition_value(Memory([state]), action, new_state)
        new_obs = self._domain.get_observation_distribution(state, action).sample()
        if self._domain.is_goal(new_obs):
            s_cost = TV.cost
        else:
            s_cost = TV.cost + self._rollout(
                new_state, h_act + (action,), h_obs + (new_obs,), depth + 1
            )
            s_cost = min(s_cost, self._VLV)
        return s_cost

    def _get_random_action(self, state, h_act, h_obs, depth):
        sel = self._domain.get_applicable_actions(Memory([state])).sample()
        return sel


def get_probability(distribution, element, n=100):
    """Utility function to get the probability of a specific element from a scikit-decide distribution
    (based on sampling if this distribution is not a DiscreteDistribution)."""

    # Avoid "dumb" sampling if the distribution is a DiscreteDistribution:
    if isinstance(distribution, DiscreteDistribution):
        return next((p for e, p in distribution.get_values() if e == element), 0.0)
    else:
        p = 0
        for i in range(n):
            x = distribution.sample()
            if x == element:
                p += 1
        return p / n
