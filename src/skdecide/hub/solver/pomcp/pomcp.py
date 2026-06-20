# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Optional

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    IntegerHyperparameter,
)

from skdecide import DiscreteDistribution, Distribution, Domain, Memory, Solver
from skdecide.builders.domain import (
    Actions,
    EnumerableTransitions,
    Goals,
    Markovian,
    PartiallyObservable,
    PositiveCosts,
    Rewards,
    Sequential,
    SingleAgent,
    UncertainInitialized,
    UncertainTransitions,
)
from skdecide.builders.solver import (
    DeterministicPolicies,
    FromAnyState,
    ParallelSolver,
    Utilities,
)

# --- C++ POMCP solver (primary) ---

try:
    from skdecide.hub.__skdecide_hub_cpp import _POMCPSolver_ as pomcp_solver

    class D(
        Domain,
        SingleAgent,
        Sequential,
        UncertainTransitions,
        Actions,
        Markovian,
        PartiallyObservable,
        Rewards,
        UncertainInitialized,
    ):
        pass

    class POMCP(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """POMCP solver for POMDPs (online, reward maximization).

        From: Silver & Veness, "Monte-Carlo Planning in Large POMDPs",
        NIPS 2010.

        POMCP applies UCT to a history tree with particle-based belief
        tracking. Planning happens online at each step: the solver samples
        particles from the current belief, runs Monte Carlo simulations
        through the history tree, and selects actions via UCB1.

        The default interface works with observations. The solver internally
        maintains and updates the current belief using a particle filter.
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            exploration_constant: float = 1.0 / math.sqrt(2.0),
            discount: float = 0.95,
            num_simulations: int = 1000,
            max_depth: int = 100,
            epsilon: float = 0.001,
            time_budget: int = 0,
            num_particles_belief_update: int = 500,
            ess_threshold_ratio: float = 2.0,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[POMCP, Domain], bool] = lambda slv, dom: False,
            verbose: bool = False,
        ) -> None:
            """Construct a POMCP solver instance.

            # Parameters
            domain_factory: Lambda function to create a domain instance.
            exploration_constant: UCB1 exploration constant (c in the paper).
                Defaults to 1/sqrt(2).
            discount: Discount factor gamma. Must be in (0, 1].
                Defaults to 0.95.
            num_simulations: Number of Monte Carlo simulations per planning
                step. Defaults to 1000.
            max_depth: Maximum search/rollout depth. Defaults to 100.
            epsilon: Discount-depth cutoff threshold. A simulation stops when
                gamma^depth < epsilon. Defaults to 0.001.
            time_budget: Maximum planning time per step in milliseconds.
                0 means no time limit. Defaults to 0.
            num_particles_belief_update: Number of particles for belief
                update via particle filter. Defaults to 500.
            ess_threshold_ratio: Effective sample size threshold for
                resampling. Resampling occurs when ESS < N / ratio.
                Defaults to 2.0.
            parallel: Parallelize domain calls. Defaults to False.
            shared_memory_proxy: Optional shared memory proxy.
                Defaults to None.
            callback: Function called at each simulation iteration, taking
                the solver and domain as arguments, returning True to stop.
                Defaults to never stop.
            verbose: Whether to log verbose messages. Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._ipc_notify = True

            self._solver = pomcp_solver(
                solver=self,
                domain=self.get_domain(),
                exploration_constant=exploration_constant,
                discount=discount,
                num_simulations=num_simulations,
                max_depth=max_depth,
                epsilon=epsilon,
                time_budget=time_budget,
                num_particles_belief_update=num_particles_belief_update,
                ess_threshold_ratio=ess_threshold_ratio,
                parallel=parallel,
                callback=callback,
                verbose=verbose,
            )

        def close(self):
            """Joins the parallel domains' processes."""
            if self._parallel:
                self._solver.close()
            ParallelSolver.close(self)

        def _solve(self, from_memory=None) -> None:
            if from_memory is None:
                from_memory = self._domain_factory().get_initial_state_distribution()
            self._solve_from(from_memory)

        def _solve_from(self, initial_belief: Distribution[D.T_state]) -> None:
            """Initialize POMCP with an initial belief distribution.

            For POMCP (an online solver), this only initializes the belief
            particles. Actual planning happens online in _get_next_action().

            # Parameters
            initial_belief: Distribution over physical states representing
                the initial belief.
            """
            self._solver.solve(initial_belief)

        def _is_solution_defined_for(
            self, observation: D.T_agent[D.T_observation]
        ) -> bool:
            return self._solver.is_solution_defined_for(observation)

        def _get_next_action(
            self,
            observation: D.T_agent[D.T_observation],
            domain: Optional[Domain] = None,
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            """Get the best action given an observation.

            The solver updates its belief via particle filter, prunes the
            history tree to the subtree under (last_action, observation),
            then runs Monte Carlo simulations from the current belief.
            """
            action = self._solver.get_next_action(observation)
            if action is None:
                print(
                    "\x1b[3;33;40m"
                    + "No best action found for observation "
                    + str(observation)
                    + ", applying random action"
                    + "\x1b[0m"
                )
                return self.call_domain_method("get_action_space").sample()
            else:
                return action

        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            return self._solver.get_utility(observation)

        def get_next_action_from_belief(
            self, belief: Distribution[D.T_state]
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            """Get the best action for an explicit belief state."""
            action = self._solver.get_next_action_from_belief(belief)
            if action is None:
                print(
                    "\x1b[3;33;40m"
                    + "No best action found for belief, applying random action"
                    + "\x1b[0m"
                )
                return self.call_domain_method("get_action_space").sample()
            return action

        def get_utility_from_belief(
            self, belief: Distribution[D.T_state]
        ) -> Value[D.T_value]:
            """Get the best value for an explicit belief state."""
            return self._solver.get_utility_from_belief(belief)

        def is_solution_defined_for_from_belief(
            self, belief: Distribution[D.T_state]
        ) -> bool:
            """Check if a solution is defined for an explicit belief state."""
            return self._solver.is_solution_defined_for_from_belief(belief)

        def reset_belief(self) -> None:
            """Reset the tracked belief to the initial belief from solve()."""
            self._solver.reset_belief()

        def get_nb_tree_nodes(self) -> int:
            """Get the number of nodes in the last history tree."""
            return self._solver.get_nb_tree_nodes()

        def get_solving_time(self) -> int:
            """Get the last planning time in milliseconds."""
            return self._solver.get_solving_time()

        def get_last_trajectory(
            self,
        ) -> list[tuple[D.T_observation, D.T_agent[D.T_concurrency[D.T_event]]]]:
            """Get the ordered list of (observation, action) pairs visited during
            the last POMCP simulation.

            Returns the trajectory (path) explored during the most recent simulation
            from the root history node. Each element is a tuple of (observation,
            action) where the observation is the observation made in that history
            state and the action is the action selected via UCB1. The trajectory
            begins with the root observation and ends at the deepest history node
            reached before the simulation terminated (due to terminal state, depth
            limit, or discount cutoff).

            Note: POMCP operates in observation/history space, not state space,
            so the trajectory reflects the observable history, not the underlying
            states.

            # Returns
            list[tuple[D.T_observation, D.T_agent[D.T_concurrency[D.T_event]]]]: List of
                (observation, action) pairs visited during the last simulation.
                Returns an empty list if solve() has not been called yet.
            """

            return self._solver.get_last_trajectory()

except ImportError:
    print(
        "Scikit-decide C++ hub library not found. Please check it is installed "
        'in "skdecide/hub".'
    )
    raise


# --- Pure Python POMCP solver (renamed from original POMCP) ---

# Original code by Patrik Haslum, based on POMCP from:
# Silver, D., & Veness, J. (2010). Monte-Carlo Planning in Large POMDPs.
# In Advances in neural information processing systems (pp. 2164–2172).


class Dp(
    Domain,
    SingleAgent,
    Sequential,
    EnumerableTransitions,
    Actions,
    Goals,
    PartiallyObservable,
    PositiveCosts,
    UncertainInitialized,
):
    pass


class pPOMCP(Solver, DeterministicPolicies):
    """Partially-Observable Monte Carlo Planning solver (pure Python, cost minimization, goal-based).

    Pure Python implementation by Patrik Haslum, based on POMCP from:
    Silver, D., & Veness, J. (2010). Monte-Carlo Planning in Large POMDPs.
    In Advances in neural information processing systems (pp. 2164-2172).

    For the C++ implementation (reward maximization, observation-based), use POMCP.
    """

    T_domain = Dp

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
        callback: Callable[[pPOMCP], bool] = lambda solver: False,
    ) -> None:
        """Construct a pPOMCP solver instance (pure Python, cost minimization).

        # Parameters
        domain_factory: Lambda function to create a domain instance.
        max_iterations: Maximum number of UCT iterations per planning step.
            Defaults to 5000.
        max_depth: Maximum search and rollout depth. Defaults to 50.
        n_samples: Number of state samples drawn from the initial state
            distribution to form the initial belief. Defaults to 5000.
        callback: Function called at each solver iteration. If returning
            True, the solve process stops. Defaults to never stop.
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
        self, observation: Dp.T_agent[Dp.T_observation], domain: Optional[Domain] = None
    ) -> Dp.T_agent[Dp.T_concurrency[Dp.T_event]]:
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

    def _is_policy_defined_for(self, observation: Dp.T_agent[Dp.T_observation]) -> bool:
        return True

    def _filter_belief_state(self, belief, action, obs):
        prob = [0] * len(belief)
        for i, state in enumerate(belief):
            d = self._domain.get_observation_distribution(state, action)
            prob[i] = _get_probability(d, obs)
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


def _get_probability(distribution, element, n=100):
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
