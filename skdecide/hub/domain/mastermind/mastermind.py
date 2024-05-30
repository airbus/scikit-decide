# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Original code by Patrik Haslum
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

from skdecide import DiscreteDistribution, Distribution, GoalPOMDPDomain, Space, Value
from skdecide.builders.domain import (
    DeterministicTransitions,
    TransformedObservable,
    UnrestrictedActions,
)
from skdecide.hub.space.gym import ListSpace, MultiDiscreteSpace

Row = Tuple[int]  # a row of code pegs (solution or guess)


class Score(NamedTuple):
    total_bulls: int
    total_cows: int


class State(NamedTuple):
    solution: Row
    score: Score


class D(
    GoalPOMDPDomain,
    DeterministicTransitions,
    UnrestrictedActions,
    TransformedObservable,
):
    T_state = State  # Type of states
    T_observation = Score  # Type of observations
    T_event = Row  # Type of events (a row guess in this case)
    T_value = int  # Type of transition values (costs)
    T_predicate = bool  # Type of logical checks
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class MasterMind(D):
    def __init__(self, n_colours=2, n_positions=2):
        self._n_colours = n_colours
        self._n_positions = n_positions
        self._h_solutions = self._list_hidden_solutions()

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        # Input is a state and an action; output is a next state.
        if (
            action is None
        ):  # TODO: handle this option on algo side rather than domain; here action should never be None
            return memory
        else:
            return State(memory.solution, self._calc_score(memory, action))

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        return Value(cost=1)

    # Overridden to help some solvers compute more efficiently (not mandatory, but good practice)
    def _is_transition_value_dependent_on_next_state_(self) -> bool:
        return False

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        return self._is_goal(state.score)

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        # Return the possible actions (guesses) as an enumerable space
        return ListSpace(self._h_solutions)

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        # Return the space of goal OBSERVATIONS
        return ListSpace([Score(total_bulls=self._n_positions, total_cows=0)])

    def _get_initial_state_distribution_(self) -> Distribution[D.T_state]:
        # Return a uniform distribution over all initial states
        n = len(self._h_solutions)
        return DiscreteDistribution(
            [(State(solution=s, score=Score(0, 0)), 1 / n) for s in self._h_solutions]
        )

    def _get_observation(
        self,
        state: D.T_state,
        action: Optional[D.T_agent[D.T_concurrency[D.T_event]]] = None,
    ) -> D.T_agent[D.T_observation]:
        # `action` is the last applied action (or None if the state is an initial state)
        # `state` is the state to observe (that resulted from applying the action)
        if action is None:
            return Score(0, 0)
        return self._calc_score(state, action)

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return MultiDiscreteSpace(
            nvec=[self._n_positions + 1, self._n_positions + 1], element_class=Score
        )

    def _list_hidden_solutions(self):
        """Return a list of all possible hidden solutions (n_colours ** n_positions)."""
        h_solutions = [tuple()]
        for i in range(self._n_positions):
            h_solutions = [
                s + (c,) for s in h_solutions for c in range(self._n_colours)
            ]
        return h_solutions

    def _calc_score(self, state, guess):
        """Compute the score of a guess."""
        solution = state.solution
        bulls = [False for _ in range(len(guess))]
        for i in range(len(guess)):
            if guess[i] == solution[i]:
                bulls[i] = True
        cows = [False for _ in range(len(guess))]
        for i in range(len(guess)):
            if guess[i] != solution[i]:
                for j in range(len(guess)):
                    if guess[i] == solution[j] and not bulls[j] and not cows[j]:
                        cows[j] = True
                        break
        return Score(total_bulls=sum(bulls), total_cows=sum(cows))
