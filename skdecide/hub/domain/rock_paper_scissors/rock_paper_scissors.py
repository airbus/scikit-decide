# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Optional

from skdecide import Domain, Space, TransitionOutcome, Value
from skdecide.builders.domain import *
from skdecide.hub.space.gym import EnumSpace


class Move(Enum):
    rock = 0
    paper = 1
    scissors = 2


class State(NamedTuple):
    num_move: int


class D(
    Domain,
    MultiAgent,
    Sequential,
    Environment,
    UnrestrictedActions,
    Initializable,
    Markovian,
    TransformedObservable,
    Rewards,
):
    T_state = State  # Type of states
    T_observation = Move  # Type of observations
    T_event = Move  # Type of events
    T_value = int  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of logical checks
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class RockPaperScissors(D):
    def __init__(self, max_moves: int = 10):
        self._max_moves = max_moves

    def get_agents(self):
        return {"player1", "player2"}

    def _state_step(
        self, action: D.T_agent[D.T_concurrency[D.T_event]]
    ) -> TransitionOutcome[
        D.T_state,
        D.T_agent[Value[D.T_value]],
        D.T_agent[D.T_predicate],
        D.T_agent[D.T_info],
    ]:

        # Get players' moves
        move1, move2 = action["player1"], action["player2"]

        # Compute rewards
        r1, r2 = {
            (Move.rock, Move.rock): (0, 0),
            (Move.rock, Move.paper): (-1, 1),
            (Move.rock, Move.scissors): (1, -1),
            (Move.paper, Move.rock): (1, -1),
            (Move.paper, Move.paper): (0, 0),
            (Move.paper, Move.scissors): (-1, 1),
            (Move.scissors, Move.rock): (-1, 1),
            (Move.scissors, Move.paper): (1, -1),
            (Move.scissors, Move.scissors): (0, 0),
        }[move1, move2]

        # Compute num_move increment
        last_state = self._memory
        num_move = last_state.num_move + 1

        return TransitionOutcome(
            state=State(num_move=num_move),
            value={"player1": Value(reward=r1), "player2": Value(reward=r2)},
            termination={k: (num_move >= self._max_moves) for k in self.get_agents()},
        )

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return {"player1": EnumSpace(Move), "player2": EnumSpace(Move)}

    def _state_reset(self) -> D.T_state:
        return State(num_move=0)

    def _get_observation(
        self,
        state: D.T_state,
        action: Optional[D.T_agent[D.T_concurrency[D.T_event]]] = None,
    ) -> D.T_agent[D.T_observation]:
        # The observation is simply the last opponent move (or Move.rock initially by default)
        obs1 = action["player2"] if action is not None else Move.rock
        obs2 = action["player1"] if action is not None else Move.rock
        return {"player1": obs1, "player2": obs2}

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return {"player1": EnumSpace(Move), "player2": EnumSpace(Move)}
