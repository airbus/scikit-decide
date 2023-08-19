# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unified_planning.shortcuts import Problem, ProblemKind

from skdecide import Domain, ImplicitSpace, Space, TransitionOutcome, Value
from skdecide.builders.domain import (
    DeterministicInitialized,
    DeterministicTransitions,
    FullyObservable,
    Goals,
    Initializable,
    Markovian,
    Memoryless,
    PositiveCosts,
    Sequential,
    SingleAgent,
    UnrestrictedActions,
)
from skdecide.hub.space.gym import GymSpace, ListSpace

class D(
    Domain,
    SingleAgent,
    Sequential,
    UnrestrictedActions,
    Initializable,
    Memoryless,
    FullyObservable,
    PositiveCosts,
):
    pass


class UPDomainFactory:
    """This class wraps a Unified Planning's problem (up.model.Problem) as a scikit-decide domain.
    Depending on the problem kind (up.model.ProblemKind) of the UP problem, this class will create
    a domain with specific features that corresond to the actual ProblemKind

    !!! warning
        Using this class requires unified-planning to be installed.
    """
    
    def __init__(self, up_problem: Problem) -> None:
        pass