from enum import Enum

from skdecide.hub.domain.flight_planning.aircraft_performance.performance.phase_enum import (
    PhaseEnum,
)


class RatingEnum(Enum):
    """
    Enum for phase
    """

    TO = {"phase": PhaseEnum.TAKE_OFF}
    MCL = {"phase": PhaseEnum.CLIMB}
    CR = {"phase": PhaseEnum.CRUISE}
    IDLE = {"phase": None}
