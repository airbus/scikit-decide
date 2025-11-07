from enum import Enum


class PhaseEnum(Enum):
    """
    Enum for phase
    """

    TAKE_OFF = 1
    CLIMB = 2
    CRUISE = 3
    DESCENT = 4
