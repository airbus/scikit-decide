from abc import ABC
from dataclasses import dataclass

from skdecide.hub.domain.flight_planning.aircraft_performance.weather.atmosphere_model_enum import (
    AtmosphereModelEnum,
)


@dataclass
class AtmosphereSettings(ABC):
    """
    Abstract dataclass for aerodynamics service settings
    """

    atmosphere_model_type: AtmosphereModelEnum = None
