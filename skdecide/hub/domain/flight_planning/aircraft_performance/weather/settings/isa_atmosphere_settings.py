from dataclasses import dataclass

from skdecide.hub.domain.flight_planning.aircraft_performance.weather.atmosphere_model_enum import (
    AtmosphereModelEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.settings.atmosphere_settings_interface import (
    AtmosphereSettings,
)


@dataclass
class IsaAtmosphereSettings(AtmosphereSettings):
    tropopause_alt_ft: float = None  # default tropopause ft
    d_isa: float = 0
    tailwind_m_per_sec: float = 0

    def __post_init__(self):
        self.atmosphere_model_type = AtmosphereModelEnum.ISA
