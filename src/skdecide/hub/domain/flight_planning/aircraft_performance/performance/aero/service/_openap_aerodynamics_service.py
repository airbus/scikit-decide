import math

import numpy as np
from openap.drag import Drag
from openap.extra.aero import T0, R, a0, beta, ft, g0, kts

from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
    AircraftState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.service.speed_conversion_service import (
    SpeedConversionService,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.settings.aerodynamics_settings_interface import (
    AerodynamicsSettings,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.settings.openap_aerodynamics_settings import (
    OpenapAerodynamicsSettings,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.service.atmosphere_service import (
    AtmosphereService,
)


class _OpenapAerodynamicsService(AerodynamicsSettings):
    performance_model_type = PerformanceModelEnum.OPENAP

    speedconversion_service = SpeedConversionService()
    atmosphere_service = AtmosphereService()

    def init_settings(self, model_path: str) -> OpenapAerodynamicsSettings:
        drag = Drag(ac=model_path)

        return OpenapAerodynamicsSettings(drag=drag, sref=drag.aircraft["wing"]["area"])

    def compute_drag_coefficient(
        self,
        aerodynamics_settings: OpenapAerodynamicsSettings,
        aircraft_state: AircraftState,
    ) -> float:
        cd0 = aerodynamics_settings.drag.polar["clean"]["cd0"]

        sweep = math.radians(aerodynamics_settings.drag.aircraft["wing"]["sweep"])
        tc = aerodynamics_settings.drag.aircraft["wing"]["t/c"]
        if tc is None:
            tc = 0.11

        cos_sweep = math.cos(sweep)
        mach_crit = (
            0.87
            - 0.108 / cos_sweep
            - 0.1 * aircraft_state.cl / (cos_sweep**2)
            - tc / cos_sweep
        ) / cos_sweep

        dmach = np.where(
            aircraft_state.mach - mach_crit <= 0, 0, aircraft_state.mach - mach_crit
        )

        dCdw = np.where(dmach, 20 * dmach**4, 0)

        return cd0 + dCdw

    def compute_crossover(
        self,
        aerodynamics_settings: OpenapAerodynamicsSettings,
        aircraft_state: AircraftState,
    ) -> float:
        v_cas = aircraft_state.cas_climb_kts * kts
        delta = ((0.2 * (v_cas / a0) ** 2 + 1) ** 3.5 - 1) / (
            (0.2 * aircraft_state.mach_cruise**2 + 1) ** 3.5 - 1
        )
        h = T0 / beta * (delta ** (-1 * R * beta / g0) - 1)
        return h / ft
