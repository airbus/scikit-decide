import math

import numpy as np

import skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.std_atm as std_atm
from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
    AircraftState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.service.speed_conversion_service import (
    SpeedConversionService,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.settings.aerodynamics_settings_interface import (
    AerodynamicsSettings,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.settings.poll_schumann_aerodynamics_settings import (
    PollSchumannAerodynamicsSettings,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.utils.poll_schumann_utils.aircraft_parameters import (
    crossover_pressure_altitude,
    impact_pressure_max_operating_limits,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.utils.poll_schumann_utils.engine_loader import (
    load_aircraft_engine_params,
)


class _PollSchumannAerodynamicsService(AerodynamicsSettings):
    performance_model_type = PerformanceModelEnum.POLL_SCHUMANN

    speedconversion_service = SpeedConversionService()

    def init_settings(self, model_path: str) -> PollSchumannAerodynamicsSettings:
        ac_parameters = load_aircraft_engine_params(model_path)

        return PollSchumannAerodynamicsSettings(
            ac_parameters=ac_parameters, sref=ac_parameters["wing_surface_area"]
        )

    def compute_drag_coefficient(
        self,
        aerodynamics_settings: PollSchumannAerodynamicsSettings,
        aircraft_state: AircraftState,
    ) -> float:
        reynolds = self._compute_reynolds(aerodynamics_settings, aircraft_state)
        cf = self._compute_skin_friction_coefficient(reynolds=reynolds)
        cd_0 = self._compute_zero_lift_drag_coefficient(
            aerodynamics_settings=aerodynamics_settings, cf=cf
        )
        e_ls = self._compute_oswald_efficiency_factor(
            aerodynamics_settings=aerodynamics_settings, cd_0=cd_0
        )
        cd_wave = self._compute_wave_drag_coefficient(
            aerodynamics_settings=aerodynamics_settings, aircraft_state=aircraft_state
        )

        return (
            cd_0
            + cd_wave
            + aircraft_state.cl**2
            / (
                math.pi
                * e_ls
                * aerodynamics_settings.ac_parameters["wing_aspect_ratio"]
            )
        )

    def _compute_reynolds(
        self,
        aerodynamics_settings: PollSchumannAerodynamicsSettings,
        aircraft_state: AircraftState,
    ) -> float:
        gamma = 1.4
        R = 287.05

        mu_inf = (
            1.458e-6
            * (aircraft_state.weather_state.static_temperature_k**1.5)
            / (110.4 + aircraft_state.weather_state.static_temperature_k)
        )
        return (
            aerodynamics_settings.ac_parameters["wing_surface_area"] ** 0.5
            * aircraft_state.mach
            * (aircraft_state.weather_state.static_pressure_pa / mu_inf)
            * (gamma / (R * aircraft_state.weather_state.static_temperature_k)) ** 0.5
        )

    def _compute_skin_friction_coefficient(self, reynolds: float) -> float:
        return 0.0269 / (reynolds**0.14)

    def _compute_zero_lift_drag_coefficient(
        self, aerodynamics_settings: PollSchumannAerodynamicsSettings, cf: float
    ) -> float:
        return cf * aerodynamics_settings.ac_parameters["psi_0"]

    def _compute_oswald_efficiency_factor(
        self, aerodynamics_settings: PollSchumannAerodynamicsSettings, cd_0: float
    ) -> float:
        numer = 1.0

        k1 = 0.80 * (1 - 0.53 * aerodynamics_settings.ac_parameters["cos_sweep"]) * cd_0
        denom = (
            1.0
            + 0.03
            + aerodynamics_settings.ac_parameters["delta_2"]
            + math.pi * k1 * aerodynamics_settings.ac_parameters["wing_aspect_ratio"]
        )

        return numer / denom

    def _compute_wave_drag_coefficient(
        self,
        aerodynamics_settings: PollSchumannAerodynamicsSettings,
        aircraft_state: AircraftState,
    ) -> float:
        mach = aircraft_state.mach
        cl = aircraft_state.cl

        m_cc = aerodynamics_settings.ac_parameters["wing_constant"] - 0.10 * (
            cl / aerodynamics_settings.ac_parameters["cos_sweep"] ** 2
        )
        x = mach * aerodynamics_settings.ac_parameters["cos_sweep"] / m_cc

        c_d_w = np.where(
            x < aerodynamics_settings.ac_parameters["j_2"],
            0.0,
            aerodynamics_settings.ac_parameters["cos_sweep"] ** 3
            * aerodynamics_settings.ac_parameters["j_1"]
            * (x - aerodynamics_settings.ac_parameters["j_2"]) ** 2,
        )

        output = np.where(
            x < aerodynamics_settings.ac_parameters["x_ref"],
            c_d_w,
            c_d_w
            + aerodynamics_settings.ac_parameters["j_3"]
            * (x - aerodynamics_settings.ac_parameters["x_ref"]) ** 4,
        )

        return output

    def compute_crossover(
        self,
        aerodynamics_settings: PollSchumannAerodynamicsSettings,
        aircraft_state: AircraftState,
    ) -> float:
        return crossover_pressure_altitude(
            max_mach_num=aircraft_state.mach_cruise,
            p_i_max=impact_pressure_max_operating_limits(
                max_mach_num=aircraft_state.mach_cruise
            ),
        )
