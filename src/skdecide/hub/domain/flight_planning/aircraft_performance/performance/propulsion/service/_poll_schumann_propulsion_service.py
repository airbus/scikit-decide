import math

import numpy as np

from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
    AircraftState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.service._poll_schumann_aerodynamics_service import (
    _PollSchumannAerodynamicsService,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.settings.poll_schumann_aerodynamics_settings import (
    PollSchumannAerodynamicsSettings,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.propulsion.service.propulsion_service_interface import (
    PropulsionServiceInterface,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.propulsion.settings.poll_schumann_propulsion_settings import (
    PollSchumannPropulsionSettings,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.utils.poll_schumann_utils.engine_loader import (
    load_aircraft_engine_params,
)


class _PollSchumannPropulsionService(PropulsionServiceInterface):
    """
    Implementation of propulsion servisce for OpenAP models
    """

    performance_model_type = PerformanceModelEnum.POLL_SCHUMANN
    aerodynamics_service = _PollSchumannAerodynamicsService()

    def init_settings(self, model_path: str) -> PollSchumannPropulsionSettings:
        ac_parameters = load_aircraft_engine_params(model_path)

        self.aerodynamics_settings = PollSchumannAerodynamicsSettings(
            PerformanceModelEnum.POLL_SCHUMANN, ac_parameters=ac_parameters
        )

        return PollSchumannPropulsionSettings(
            ac_parameters=ac_parameters, sref=ac_parameters["wing_surface_area"]
        )

    def compute_total_fuel_flow_kg_per_sec(
        self,
        propulsion_settings: PollSchumannPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        ff_min = self._fuel_flow_idle(propulsion_settings, aircraft_state)
        ff_max = self._equivalent_fuel_flow_rate_at_cruise(
            propulsion_settings, aircraft_state
        )

        ff = self._fuel_mass_flow_rate(propulsion_settings, aircraft_state)
        if aircraft_state.phase.name == "DESCENT":
            ff_max = 0.3 * propulsion_settings.ac_parameters["ff_max_sls"]

        return np.clip(ff, ff_min, ff_max)

    def compute_total_net_thrust_n(
        self,
        propulsion_settings: PollSchumannPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        theta_rad = math.radians(aircraft_state.gamma_air_deg)
        # TODO: Implement acceleration
        dv_dt = 0.0

        cd = self.aerodynamics_service.compute_drag_coefficient(
            self.aerodynamics_settings, aircraft_state
        )

        f_thrust = (
            (
                aircraft_state.gw_kg
                * 9.807
                * math.cos(theta_rad)
                * (cd / aircraft_state.cl)
            )
            + (aircraft_state.gw_kg * 9.807 * math.sin(theta_rad))
            + (aircraft_state.gw_kg * dv_dt)
        )

        return max(f_thrust, 0.0)

    def compute_min_rating(
        self,
        propulsion_settings: PollSchumannPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        pass

    def compute_max_rating(
        self,
        propulsion_settings: PollSchumannPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        pass

    def _fuel_flow_idle(
        self,
        propulsion_settings: PollSchumannPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        """
        Calculate minimum fuel mass flow rate at flight idle conditions.
        """

        x = aircraft_state.zp_ft / 10_000.0
        return propulsion_settings.ac_parameters["ff_idle_sls"] * (
            1.0 - 0.178 * x + 0.0085 * x**2
        )

    def _equivalent_fuel_flow_rate_at_cruise(
        self,
        propulsion_settings: PollSchumannPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        theta_amb = aircraft_state.weather_state.static_temperature_k / 288.15
        delta_amb = aircraft_state.weather_state.static_pressure_pa / 101325.0

        denom = (theta_amb**3.8 / delta_amb) * np.exp(0.2 * aircraft_state.mach**2)
        denom = np.clip(denom, 1.0, None)

        return propulsion_settings.ac_parameters["ff_max_sls"] / denom

    def _fuel_mass_flow_rate(
        self,
        propulsion_settings: PollSchumannPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        gamma = 1.4
        R = 287.05
        q_fuel = 43e6

        eta = self.__overall_propulsion_efficiency(propulsion_settings, aircraft_state)
        ct = self.__compute_thrust_coefficient(propulsion_settings, aircraft_state)
        return (
            (gamma / 2)
            * (ct * aircraft_state.mach**3 / eta)
            * (gamma * R * aircraft_state.weather_state.static_temperature_k) ** 0.5
            * aircraft_state.weather_state.static_pressure_pa
            * propulsion_settings.ac_parameters["wing_surface_area"]
            / q_fuel
        )

    def __compute_thrust_coefficient(
        self,
        propulsion_settings: PollSchumannPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        ct = self.___engine_thrust_coefficient(propulsion_settings, aircraft_state)
        ct_eta_b = self.___thrust_coefficient_at_max_efficiency(
            propulsion_settings, aircraft_state
        )
        ct_available = self.___max_available_thrust_coefficient(
            propulsion_settings, aircraft_state, ct_eta_b=ct_eta_b
        )
        return np.clip(ct, 0, ct_available)

    def ___engine_thrust_coefficient(
        self,
        propulsion_settings: PollSchumannPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        gamma = 1.4
        f_thrust = self.compute_total_net_thrust_n(propulsion_settings, aircraft_state)

        return f_thrust / (
            0.5
            * gamma
            * aircraft_state.weather_state.static_pressure_pa
            * aircraft_state.mach**2
            * propulsion_settings.ac_parameters["wing_surface_area"]
        )

    def ___thrust_coefficient_at_max_efficiency(
        self,
        propulsion_settings: PollSchumannPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        m_over_m_des = aircraft_state.mach / propulsion_settings.ac_parameters["m_des"]
        h_2 = (
            (1.0 + 0.55 * aircraft_state.mach)
            / (1.0 + 0.55 * propulsion_settings.ac_parameters["m_des"])
        ) / (m_over_m_des**2)

        return h_2 * propulsion_settings.ac_parameters["c_t_des"]

    def ___max_available_thrust_coefficient(
        self,
        propulsion_settings: PollSchumannPropulsionSettings,
        aircraft_state: AircraftState,
        ct_eta_b: float,
    ) -> float:
        tr_max = self.___normalised_max_throttle_parameter(
            propulsion_settings, aircraft_state
        )

        ct_max_over_ct_eta_b = 1.0 + 2.5 * (tr_max - 1.0)
        return ct_max_over_ct_eta_b * ct_eta_b * (1.0 + 0.05)

    def ___normalised_max_throttle_parameter(
        self,
        propulsion_settings: PollSchumannPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        return (
            propulsion_settings.ac_parameters["tet_mcc"]
            / aircraft_state.weather_state.static_temperature_k
        ) / (
            propulsion_settings.ac_parameters["tr_ec"]
            * (
                1.0
                - 0.53
                * (aircraft_state.mach - propulsion_settings.ac_parameters["m_ec"]) ** 2
            )
            * (1.0 + 0.2 * aircraft_state.mach**2)
        )

    def __overall_propulsion_efficiency(
        self,
        propulsion_settings: PollSchumannPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        ct_eta_b = self.___thrust_coefficient_at_max_efficiency(
            propulsion_settings, aircraft_state
        )

        # ct = np.clip(ct, 0, ct_available)
        ct = self.__compute_thrust_coefficient(propulsion_settings, aircraft_state)
        eta_over_eta_b_min = 0.5

        eta_over_eta_b = self.____propulsion_efficiency_over_max_propulsion_efficiency(
            aircraft_state, ct, ct_eta_b
        )

        if eta_over_eta_b_min is not None:
            eta_over_eta_b.clip(min=eta_over_eta_b_min, out=eta_over_eta_b)

        eta_b = (
            propulsion_settings.ac_parameters["eta_1"]
            * aircraft_state.mach ** propulsion_settings.ac_parameters["eta_2"]
        )

        return eta_over_eta_b * eta_b

    def ____propulsion_efficiency_over_max_propulsion_efficiency(
        self, aircraft_state: AircraftState, ct: float, ct_eta_b: float
    ) -> float:
        ct_over_c_t_eta_b = ct / ct_eta_b

        sigma = np.where(
            aircraft_state.mach < 0.4, 1.3 * (0.4 - aircraft_state.mach), 0.0
        )

        eta_over_eta_b_low = (
            10.0
            * (1.0 + 0.8 * (sigma - 0.43) - 0.6027 * sigma * 0.43)
            * ct_over_c_t_eta_b
            + 33.3333
            * (-1.0 - 0.97 * (sigma - 0.43) + 0.8281 * sigma * 0.43)
            * (ct_over_c_t_eta_b**2)
            + 37.037
            * (1.0 + (sigma - 0.43) - 0.9163 * sigma * 0.43)
            * (ct_over_c_t_eta_b**3)
        )
        eta_over_eta_b_hi = (
            (1.0 + (sigma - 0.43) - sigma * 0.43)
            + (4.0 * sigma * 0.43 - 2.0 * (sigma - 0.43)) * ct_over_c_t_eta_b
            + ((sigma - 0.43) - 6 * sigma * 0.43) * (ct_over_c_t_eta_b**2)
            + 4.0 * sigma * 0.43 * (ct_over_c_t_eta_b**3)
            - sigma * 0.43 * (ct_over_c_t_eta_b**4)
        )
        return np.where(ct_over_c_t_eta_b < 0.3, eta_over_eta_b_low, eta_over_eta_b_hi)
