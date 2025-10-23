from openap.extra.aero import ft, mach2tas
from openap.fuel import FuelFlow
from openap.thrust import Thrust

from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
    AircraftState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.service.speed_conversion_service import (
    SpeedConversionService,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.phase_enum import (
    PhaseEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.propulsion.service.propulsion_service_interface import (
    PropulsionServiceInterface,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.propulsion.settings.openap_propulsion_settings import (
    OpenapPropulsionSettings,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.service.atmosphere_service import (
    AtmosphereService,
)


class _OpenapPropulsionService(PropulsionServiceInterface):
    """
    Implementation of propulsion service for OpenAP models
    """

    performance_model_type = PerformanceModelEnum.OPENAP

    speedconversion_service = SpeedConversionService()
    atmosphere_service = AtmosphereService()

    def init_settings(self, model_path: str) -> OpenapPropulsionSettings:
        fuel = FuelFlow(ac=model_path)
        thrust = Thrust(ac=model_path)

        return OpenapPropulsionSettings(
            fuel=fuel, thrust=thrust, sref=fuel.aircraft["wing"]["area"]
        )

    def compute_total_fuel_flow_kg_per_sec(
        self,
        propulsion_settings: OpenapPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        tas_meters_per_sec = mach2tas(aircraft_state.mach, aircraft_state.zp_ft * ft)

        try:
            vs = (
                (aircraft_state.zp_ft - aircraft_state.zp_ft_memory[-1])
                * 60
                / aircraft_state.time_step
            )
        except:
            vs = 0

        return propulsion_settings.fuel.enroute(
            mass=aircraft_state.gw_kg,
            tas=self.speedconversion_service.convert_speed_to_kt(tas_meters_per_sec),
            alt=aircraft_state.zp_ft,
            vs=vs,
        )

    def compute_total_net_thrust_n(
        self,
        propulsion_settings: OpenapPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        tas_meters_per_sec = mach2tas(aircraft_state.mach, aircraft_state.zp_ft * ft)

        if aircraft_state.phase == PhaseEnum.CRUISE:
            return propulsion_settings.thrust.cruise(
                tas=self.speedconversion_service.convert_speed_to_kt(
                    tas_meters_per_sec
                ),
                alt=aircraft_state.zp_ft,
            )
        elif aircraft_state.phase == PhaseEnum.CLIMB:
            return propulsion_settings.thrust.climb(
                tas=self.speedconversion_service.convert_speed_to_kt(
                    tas_meters_per_sec
                ),
                alt=aircraft_state.zp_ft,
                roc=aircraft_state.rocd_ft_min,
            )
        elif aircraft_state.phase == PhaseEnum.DESCENT:
            return propulsion_settings.thrust.descent_idle(
                tas=self.speedconversion_service.convert_speed_to_kt(
                    tas_meters_per_sec
                ),
                alt=aircraft_state.zp_ft,
            )

    def compute_min_rating(
        self,
        propulsion_settings: OpenapPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        pass

    def compute_max_rating(
        self,
        propulsion_settings: OpenapPropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        pass
