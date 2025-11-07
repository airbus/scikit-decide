import math

from skdecide.hub.domain.flight_planning.aircraft_performance.bean.four_dimensions_state import (
    FourDimensionsState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.bean.weather_state import (
    WeatherState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.atmosphere_model_enum import (
    AtmosphereModelEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.service.isa_atmosphere_service import (
    ISAAtmosphereService,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.settings.atmosphere_settings_interface import (
    AtmosphereSettings,
)


class AtmosphereService:
    """
    Class atmosphere variables computations and constants
    """

    # FMS definition of perfect fluid constants
    _MU0 = 17.82 * math.pow(10, -6)  # Dynamic viscosity Pa / s
    _R = 287.053
    _GAMMA = 1.4

    def __init__(self):
        # Retrieve all the atmosphere services
        self.all_atmosphere_services = {AtmosphereModelEnum.ISA: ISAAtmosphereService()}

    def retrieve_weather_state(
        self,
        atmosphere_settings: AtmosphereSettings,
        four_dimensions_state: FourDimensionsState,
    ) -> WeatherState:
        """
        From the 4D state location and atmosphere settings, compute the weather state using the appropriate atmosphere service
        :param atmosphere_settings: Settings defining the atmosphere (type, constants...)
        :param four_dimensions_state: 4D state (zp, location, time)
        :return: Weather state (Temperature, pressure...)
        """
        weather_state = self.all_atmosphere_services[
            atmosphere_settings.atmosphere_model_type
        ].retrieve_weather_state(
            atmosphere_settings=atmosphere_settings,
            four_dimensions_state=four_dimensions_state,
        )

        weather_state.rho_kg_m3 = self.get_volume_mass(
            static_pressure_pa=weather_state.static_pressure_pa,
            static_temperature_k=weather_state.static_temperature_k,
        )
        weather_state.mu_pa_s = self.get_dynamic_viscosity(
            static_temperature_k=weather_state.static_temperature_k
        )
        weather_state.nu_m2_s = self.get_kinematic_viscosity(
            static_pressure_pa=weather_state.static_pressure_pa,
            static_temperature_k=weather_state.static_temperature_k,
        )

        return weather_state

    def get_dynamic_viscosity(self, static_temperature_k: float) -> float:
        """
        FMS formula to compute Mu
        :param static_temperature_k: static air temperature (K)
        :return: Dynamic viscosity value
        """

        return (
            0.0834434
            * self._MU0
            * math.pow(static_temperature_k, 1.5)
            / (static_temperature_k + 110.4)
        )

    def get_volume_mass(self, static_pressure_pa: float, static_temperature_k: float):
        """
        Density computation of a perfect gaz
        :param static_pressure_pa:
        :param static_temperature_k:
        :return:
        """
        return static_pressure_pa / (self._R * static_temperature_k)

    def get_kinematic_viscosity(
        self, static_pressure_pa: float, static_temperature_k: float
    ):
        return self.get_dynamic_viscosity(
            static_temperature_k=static_temperature_k
        ) / self.get_volume_mass(
            static_pressure_pa=static_pressure_pa,
            static_temperature_k=static_temperature_k,
        )

    def get_sound_celerity(self, static_temperature_k: float):
        return math.sqrt(self._GAMMA * self._R * static_temperature_k)
