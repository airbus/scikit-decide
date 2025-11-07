import math

from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
    AircraftState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.bean.weather_state import (
    WeatherState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.service.atmosphere_service import (
    AtmosphereService,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.service.isa_atmosphere_service import (
    ISAAtmosphereService,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.settings.isa_atmosphere_settings import (
    IsaAtmosphereSettings,
)


class SpeedConversionService:
    """
    Class used for classical speed conversion (MACH, TAS, CAS)
    """

    _GAMMA = 1.4
    _KT_TO_MS = 1852.0 / 3600.0

    def convert_speed_to_kt(self, speed_ms: float) -> float:
        return speed_ms / self._KT_TO_MS

    def convert_speed_to_ms(self, speed_kt: float) -> float:
        return speed_kt * self._KT_TO_MS

    def convert_mach_to_cas_kt(self, mach: float, weather_state: WeatherState) -> float:
        """
        Mach to CAS conversion
        :param mach: Input mach number
        :param weather_state: Current weather state
        :return: CAS speed in kt
        """

        # Retrieve the ISA weather state (
        weather_state_isa = ISAAtmosphereService().retrieve_weather_state(
            IsaAtmosphereSettings(), AircraftState(zp_ft=0)
        )

        a0 = AtmosphereService().get_sound_celerity(
            static_temperature_k=weather_state_isa.static_temperature_k
        )

        cas = 1.0 + (self._GAMMA - 1.0) / 2.0 * mach**2
        cas = (
            weather_state.static_pressure_pa
            * math.pow(cas, self._GAMMA / (self._GAMMA - 1.0))
            - weather_state.static_pressure_pa
        )
        cas += weather_state_isa.static_pressure_pa
        cas = math.pow(
            cas / weather_state_isa.static_pressure_pa,
            (self._GAMMA - 1.0) / self._GAMMA,
        )
        cas += -1
        cas *= 2 / (self._GAMMA - 1.0) * a0**2
        cas = math.sqrt(cas)

        return cas / self._KT_TO_MS

    def convert_cas_to_mach(self, cas_kt: float, weather_state: WeatherState) -> float:
        """
        CAS to Mach conversion
        :param :cas_kt: Input CAS speed in kt
        :param weather_state: Current weather state
        :return: MACH number
        """

        # Retrieve the ISA weather state (
        weather_state_isa = ISAAtmosphereService().retrieve_weather_state(
            IsaAtmosphereSettings(), AircraftState(zp_ft=0)
        )

        a0 = AtmosphereService().get_sound_celerity(
            static_temperature_k=weather_state_isa.static_temperature_k
        )

        mach = 1.0 + (self._GAMMA - 1.0) / 2.0 * (cas_kt * self._KT_TO_MS / a0) ** 2
        mach = (
            weather_state_isa.static_pressure_pa
            * math.pow(mach, self._GAMMA / (self._GAMMA - 1.0))
            - weather_state_isa.static_pressure_pa
        )
        mach += weather_state.static_pressure_pa
        mach *= 1 / weather_state.static_pressure_pa
        mach = math.pow(mach, (self._GAMMA - 1.0) / self._GAMMA) - 1
        mach = 2 * mach / (self._GAMMA - 1.0)
        mach = math.sqrt(mach)

        return mach
