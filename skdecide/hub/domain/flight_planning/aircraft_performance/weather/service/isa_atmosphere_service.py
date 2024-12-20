from typing import Optional

import numpy as np

from skdecide.hub.domain.flight_planning.aircraft_performance.bean.four_dimensions_state import (
    FourDimensionsState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.bean.weather_state import (
    WeatherState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.atmosphere_model_enum import (
    AtmosphereModelEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.service.atmosphere_service_interface import (
    AtmopshereServiceInterface,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.settings.isa_atmosphere_settings import (
    IsaAtmosphereSettings,
)


class ISAAtmosphereService(AtmopshereServiceInterface):
    """
    Implementation of the ISA atmosphere
    """

    atmosphere_model_type = AtmosphereModelEnum.ISA

    ISA_T0_K = 288.15
    TEMP_GRAD = -1.9812 / 1000  # gradient temperature before tropopause
    DEFAULT_TROPOPAUSE_FT = 36089

    def retrieve_weather_state(
        self,
        atmosphere_settings: IsaAtmosphereSettings,
        four_dimensions_state: FourDimensionsState,
    ) -> WeatherState:
        """
        From the aircraft state location and atmosphere settings, compute the weather state using the appropriate atmosphere service
        :param atmosphere_settings: Settings defining the atmosphere (type, constants...)
        :param four_dimensions_state: 4D state (zp, location, time)
        :return: Weather state (Temperature, pressure...)
        """
        alt_ft = four_dimensions_state.zp_ft

        if alt_ft is None:
            raise Exception(
                "Aircraft state zp must be initiated for isa atmosphere computation"
            )

        isa_temperature_k = self.__isa_temperature_k(
            altitude_ft=alt_ft, tropo_ft=atmosphere_settings.tropopause_alt_ft
        )

        pressure_pa = self.__pressure(altitude_ft=alt_ft)

        return WeatherState(
            static_temperature_k=isa_temperature_k + atmosphere_settings.d_isa,
            static_pressure_pa=pressure_pa,
            d_isa=atmosphere_settings.d_isa,
            tailwind_m_per_sec=atmosphere_settings.tailwind_m_per_sec,
        )

    def __isa_temperature_k(
        self, altitude_ft: float, tropo_ft: Optional[float] = None
    ) -> float:
        """Compute the temperature at the given altitude using a custom value for the tropopause altitude
        Args:
            altitude_ft (float): user altitude at which the temperature will be computed
            tropo_ft (float): user custom tropopause altitude
        Returns:
            float: isa temperature (K)
        """

        return self.ISA_T0_K + self.TEMP_GRAD * min(
            altitude_ft,
            tropo_ft if tropo_ft is not None else self.DEFAULT_TROPOPAUSE_FT,
        )

    def __pressure(self, altitude_ft=0.0):
        """
        Compute Static Pressure in Pa from altitude in ft
        Remark: altitude is optional and can be provided as a float,
        a numpy or a pd.series
        :param altitude_ft: altitude (ft) (default : 0)
        :return: pressure (Pa)
        """

        """
        The atmosphere is split into 5 zones:
        - zone 1: pressure height up to 11000. m
        - zone 2: pressure height btw 11000. to 20000. m
        - zone 3: pressure height btw 20000. to 32000. m
        - zone 4: pressure height btw 32000. to 47000. m
        - zone 5: pressure height btw above 47000. m
        on zone 1,3,4, pressure is computed using coefficient C, D, E:
        pressure = (C + D * pressure height)^E
        on zone 2 and 5, pressure is computed using coefficient F, G:
        temperature = F * exp(G * pressure height)
        """

        press_constant = {}
        press_constant["zone1"] = {"C": 8.9619638, "D": -0.20216125e-3, "E": 5.2558797}
        press_constant["zone2"] = {"F": 128244.5, "G": -0.15768852e-3}
        press_constant["zone3"] = {"C": 0.70551848, "D": 3.5876861e-6, "E": -34.163218}
        press_constant["zone4"] = {"C": 0.34926867, "D": 7.0330980e-6, "E": -12.201149}
        press_constant["zone5"] = {"F": 41828.420, "G": -0.12622656e-3}

        altitude_m = altitude_ft * 0.3048

        if altitude_m < 11000.0:
            zone = "zone1"
            pressure_pa = np.power(
                press_constant[zone]["C"] + press_constant[zone]["D"] * altitude_m,
                press_constant[zone]["E"],
            )

        elif altitude_m < 20000.0:
            zone = "zone2"
            pressure_pa = press_constant[zone]["F"] * np.exp(
                press_constant[zone]["G"] * altitude_m
            )

        elif altitude_m < 32000:
            zone = "zone3"
            pressure_pa = np.power(
                press_constant[zone]["C"] + press_constant[zone]["D"] * altitude_m,
                press_constant[zone]["E"],
            )

        elif altitude_m < 47000:
            zone = "zone4"
            pressure_pa = np.power(
                press_constant[zone]["C"] + press_constant[zone]["D"] * altitude_m,
                press_constant[zone]["E"],
            )
        else:
            zone = "zone5"
            pressure_pa = press_constant[zone]["F"] * np.exp(
                press_constant[zone]["G"] * altitude_m
            )

        return float(pressure_pa)
