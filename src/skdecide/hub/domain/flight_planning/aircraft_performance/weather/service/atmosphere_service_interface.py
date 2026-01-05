from abc import ABC, abstractmethod

from skdecide.hub.domain.flight_planning.aircraft_performance.bean.four_dimensions_state import (
    FourDimensionsState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.bean.weather_state import (
    WeatherState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.atmosphere_model_enum import (
    AtmosphereModelEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.settings.atmosphere_settings_interface import (
    AtmosphereSettings,
)


class AtmopshereServiceInterface(ABC):
    """
    Interface defining atmosphere services template
    """

    @property
    @abstractmethod
    def atmosphere_model_type(self) -> AtmosphereModelEnum:
        """
        Property defining the type of modelling
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_weather_state(
        self,
        atmosphere_settings: AtmosphereSettings,
        four_dimensions_state: FourDimensionsState,
    ) -> WeatherState:
        """
        :param atmosphere_settings:
        :param four_dimensions_state:
        :return:
        """
        raise NotImplementedError
