from dataclasses import dataclass
from typing import List, Optional

from skdecide.hub.domain.flight_planning.aircraft_performance.bean.four_dimensions_state import (
    FourDimensionsState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.bean.weather_state import (
    WeatherState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.phase_enum import (
    PhaseEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.rating_enum import (
    RatingEnum,
)


@dataclass
class AircraftState(FourDimensionsState):
    """
    Class representing an aircraft state and variables.
    Aircraft state: dynamic and weather state
    """

    def __init__(
        self,
        performance_model_type: PerformanceModelEnum = None,
        model_type: Optional[str] = None,
        zp_ft: Optional[float] = None,
        gw_kg: Optional[float] = None,
        cg: Optional[float] = None,
        weather_state: Optional[WeatherState] = None,
        tas_meters_per_sec: Optional[float] = None,
        mach: Optional[float] = None,
        mach_cruise: Optional[float] = None,
        cas_climb_kts: Optional[float] = None,
        cas_descent_kts: Optional[float] = None,
        total_temperature_k: Optional[float] = None,
        total_pressure_pa: Optional[float] = None,
        # Flight
        ground_dist_m: Optional[float] = None,
        gamma_air_deg: Optional[float] = None,
        cost_index: Optional[float] = None,  # kg / min
        rocd_ft_min: Optional[float] = None,
        # aero
        cl: Optional[float] = None,
        lift_n: Optional[float] = None,
        cx: Optional[float] = None,
        drag_n: Optional[float] = None,
        # propu
        is_one_eo: Optional[bool] = False,
        is_air_cond_on: Optional[bool] = False,
        rating_level: Optional[RatingEnum] = None,
        tsp: Optional[float] = None,  # reduced tsp
        thrust_n: Optional[float] = None,
        fuel_flow_kg_per_sec: Optional[float] = None,
        # phase
        phase: Optional[PhaseEnum] = None,
    ):

        self.performance_model_type = performance_model_type

        self.model_type = model_type

        self.zp_ft = zp_ft
        self.gw_kg = gw_kg
        self.cg = cg

        self.weather_state = weather_state

        self.tas_meters_per_sec = tas_meters_per_sec
        self.mach = mach
        self.mach_cruise = mach_cruise
        self.cas_climb_kts = cas_climb_kts
        self.cas_descent_kts = cas_descent_kts
        self.total_temperature_k = total_temperature_k
        self.total_pressure_pa = total_pressure_pa

        self.ground_dist_m = ground_dist_m
        self.gamma_air_deg = gamma_air_deg
        self.cost_index = cost_index
        self.rocd_ft_min = rocd_ft_min

        self.cl = cl
        self.lift_n = lift_n
        self.cx = cx
        self.drag_n = drag_n

        self.is_one_eo = is_one_eo
        self.is_air_cond_on = is_air_cond_on
        self.rating_level = rating_level
        self.tsp = tsp
        self.thrust_n = thrust_n
        self.fuel_flow_kg_per_sec = fuel_flow_kg_per_sec

        self.phase = phase

        self.time_step = None

        if self.performance_model_type.name == PerformanceModelEnum.OPENAP.name:
            self._init_openap_settings()
        elif (
            self.performance_model_type.name == PerformanceModelEnum.POLL_SCHUMANN.name
        ):
            self._init_pollschumann_settings()
        else:
            raise ValueError("Error in aircraft state settings init.")

    def _init_openap_settings(self):
        from openap.prop import aircraft

        ac_params = aircraft(self.model_type)

        self.MTOW = ac_params["mtow"]
        self.MFC = ac_params["limits"]["MFC"]
        self.MMO = ac_params["cruise"]["mach"]

    def _init_pollschumann_settings(self):
        from skdecide.hub.domain.flight_planning.aircraft_performance.utils.poll_schumann_utils.engine_loader import (
            load_aircraft_engine_params,
        )

        ac_params = load_aircraft_engine_params(self.model_type)

        self.MTOW = ac_params["amass_mtow"]
        self.MMO = ac_params["max_mach_num"]
        self.MFC = ac_params["amass_mtow"] - ac_params["amass_mzfw"]

    def update_settings(
        self,
        gw_kg: Optional[float] = None,
        zp_ft: Optional[float] = None,
        mach: Optional[float] = None,
    ):

        if gw_kg is not None:
            self.gw_kg = gw_kg
        if zp_ft is not None:
            self.zp_ft = zp_ft
        if mach is not None:
            self.mach = mach
