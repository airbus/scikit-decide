from dataclasses import dataclass
from typing import Optional

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
        x_graph: Optional[int] = None,
        y_graph: Optional[int] = None,
        z_graph: Optional[int] = None,
        latitude_deg: Optional[float] = None,
        longitude_deg: Optional[float] = None,
        total_temperature_k: Optional[float] = None,
        total_pressure_pa: Optional[float] = None,
        # Flight
        ground_dist_m: Optional[float] = None,
        gamma_air_deg: Optional[float] = None,
        cost_index: Optional[float] = None,  # kg / min
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
        self.x_graph = x_graph
        self.y_graph = y_graph
        self.z_graph = z_graph
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg

        self.weather_state = weather_state

        self.tas_meters_per_sec = tas_meters_per_sec
        self.mach = mach
        self.total_temperature_k = total_temperature_k
        self.total_pressure_pa = total_pressure_pa

        self.ground_dist_m = ground_dist_m
        self.gamma_air_deg = gamma_air_deg
        self.cost_index = cost_index

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

    def clone(self):
        new_state = AircraftState(
            performance_model_type=self.performance_model_type,
            model_type=self.model_type,
        )

        new_state.gw_kg = self.gw_kg
        new_state.cg = self.cg
        new_state.weather_state = self.weather_state
        new_state.zp_ft = self.zp_ft
        new_state.time_sec = self.time_sec
        new_state.latitude_deg = self.latitude_deg
        new_state.longitude_deg = self.longitude_deg
        new_state.x_graph = self.x_graph
        new_state.y_graph = self.y_graph
        new_state.z_graph = self.z_graph

        new_state.tas_meters_per_sec = self.tas_meters_per_sec
        new_state.mach = self.mach
        new_state.total_pressure_pa = self.total_pressure_pa
        new_state.total_temperature_k = self.total_temperature_k

        new_state.ground_dist_m = self.ground_dist_m
        new_state.gamma_air_deg = self.gamma_air_deg
        new_state.cost_index = self.cost_index

        new_state.cl = self.cl
        new_state.lift_n = self.lift_n
        new_state.cx = self.cx
        new_state.drag_n = self.drag_n
        new_state.thrust_n = self.thrust_n
        new_state.tsp = self.tsp
        new_state.fuel_flow_kg_per_sec = self.fuel_flow_kg_per_sec

        new_state.phase = self.phase
        new_state.rating_level = self.rating_level

        return new_state
