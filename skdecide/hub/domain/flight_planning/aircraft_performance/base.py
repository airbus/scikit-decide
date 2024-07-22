# models fuel flows
# typing
import math
from inspect import signature
from typing import Dict, Optional

import numpy as np
import openap
from openap.extra.aero import crossover_alt, distance, fpm, ft, kts, latlon, mach2tas

# other
from openap.prop import aircraft

from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils import (
    pollschumann,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import (
    load_aircraft_engine_params,
)


class AircraftPerformanceModel:
    def __init__(self, actype: str, perf_model: str = None):
        self.perf_model_name = perf_model

        if perf_model == "openap":
            self.perf_model = OpenAP(actype)
        elif perf_model == "PS":
            print("Poll-Schumann model")
            self.perf_model = PollSchumannModel(actype)
        else:
            raise ValueError(f"Unknown performance model: {perf_model}")

    def compute_fuel_consumption(
        self,
        values_current: Dict[str, float],
        delta_time: float,
        vs: Optional[float] = 0.0,
    ) -> float:

        return self.perf_model.compute_fuel_consumption(
            values_current, delta_time, vs=vs
        )

    def compute_crossover_altitude(self) -> float:
        return self.perf_model.compute_crossover_altitude()


# create OpenAP class that inherits from the base class
class OpenAP(AircraftPerformanceModel):
    def __init__(self, actype: str):
        self.ac = aircraft(actype)
        self.fuel_flow = openap.FuelFlow(ac=actype, polydeg=2).enroute

    def compute_fuel_consumption(
        self,
        values_current: Dict[str, float],
        delta_time: float,
        vs: Optional[float] = 0.0,
    ) -> float:

        mass_current, altitude_current, speed_current = (
            values_current["mass"],
            values_current["alt"],
            values_current["speed"],
        )
        if "vs" in signature(self.fuel_flow).parameters:
            ff = self.fuel_flow(mass_current, speed_current, altitude_current, vs=vs)
        else:
            path_angle = math.degrees(np.arctan2(vs * fpm, speed_current * kts))
            ff = self.fuel_flow(
                mass_current, speed_current, altitude_current, path_angle=path_angle
            )

        return delta_time * ff

    def compute_crossover_altitude(self, cas, mach) -> float:
        return crossover_alt(cas, mach)


# create aircraft performance model based on Poll-Schumann model that inherits from the base class
class PollSchumannModel(AircraftPerformanceModel):
    def __init__(self, actype: str):
        self.actype = actype
        self.fuel_flow = pollschumann.FuelFlow(actype)

    def compute_fuel_consumption(
        self,
        values_current: Dict[str, float],
        delta_time: float,
        vs: Optional[float] = 0.0,
    ) -> float:
        ff = self.fuel_flow(values_current, delta_time=delta_time, vs=vs)

        return delta_time * ff

    def compute_crossover_altitude(self) -> float:
        return load_aircraft_engine_params(self.actype)["p_inf_co"]
