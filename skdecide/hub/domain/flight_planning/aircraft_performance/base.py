# models fuel flows
from poll_schumann import pollschumann
import openap

from typing import Dict, Optional

class AircraftPerformanceModel():
    def __init__(self, actype: str, perf_model: str = None):
        self.perf_model_name = perf_model

        if perf_model == 'openap':
            self.perf_model = OpenAP(actype)
        elif perf_model == 'PS':
            self.perf_model = PollSchumannModel(actype)
        else:
            raise ValueError(f'Unknown performance model: {perf_model}')
        
    def compute_fuel_consumption(
            self,
            values_current: Dict[str, float],
            delta_time: float,
            path_angle: Optional[float] = 0.0) -> float:
        
        return self.perf_model.compute_fuel_consumption(values_current, delta_time, path_angle=path_angle)
    
# create OpenAP class that inherits from the base class
class OpenAP(AircraftPerformanceModel):
    def __init__(self, actype: str):
        self.fuel_flow = openap.FuelFlow(ac=actype, polydeg=2).enroute

    def compute_fuel_consumption(
            self,
            values_current: Dict[str, float],
            delta_time: float,
            path_angle: Optional[float] = 0.0) -> float:
        
        mass_current, altitude_current, speed_current = values_current["mass"], values_current["alt"], values_current["speed"]
        ff = self.fuel_flow(mass_current, speed_current, altitude_current, path_angle=path_angle)

        return delta_time * ff
    
# create aircraft performance model based on Poll-Schumann model that inherits from the base class
class PollSchumannModel(AircraftPerformanceModel):
    def __init__(self, actype: str):
        self.fuel_flow = pollschumann.FuelFlow(actype)


    def compute_fuel_consumption(
            self,
            values_current: Dict[str, float],
            delta_time: float,
            path_angle: Optional[float] = 0.0) -> float:
        
        ff = self.fuel_flow(values_current, delta_time=delta_time, path_angle=path_angle)

        return delta_time * ff