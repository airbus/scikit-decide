from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WeatherState:
    """
    Class representing the atmosphere state
    """

    static_temperature_k: Optional[float] = None
    static_pressure_pa: Optional[float] = None
    rho_kg_m3: Optional[float] = None
    mu_pa_s: Optional[float] = None
    nu_m2_s: Optional[float] = None
    d_isa: Optional[float] = None
    tailwind_m_per_sec: Optional[float] = None
