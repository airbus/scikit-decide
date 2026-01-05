from dataclasses import dataclass
from typing import Optional


@dataclass
class FourDimensionsState:
    """
    Class representing a 4-D state (3D position and time)
    """

    zp_ft: Optional[float] = None  # pressure altitude
    time_sec: Optional[float] = None
    x_graph: Optional[int] = None
    y_graph: Optional[int] = None
    z_graph: Optional[int] = None
    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
