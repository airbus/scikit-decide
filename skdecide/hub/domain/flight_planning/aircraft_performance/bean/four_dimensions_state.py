from dataclasses import dataclass
from typing import Optional


@dataclass
class FourDimensionsState:
    """
    Class representing a 4-D state (3D position and time)
    """

    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
    zp_ft: Optional[float] = None  # pressure altitude
    time_sec: Optional[float] = None
