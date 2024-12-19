from dataclasses import dataclass


@dataclass
class SpeedSchedule:
    """
    Bean to store speed schedule law for climb or descent
    """

    cas_low_kt: float = 250  # CAS under 10'000 ft
    cas_high_kt: float = None  # CAS above 10'000 ft
    mach: float = None  # Mach law
