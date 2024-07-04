import inspect
import math
import sys

import pytest

if sys.version_info < (3, 9):
    pytest.skip("cartopy requires python3.9 or higher", allow_module_level=True)
if sys.platform.startswith("win"):
    pytest.skip("pygrib does not install on windows", allow_module_level=True)


@pytest.fixture(scope="function")
def ac_model(model):
    from skdecide.hub.domain.flight_planning.aircraft_performance.base import (
        AircraftPerformanceModel,
    )

    out = AircraftPerformanceModel(actype="A320", perf_model=model)
    yield out


@pytest.mark.parametrize(
    "model, mass, alt, speed, delta_time, path_angle, temp, expected, expected_openap1",
    [
        ("openap", 200_000, 10_000, 300, 10, 0.0, 273, 17.1, 9.5),
        ("openap", 200_000, 10_000, 310, 50, 10.0, 273, 84.8, 46.6),
        ("openap", 220_000, 10_000, 280, 60, -5.0, 273, 12.0, 17.3),
        ("PS", 200_000, 10_000, 300, 10, 0.0, 273, 6.4, None),
        ("PS", 200_000, 10_000, 310, 50, 10.0, 273, 86.7, None),
        ("PS", 220_000, 10_000, 280, 60, -5.0, 273, 11.0, None),
    ],
)
def test_perf_model(
    ac_model,
    model,
    mass,
    alt,
    speed,
    delta_time,
    path_angle,
    temp,
    expected,
    expected_openap1,
):
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.utils.aero import (
        ft,
        kts,
    )

    if model == "openap":
        from openap import FuelFlow

        if "path_angle" in inspect.signature(FuelFlow.enroute).parameters:
            # openap <= 1.5
            expected = expected_openap1

    fpm = ft / 60
    vs = math.tan(math.radians(path_angle)) * speed * kts / fpm

    assert ac_model.compute_fuel_consumption(
        values_current={"mass": mass, "alt": alt, "speed": speed, "temp": temp},
        delta_time=delta_time,
        vs=vs,
    ) == pytest.approx(expected, abs=1e-1)
