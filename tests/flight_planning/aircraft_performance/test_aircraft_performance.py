import pytest
from pathlib import Path
from skdecide.hub.domain.flight_planning.aircraft_performance.base import AircraftPerformanceModel


# @pytest.fixture(scope="module")
# def data_dir():
#     yield Path(__file__).parent / "data"

# def test_toto():
#     assert 1+1==2

# def test_titi():
#     assert 1+1==3

# def test_tata(tmpdir, data_dir):
#     print(tmpdir)
#     path = tmpdir.join("test.txt")

#     with open(data_dir / "test.csv", "r") as f:
#         data = f.read()

#     with path.open("w") as f:
#         f.write(data)

#     assert path.exists()

# def add(a, b):
#     return a + b

# @pytest.mark.parametrize("a, b, expected", [(1, 2, 3), (3, 5, 8), (1, 3, 19)])
# def test_add(a, b, expected):
#     assert add(a, b) == expected


@pytest.fixture(scope="function")
def ac_model(model):
    out = AircraftPerformanceModel(actype="A320", perf_model=model)
    yield out

@pytest.mark.parametrize("model, mass, alt, speed, delta_time, path_angle, temp, expected", [
    ("openap", 200_000, 10_000, 300, 10, 0.0, 273, 9.467747725832536),
    ("openap", 200_000, 10_000, 310, 50, 10.0, 273, 46.57236005543845),
    ("openap", 220_000, 10_000, 280, 60, -5.0, 273, 17.316834732381043),
    ("PS", 200_000, 10_000, 300, 10, 0.0, 273, 6.449999999999999),
    ("PS", 200_000, 10_000, 310, 50, 10.0, 273, 86.65624331930233),
    ("PS", 220_000, 10_000, 280, 60, -5.0, 273, 38.699999999999996),
])
def test_perf_model(ac_model, model, mass, alt, speed, delta_time, path_angle, temp, expected):
    assert ac_model.compute_fuel_consumption(
        values_current={"mass": mass, "alt": alt, "speed": speed, "temp": temp},
        delta_time=delta_time,
        path_angle=path_angle,
    ) == pytest.approx(expected, abs=1e-1)