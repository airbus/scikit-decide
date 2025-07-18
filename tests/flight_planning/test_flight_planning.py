import sys

import pytest

from skdecide.hub.solver.lazy_astar import LazyAstar


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="cartopy requires python3.9 or higher"
)
@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="pygrib does not install on windows"
)
def test_aircraft_state():
    from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
        AircraftState,
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

    acState_openap = AircraftState(
        model_type="A320",  # only for OPENAP and POLL_SCHUMANN
        performance_model_type=PerformanceModelEnum.OPENAP,  # PerformanceModelEnum.OPENAP, PerformanceModelEnum.BADA
        gw_kg=80_000,
        zp_ft=10_000,
        mach=0.78,
        phase=PhaseEnum.CLIMB,
        rating_level=RatingEnum.MCL,
        cg=0.3,
    )

    # check that the aircraft state is correctly initialized
    assert acState_openap.model_type == "A320"
    assert acState_openap.performance_model_type == PerformanceModelEnum.OPENAP
    assert acState_openap.gw_kg == 80_000
    assert acState_openap.zp_ft == 10_000
    assert acState_openap.mach == 0.78
    assert acState_openap.phase == PhaseEnum.CLIMB
    assert acState_openap.rating_level == RatingEnum.MCL
    assert acState_openap.cg == 0.3

    acState_poll_schumann = AircraftState(
        # model_type="BADA/A320-271N", # only for BADA
        model_type="A320",  # only for OPENAP and POLL_SCHUMANN
        performance_model_type=PerformanceModelEnum.POLL_SCHUMANN,  # PerformanceModelEnum.OPENAP
        gw_kg=80_000,
        zp_ft=10_000,
        mach=0.78,
        phase=PhaseEnum.CLIMB,
        rating_level=RatingEnum.MCL,
        cg=0.3,
        gamma_air_deg=0,
    )

    # check that the aircraft state is correctly initialized
    assert acState_poll_schumann.model_type == "A320"
    assert (
        acState_poll_schumann.performance_model_type
        == PerformanceModelEnum.POLL_SCHUMANN
    )


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="cartopy requires python3.9 or higher"
)
@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="pygrib does not install on windows"
)
def test_flight_planning():
    import numpy as np
    from pygeodesy.ellipsoidalVincenty import LatLon

    from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
        AircraftState,
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
    from skdecide.hub.domain.flight_planning.domain import FlightPlanningDomain

    acState_poll_schumann = AircraftState(
        model_type="A320",  # only for OPENAP and POLL_SCHUMANN
        performance_model_type=PerformanceModelEnum.POLL_SCHUMANN,  # PerformanceModelEnum.OPENAP
        gw_kg=80_000,
        zp_ft=10_000,
        mach=0.78,
        phase=PhaseEnum.CLIMB,
        rating_level=RatingEnum.MCL,
        cg=0.3,
        gamma_air_deg=0,
    )

    domain_factory = lambda: FlightPlanningDomain(
        aircraft_state=acState_poll_schumann,
        mach_cruise=0.78,
        mach_climb=0.7,
        mach_descent=0.65,
        nb_forward_points=20,
        nb_lateral_points=10,
        nb_climb_descent_steps=5,
        flight_levels_ft=list(np.arange(30_000, 38_000 + 2_000, 2_000)),
        graph_width="medium",
        origin=LatLon(43.629444, 1.363056),
        destination="EDDB",
        objective="fuel",
    )

    domain = domain_factory()

    solver = LazyAstar(
        domain_factory=domain_factory, heuristic=lambda d, s: d.heuristic(s)
    )
    solver.solve()

    assert solver.check_domain(domain)


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="pygrib does not install on windows"
)
@pytest.mark.parametrize(
    "heuristic_name", ["distance", "time", "fake", "lazy_fuel", "lazy_time"]
)
def test_heuristic(heuristic_name):
    import numpy as np
    from pygeodesy.ellipsoidalVincenty import LatLon

    from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
        AircraftState,
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
    from skdecide.hub.domain.flight_planning.domain import FlightPlanningDomain

    acState_poll_schumann = AircraftState(
        model_type="A320",  # only for OPENAP and POLL_SCHUMANN
        performance_model_type=PerformanceModelEnum.POLL_SCHUMANN,  # PerformanceModelEnum.OPENAP
        gw_kg=80_000,
        zp_ft=10_000,
        mach=0.78,
        phase=PhaseEnum.CLIMB,
        rating_level=RatingEnum.MCL,
        cg=0.3,
        gamma_air_deg=0,
    )

    domain_factory = lambda: FlightPlanningDomain(
        aircraft_state=acState_poll_schumann,
        mach_cruise=0.78,
        mach_climb=0.7,
        mach_descent=0.65,
        nb_forward_points=20,
        nb_lateral_points=10,
        nb_climb_descent_steps=5,
        flight_levels_ft=list(np.arange(30_000, 38_000 + 2_000, 2_000)),
        graph_width="medium",
        origin=LatLon(43.629444, 1.363056),
        destination="EDDB",
        objective="fuel",
        heuristic_name=heuristic_name,
    )

    domain = domain_factory()
    s = domain.reset()
    domain.heuristic(s)
