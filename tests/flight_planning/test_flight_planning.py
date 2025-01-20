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
        mach_cruise=0.78,
        cas_climb_kts=170,
        cas_descent_kts=250,
        phase=PhaseEnum.CLIMB,
        rating_level=RatingEnum.MCL,
        cg=0.3,
        gamma_air_deg=0,
    )

    # check that the aircraft state is correctly initialized
    assert acState_openap.model_type == "A320"
    assert acState_openap.performance_model_type == PerformanceModelEnum.OPENAP
    assert acState_openap.gw_kg == 80_000
    assert acState_openap.zp_ft == 10_000
    assert acState_openap.mach_cruise == 0.78
    assert acState_openap.cas_climb_kts == 170
    assert acState_openap.cas_descent_kts == 250
    assert acState_openap.phase == PhaseEnum.CLIMB
    assert acState_openap.rating_level == RatingEnum.MCL
    assert acState_openap.cg == 0.3
    assert acState_openap.gamma_air_deg == 0

    acState_poll_schumann = AircraftState(
        # model_type="BADA/A320-271N", # only for BADA
        model_type="A320",  # only for OPENAP and POLL_SCHUMANN
        performance_model_type=PerformanceModelEnum.POLL_SCHUMANN,  # PerformanceModelEnum.OPENAP
        gw_kg=80_000,
        zp_ft=10_000,
        mach_cruise=0.78,
        cas_climb_kts=170,
        cas_descent_kts=250,
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
    from skdecide.hub.domain.flight_planning.domain import (
        FlightPlanningDomain,
        WeatherDate,
    )

    acState_poll_schumann = AircraftState(
        model_type="A320",  # only for OPENAP and POLL_SCHUMANN
        performance_model_type=PerformanceModelEnum.POLL_SCHUMANN,  # PerformanceModelEnum.OPENAP
        gw_kg=80_000,
        zp_ft=10_000,
        mach_cruise=0.78,
        cas_climb_kts=170,
        cas_descent_kts=250,
        phase=PhaseEnum.CLIMB,
        rating_level=RatingEnum.MCL,
        cg=0.3,
        gamma_air_deg=0,
    )

    domain_factory = lambda: FlightPlanningDomain(
        origin="LFBO",
        destination="LFPG",
        aircraft_state=acState_poll_schumann,
        objective="fuel",
        heuristic_name="lazy_fuel",
        cruise_height_min=32_000,
        cruise_height_max=38_000,
        graph_width="medium",
        nb_vertical_points=3,
        nb_forward_points=5,
        nb_lateral_points=11,
    )

    domain = domain_factory()

    solver = LazyAstar(
        domain_factory=domain_factory, heuristic=lambda d, s: d.heuristic(s)
    )
    solver.solve()

    assert solver.check_domain(domain)
