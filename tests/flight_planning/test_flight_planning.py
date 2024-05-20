import sys

import pytest
from openap.prop import aircraft

from skdecide.hub.solver.lazy_astar import LazyAstar


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="cartopy requires python3.9 or higher"
)
@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="pygrib does not install on windows"
)
def test_flight_planning():
    from skdecide.hub.domain.flight_planning.domain import FlightPlanningDomain

    origin = "LFPG"
    destination = "LFBO"
    actype = "A320"
    heuristic = "lazy_fuel"
    cost_function = "fuel"
    domain_factory = lambda: FlightPlanningDomain(
        origin=origin,
        destination=destination,
        actype=actype,
        heuristic_name=heuristic,
        objective=cost_function,
        fuel_loop=False,
        graph_width="normal",
    )
    domain = domain_factory()
    solver = LazyAstar(
        domain_factory=domain_factory, heuristic=lambda d, s: d.heuristic(s)
    )
    solver.solve()

    assert solver.check_domain(domain)


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="cartopy requires python3.9 or higher"
)
@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="pygrib does not install on windows"
)
def test_flight_planning_fuel_loop():
    from skdecide.hub.domain.flight_planning.domain import FlightPlanningDomain

    origin = "LFPG"
    destination = "LFBO"
    actype = "A320"
    heuristic = "lazy_fuel"
    cost_function = "fuel"
    constraints = {"fuel": aircraft(actype)["limits"]["MFC"]}
    domain_factory = lambda: FlightPlanningDomain(
        origin=origin,
        destination=destination,
        actype=actype,
        constraints=constraints,
        heuristic_name=heuristic,
        objective=cost_function,
        fuel_loop=True,
        graph_width="normal",
        fuel_loop_tol=1.0,
    )
    domain = domain_factory()
    solver = LazyAstar(
        domain_factory=domain_factory, heuristic=lambda d, s: d.heuristic(s)
    )
    solver.solve()

    assert solver.check_domain(domain)
