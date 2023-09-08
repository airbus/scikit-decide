from skdecide.hub.domain.flight_planning.domain import FlightPlanningDomain
from skdecide.hub.solver.lazy_astar import LazyAstar


def test_flight_planning():
    noexcept = True

    try:
        origin = "LFPG"
        destination = "LFBO"
        aircraft = "A320"
        heuristic = "lazy_fuel"
        cost_function = "fuel"
        domain_factory = lambda: FlightPlanningDomain(
            origin,
            destination,
            aircraft,
            heuristic_name=heuristic,
            objective=cost_function,
            fuel_loop=False,
            graph_width="normal",
        )
        domain = domain_factory()
        solver = LazyAstar(heuristic=lambda d, s: d.heuristic(s))
        domain.solve(domain_factory, make_img=False, solver=solver)

    except Exception as e:
        print(e)
        noexcept = False
    assert solver.check_domain(domain) and noexcept
