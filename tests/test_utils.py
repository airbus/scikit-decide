from skdecide.hub.domain.maze import Maze
from skdecide.hub.solver.lazy_astar import LazyAstar
from skdecide.utils import (
    get_registered_domains,
    get_registered_solvers,
    load_registered_domain,
    load_registered_solver,
)


def test_get_registered_domains():
    domains = get_registered_domains()
    assert "Maze" in domains


def test_load_registered_domain():
    domain_class = load_registered_domain("NotExistingDomain")
    assert domain_class is None
    maze_class = load_registered_domain("Maze")
    assert maze_class is Maze


def test_get_registered_solvers():
    domains = get_registered_solvers()
    assert "LazyAstar" in domains


def test_load_registered_solver():
    solver_class = load_registered_solver("NotExistingSolver")
    assert solver_class is None
    lazyastar_class = load_registered_solver("LazyAstar")
    assert lazyastar_class is LazyAstar
