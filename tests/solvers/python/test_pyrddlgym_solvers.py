import pyRDDLGym_jax.examples.configs
from pyRDDLGym_jax.core.simulator import JaxRDDLSimulator

from skdecide.hub.domain.rddl import RDDLDomain
from skdecide.hub.solver.rddl.rddl import RDDLGurobiSolver, RDDLJaxSolver
from skdecide.utils import load_registered_solver, rollout


def test_pyrddlgymdomain_jax():
    # get solver config
    config_name = "Cartpole_Continuous_gym_drp.cfg"
    config_dir = pyRDDLGym_jax.examples.configs.__path__[0]
    config_path = f"{config_dir}/{config_name}"

    # domain factory (with proper backend and vectorized flag)
    domain_factory = lambda: RDDLDomain(
        rddl_domain="CartPole_Continuous_gym",
        rddl_instance="0",
        backend=JaxRDDLSimulator,
        display_with_pygame=False,
        display_within_jupyter=False,
        vectorized=True,
    )
    solver_factory = lambda: RDDLJaxSolver(
        domain_factory=domain_factory, config=config_path
    )

    # solve
    with solver_factory() as solver:
        solver.solve()
        rollout(domain_factory(), solver, max_steps=100, render=False, verbose=False)


def test_pyrddlgymdomain_gurobi():
    # domain factory (with proper backend and vectorized flag)
    domain_factory = lambda: RDDLDomain(
        rddl_domain="CartPole_Continuous_gym",
        rddl_instance="0",
        display_with_pygame=False,
        display_within_jupyter=False,
    )
    solver_factory = lambda: RDDLGurobiSolver(domain_factory=domain_factory)

    # solve
    with solver_factory() as solver:
        solver.solve()
        rollout(domain_factory(), solver, max_steps=100, render=False, verbose=False)


def test_load_solvers():
    cls = load_registered_solver("RDDLJaxSolver")
    assert cls is RDDLJaxSolver
    cls = load_registered_solver("RDDLGurobiSolver")
    assert cls is RDDLGurobiSolver
