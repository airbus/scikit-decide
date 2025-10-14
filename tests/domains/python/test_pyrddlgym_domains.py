import os
import shutil

import matplotlib
from stable_baselines3 import PPO as SB3_PPO

from skdecide.hub.domain.rddl import (
    RDDLDomain,
    RDDLDomainRL,
    RDDLDomainSimplifiedSpaces,
)
from skdecide.hub.solver.cgp import CGP
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.utils import load_registered_domain, rollout


def test_pyrddlgymdomain_sb3():
    matplotlib.use("agg")
    movie_name = "test-sb3"
    movie_path = f"rddl_movies/{movie_name}"
    domain_factory = lambda: RDDLDomainRL(
        rddl_domain="CartPole_Continuous_gym",
        rddl_instance="0",
        movie_name=movie_name,
        display_with_pygame=False,
        display_within_jupyter=False,
    )
    domain = domain_factory()
    domain.reset()
    domain.render()

    solver_factory = lambda: StableBaseline(
        domain_factory=domain_factory,
        algo_class=SB3_PPO,
        baselines_policy="MultiInputPolicy",
        learn_config={"total_timesteps": 100},
        verbose=0,
    )

    shutil.rmtree(movie_path)

    with solver_factory() as solver:
        solver.solve()
        rollout(domain_factory(), solver, max_steps=100, render=True, verbose=False)

    assert os.path.isdir(movie_path)


def test_pyrddlgymdomainsimp_cgp():
    domain_factory = lambda: RDDLDomainSimplifiedSpaces(
        rddl_domain="CartPole_Continuous_gym",
        rddl_instance="0",
        display_with_pygame=False,
        display_within_jupyter=False,
    )
    domain = domain_factory()
    domain.reset()
    domain.render()

    solver_factory = lambda: CGP(
        domain_factory=domain_factory, folder_name="TEMP_CGP", n_it=5, verbose=False
    )

    with solver_factory() as solver:
        solver.solve()
        rollout(domain_factory(), solver, max_steps=100, render=True, verbose=False)


def test_load_rddldomain():
    cls = load_registered_domain("RDDLDomain")
    assert cls is RDDLDomain
    cls = load_registered_domain("RDDLDomainRL")
    assert cls is RDDLDomainRL
    cls = load_registered_domain("RDDLDomainSimplifiedSpaces")
    assert cls is RDDLDomainSimplifiedSpaces
