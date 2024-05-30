from skdecide.hub.domain.mastermind import MasterMind
from skdecide.hub.solver.pomcp import POMCP
from skdecide.utils import rollout


def test_pomcp():
    domain_factory = lambda: MasterMind(3, 3)
    domain = domain_factory()
    if POMCP.check_domain(domain):
        with POMCP(domain_factory=domain_factory) as solver:
            solver.solve()
            rollout(
                domain,
                solver,
                num_episodes=1,
                max_steps=100,
                verbose=False,
            )
