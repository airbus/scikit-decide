from skdecide import Domain, RLDomain
from skdecide.core import autocast_all


def test_get_agents_with_autocast():
    class MyDomain(RLDomain):
        """Partially defined single agent domain."""

        def _get_observation_space_(self):
            return {1, 2, 3}

    # single agent domain
    domain = MyDomain()

    # up-cast as multi-agent domain
    upcast_domain = MyDomain()
    autocast_all(upcast_domain, RLDomain, Domain)

    assert domain.get_agents() == Domain.get_agents(upcast_domain)
    assert domain.get_agents() != Domain.get_agents(domain)
