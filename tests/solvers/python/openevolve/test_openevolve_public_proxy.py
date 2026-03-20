from typing import Any, Optional

import pytest

from skdecide import D, Domain, Space, Value
from skdecide.builders.domain import (
    DeterministicInitialized,
    DeterministicTransitions,
    Initializable,
    Markovian,
    Sequential,
    SingleAgent,
    TransformedObservable,
    UnrestrictedActions,
)
from skdecide.hub.solver.openevolve.public_proxy import (
    create_public_proxy,
    get_domain_mixin_classes,
)
from skdecide.hub.space.gym import DiscreteSpace

# --- Mock Domain for Testing ---


class D(
    Domain,
    DeterministicInitialized,
    DeterministicTransitions,
    UnrestrictedActions,
    TransformedObservable,
    SingleAgent,
    Sequential,
    Markovian,
):
    T_state = int  # Type of states
    T_observation = int  # Type of observations
    T_event = int  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of logical checks
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class MockDomain(D):
    """A mock domain implementing common scikit-decide capabilities."""

    def __init__(self):
        self._secret_value = 42
        self.public_val = 7
        self.initialized = False

    def _get_initial_state_(self) -> int:
        return 0

    def _get_next_state(self, memory: Any, action: Any) -> Any:
        return memory + action

    def _get_action_space_(self) -> DiscreteSpace:
        return DiscreteSpace(2)

    def reset(self):
        # Initializable.reset calls several private methods
        # This is the "acid test" for our proxy.
        self.initialized = True
        return super().reset()

    def public_compute(self, x):
        return x + self._secret_value

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        return 1.0

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        return False

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return DiscreteSpace(10)

    def _get_observation(
        self,
        state: D.T_state,
        action: Optional[D.T_agent[D.T_concurrency[D.T_event]]] = None,
    ) -> D.T_agent[D.T_observation]:
        return state % 10


# --- Tests ---


def test_mixin_extraction():
    """Verify that we correctly identify scikit-decide mixins."""
    mixins = get_domain_mixin_classes(MockDomain)
    assert DeterministicTransitions in mixins
    assert Initializable in mixins
    assert DeterministicInitialized in mixins
    assert UnrestrictedActions in mixins


def test_proxy_isinstance():
    """Verify the proxy passes scikit-decide capability checks."""
    domain = MockDomain()
    proxy = create_public_proxy(domain)

    assert isinstance(proxy, DeterministicTransitions)
    assert isinstance(proxy, Initializable)
    assert isinstance(proxy, DeterministicInitialized)
    assert isinstance(proxy, UnrestrictedActions)


def test_private_access_restriction():
    """Ensure the proxy raises AttributeError for private members."""
    domain = MockDomain()
    proxy = create_public_proxy(domain)

    with pytest.raises(AttributeError, match="Access to private member"):
        _ = proxy._secret_value

    with pytest.raises(AttributeError, match="Modification of private member"):
        proxy._secret_value = 100


def test_private_access_restriction_for_mixin_methods():
    """Ensure the proxy raises AttributeError for private members."""
    domain = MockDomain()
    proxy = create_public_proxy(domain)

    domain._get_applicable_actions_from(0)
    with pytest.raises(AttributeError, match="Access to private member"):
        _ = proxy._get_applicable_actions_from(0)


def test_method_redirection_and_context():
    """Verify that public methods can still access private data internally."""
    domain = MockDomain()
    proxy = create_public_proxy(domain)

    # This calls a public method that uses self._secret_value internally
    assert proxy.public_compute(10) == 52
    assert proxy.public_val == 7


def test_mixin_public_api_redirect():
    """Verify self.reset() calls the one from the original domain."""
    domain = MockDomain()
    proxy = create_public_proxy(domain)

    obs = proxy.reset()
    assert domain.initialized
    assert proxy.initialized
    assert obs is not None


def test_dir_filtering():
    """Check if tab-completion/discovery only shows public API."""
    domain = MockDomain()
    proxy = create_public_proxy(domain)

    proxy_dir = dir(proxy)
    assert "public_compute" in proxy_dir
    assert "_secret_value" not in proxy_dir
