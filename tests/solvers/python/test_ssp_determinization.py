# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for generic SSP determinization-based solvers.

These are smoke tests verifying that the solvers can be instantiated
and basic methods can be called without errors.
"""

import sys
from pathlib import Path

import pytest

# Add tests/domains/python to path so we can import stochastic_grid
test_domains_path = Path(__file__).parent.parent.parent / "domains" / "python"
sys.path.insert(0, str(test_domains_path))

from stochastic_grid import StochasticGridDomain

pytest.importorskip("skdecide.hub.__skdecide_hub_cpp")

from skdecide.core import Value
from skdecide.hub.solver.sspdethindsight import SSPDetHindsight
from skdecide.hub.solver.sspplanmerger import SSPPlanMerger
from skdecide.hub.solver.sspreplan import SSPReplan


@pytest.fixture
def grid_domain_factory():
    """Create a factory for the stochastic grid domain."""
    return lambda: StochasticGridDomain(size=5)


def zero_heuristic(domain, state):
    """Zero heuristic for testing."""
    return Value(cost=0.0)


# ========== SSPReplan Tests ==========


@pytest.mark.parametrize(
    "determinization",
    ["most_probable_outcome", "all_outcomes", "random_outcome"],
)
@pytest.mark.timeout(30)
def test_sspreplan_instantiation(grid_domain_factory, determinization):
    """Test SSPReplan can be instantiated with different determinization strategies."""
    solver = SSPReplan(
        domain_factory=grid_domain_factory,
        heuristic=zero_heuristic,
        determinization=determinization,
        max_replans=10,
        max_steps=100,
        verbose=False,
    )
    assert solver is not None
    solver.close()


@pytest.mark.timeout(30)
def test_sspreplan_accessors(grid_domain_factory):
    """Test SSPReplan introspection methods exist and return expected types."""
    with SSPReplan(
        domain_factory=grid_domain_factory,
        heuristic=zero_heuristic,
        determinization="most_probable_outcome",
        max_replans=10,
        max_steps=100,
        verbose=False,
    ) as solver:
        # Verify accessor methods exist and return correct types
        assert hasattr(solver, "get_plan")
        assert hasattr(solver, "get_nb_replans")
        assert hasattr(solver, "get_nb_steps")
        assert hasattr(solver, "get_solving_time")
        assert hasattr(solver, "get_total_cost")


# ========== SSPDetHindsight Tests ==========


@pytest.mark.parametrize("sample_width", [1, 5])
@pytest.mark.timeout(30)
def test_sspdethindsight_instantiation(grid_domain_factory, sample_width):
    """Test SSPDetHindsight can be instantiated with different sample widths."""
    solver = SSPDetHindsight(
        domain_factory=grid_domain_factory,
        heuristic=zero_heuristic,
        sample_width=sample_width,
        max_steps=100,
        verbose=False,
    )
    assert solver is not None
    solver.close()


@pytest.mark.timeout(30)
def test_sspdethindsight_accessors(grid_domain_factory):
    """Test SSPDetHindsight introspection methods exist."""
    with SSPDetHindsight(
        domain_factory=grid_domain_factory,
        heuristic=zero_heuristic,
        sample_width=5,
        max_steps=100,
        verbose=False,
    ) as solver:
        # Verify accessor methods exist
        assert hasattr(solver, "get_explored_states")
        assert hasattr(solver, "get_terminal_states")
        assert hasattr(solver, "get_nb_steps")
        assert hasattr(solver, "get_solving_time")


# ========== SSPPlanMerger Tests ==========


@pytest.mark.parametrize(
    "determinization",
    ["most_probable_outcome", "all_outcomes", "random_outcome"],
)
@pytest.mark.timeout(30)
def test_sspplanmerger_instantiation(grid_domain_factory, determinization):
    """Test SSPPlanMerger can be instantiated with different determinization strategies."""
    solver = SSPPlanMerger(
        domain_factory=grid_domain_factory,
        heuristic=zero_heuristic,
        determinization=determinization,
        max_iterations=5,
        mc_samples=10,
        max_steps=100,
        verbose=False,
    )
    assert solver is not None
    solver.close()


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.timeout(30)
def test_sspplanmerger_optimization(grid_domain_factory, optimize):
    """Test SSPPlanMerger with and without policy graph optimization."""
    solver = SSPPlanMerger(
        domain_factory=grid_domain_factory,
        heuristic=zero_heuristic,
        determinization="most_probable_outcome",
        max_iterations=5,
        mc_samples=10,
        optimize_policy_graph=optimize,
        verbose=False,
    )
    assert solver is not None
    solver.close()


@pytest.mark.timeout(30)
def test_sspplanmerger_accessors(grid_domain_factory):
    """Test SSPPlanMerger introspection methods exist."""
    with SSPPlanMerger(
        domain_factory=grid_domain_factory,
        heuristic=zero_heuristic,
        determinization="most_probable_outcome",
        max_iterations=5,
        mc_samples=10,
        verbose=False,
    ) as solver:
        # Verify accessor methods exist
        assert hasattr(solver, "get_solving_time")
