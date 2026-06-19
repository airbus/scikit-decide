# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for PDDL determinization-based solvers.

These are smoke tests verifying that the solvers can be instantiated
and basic methods can be called without errors.
"""

import os

import pytest

PDDL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "domains", "python", "pddl_domains"
)
TIREWORLD_DOMAIN = os.path.join(PDDL_DIR, "tireworld", "domain.pddl")
TIREWORLD_PROBLEM = os.path.join(PDDL_DIR, "tireworld", "p01.pddl")

pytest.importorskip("skdecide.hub.__skdecide_hub_cpp")

from skdecide.hub.domain.pddl import PPDDLDomain
from skdecide.hub.solver.pddl.ppddldethindsight import (
    FFDetHindsight,
    PPDDLDetHindsight,
)
from skdecide.hub.solver.pddl.ppddlplanmerger import RFF, PPDDLPlanMerger
from skdecide.hub.solver.pddl.ppddlreplan import FFReplan, PPDDLReplan


@pytest.fixture
def tireworld_factory():
    """Create a factory for the Tireworld PPDDL domain."""
    return lambda: PPDDLDomain(TIREWORLD_DOMAIN, TIREWORLD_PROBLEM)


# ========== FFReplan Tests ==========


@pytest.mark.parametrize(
    "determinization",
    ["most_probable_outcome", "all_outcomes", "random_outcome"],
)
@pytest.mark.timeout(30)
def test_ffreplan_instantiation(tireworld_factory, determinization):
    """Test FFReplan can be instantiated with different determinization strategies."""
    solver = FFReplan(
        domain_factory=tireworld_factory,
        determinization=determinization,
        max_replans=10,
        max_steps=100,
        verbose=False,
    )
    assert solver is not None


@pytest.mark.timeout(30)
def test_ffreplan_accessors(tireworld_factory):
    """Test FFReplan introspection methods exist."""
    solver = FFReplan(
        domain_factory=tireworld_factory,
        determinization="most_probable_outcome",
        max_replans=10,
        max_steps=100,
        verbose=False,
    )
    # Verify accessor methods exist
    assert hasattr(solver, "get_plan")
    assert hasattr(solver, "get_nb_replans")
    assert hasattr(solver, "get_nb_steps")
    assert hasattr(solver, "get_solving_time")
    assert hasattr(solver, "get_total_cost")


# ========== PPDDLReplan Tests ==========


@pytest.mark.parametrize(
    "determinization",
    ["most_probable_outcome", "all_outcomes", "random_outcome"],
)
@pytest.mark.timeout(30)
def test_ppddlreplan_instantiation(tireworld_factory, determinization):
    """Test PPDDLReplan can be instantiated with different determinization strategies."""
    solver = PPDDLReplan(
        domain_factory=tireworld_factory,
        determinization=determinization,
        max_replans=10,
        max_steps=100,
        verbose=False,
    )
    assert solver is not None


@pytest.mark.timeout(30)
def test_ppddlreplan_accessors(tireworld_factory):
    """Test PPDDLReplan introspection methods exist."""
    solver = PPDDLReplan(
        domain_factory=tireworld_factory,
        determinization="most_probable_outcome",
        max_replans=10,
        max_steps=100,
        verbose=False,
    )
    # Verify accessor methods exist
    assert hasattr(solver, "get_plan")
    assert hasattr(solver, "get_nb_replans")


# ========== FFDetHindsight Tests ==========


@pytest.mark.parametrize("sample_width", [1, 5])
@pytest.mark.timeout(30)
def test_ffdethindsight_instantiation(tireworld_factory, sample_width):
    """Test FFDetHindsight can be instantiated with different sample widths."""
    solver = FFDetHindsight(
        domain_factory=tireworld_factory,
        sample_width=sample_width,
        max_steps=100,
        verbose=False,
    )
    assert solver is not None


@pytest.mark.timeout(30)
def test_ffdethindsight_accessors(tireworld_factory):
    """Test FFDetHindsight introspection methods exist."""
    solver = FFDetHindsight(
        domain_factory=tireworld_factory,
        sample_width=5,
        max_steps=100,
        verbose=False,
    )
    # Verify accessor methods exist
    assert hasattr(solver, "get_explored_states")
    assert hasattr(solver, "get_terminal_states")


# ========== PPDDLDetHindsight Tests ==========


@pytest.mark.parametrize("sample_width", [1, 5])
@pytest.mark.timeout(30)
def test_ppddldethindsight_instantiation(tireworld_factory, sample_width):
    """Test PPDDLDetHindsight can be instantiated with different sample widths."""
    solver = PPDDLDetHindsight(
        domain_factory=tireworld_factory,
        sample_width=sample_width,
        max_steps=100,
        verbose=False,
    )
    assert solver is not None


@pytest.mark.timeout(30)
def test_ppddldethindsight_accessors(tireworld_factory):
    """Test PPDDLDetHindsight introspection methods exist."""
    solver = PPDDLDetHindsight(
        domain_factory=tireworld_factory,
        sample_width=5,
        max_steps=100,
        verbose=False,
    )
    # Verify accessor methods exist
    assert hasattr(solver, "get_explored_states")


# ========== RFF Tests ==========


@pytest.mark.parametrize(
    "determinization",
    ["most_probable_outcome", "all_outcomes", "random_outcome"],
)
@pytest.mark.timeout(30)
def test_rff_instantiation(tireworld_factory, determinization):
    """Test RFF can be instantiated with different determinization strategies."""
    solver = RFF(
        domain_factory=tireworld_factory,
        determinization=determinization,
        max_iterations=5,
        mc_samples=10,
        max_steps=100,
        verbose=False,
    )
    assert solver is not None


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.timeout(30)
def test_rff_optimization(tireworld_factory, optimize):
    """Test RFF with and without policy graph optimization."""
    solver = RFF(
        domain_factory=tireworld_factory,
        determinization="most_probable_outcome",
        max_iterations=5,
        mc_samples=10,
        optimize_policy_graph=optimize,
        verbose=False,
    )
    assert solver is not None


@pytest.mark.timeout(30)
def test_rff_accessors(tireworld_factory):
    """Test RFF introspection methods exist."""
    solver = RFF(
        domain_factory=tireworld_factory,
        determinization="most_probable_outcome",
        max_iterations=5,
        mc_samples=10,
        verbose=False,
    )
    # Verify accessor methods exist
    assert hasattr(solver, "get_solving_time")


# ========== PPDDLPlanMerger Tests ==========


@pytest.mark.parametrize(
    "determinization",
    ["most_probable_outcome", "all_outcomes"],
)
@pytest.mark.timeout(30)
def test_ppddlplanmerger_instantiation(tireworld_factory, determinization):
    """Test PPDDLPlanMerger can be instantiated with different determinization strategies."""
    solver = PPDDLPlanMerger(
        domain_factory=tireworld_factory,
        determinization=determinization,
        max_iterations=5,
        mc_samples=10,
        max_steps=100,
        verbose=False,
    )
    assert solver is not None


@pytest.mark.timeout(30)
def test_ppddlplanmerger_accessors(tireworld_factory):
    """Test PPDDLPlanMerger introspection methods exist."""
    solver = PPDDLPlanMerger(
        domain_factory=tireworld_factory,
        determinization="most_probable_outcome",
        max_iterations=5,
        mc_samples=10,
        verbose=False,
    )
    # Verify accessor methods exist
    assert hasattr(solver, "get_solving_time")
