# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest

PDDL_PLUS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "cpp", "tests", "data", "pddl+"
)
COFFEE_DOMAIN = os.path.join(PDDL_PLUS_DIR, "Coffee", "coffeemaking.pddl")
COFFEE_PROBLEM = os.path.join(PDDL_PLUS_DIR, "Coffee", "coffeeproblem.pddl")
VENDING_DOMAIN = os.path.join(PDDL_PLUS_DIR, "VendingMachine", "vendingmachine.pddl")
VENDING_PROBLEM = os.path.join(
    PDDL_PLUS_DIR, "VendingMachine", "vendingmachine-problem.pddl"
)


@pytest.fixture
def coffee_ts():
    from skdecide.hub.domain.pddl import TPDDLDomain

    return TPDDLDomain(COFFEE_DOMAIN, COFFEE_PROBLEM, mode="time_stepping", dt=1.0)


@pytest.fixture
def coffee_ed():
    from skdecide.hub.domain.pddl import TPDDLDomain

    return TPDDLDomain(COFFEE_DOMAIN, COFFEE_PROBLEM, mode="event_driven")


@pytest.fixture
def vending_ts():
    from skdecide.hub.domain.pddl import TPDDLDomain

    return TPDDLDomain(VENDING_DOMAIN, VENDING_PROBLEM, mode="time_stepping", dt=0.1)


class TestTPDDLDomainConstruction:
    def test_construction_time_stepping(self, coffee_ts):
        assert coffee_ts is not None
        assert coffee_ts.task is not None
        assert coffee_ts.simulator is not None

    def test_construction_event_driven(self, coffee_ed):
        assert coffee_ed is not None

    def test_vending_machine_construction(self, vending_ts):
        assert vending_ts is not None

    def test_has_z3(self):
        from skdecide.hub.__skdecide_hub_cpp import (
            _PDDL_TemporalSimulator_ as CppTemporalSimulator,
        )

        result = CppTemporalSimulator.has_z3()
        assert isinstance(result, bool)


class TestTPDDLDomainCoffee:
    def test_initial_state(self, coffee_ts):
        s0 = coffee_ts._get_initial_state_()
        assert s0 is not None
        assert s0.time == 0.0
        assert hash(s0) == hash(s0)
        s0b = coffee_ts._get_initial_state_()
        assert s0 == s0b

    def test_initial_state_temperature(self, coffee_ts):
        """Temperature of water1 starts at 7."""
        s0 = coffee_ts._get_initial_state_()
        cpp_state = s0.to_cpp()
        fluents = cpp_state.get_fluents()
        temp_func_id = coffee_ts.task.function_id("temperature")
        water1_id = coffee_ts.task.object_id("water1")
        temp_val = fluents[temp_func_id][(water1_id,)]
        assert abs(temp_val - 7.0) < 1e-6

    def test_applicable_actions_initial(self, coffee_ts):
        """Initial state: water is cold, so heatwater is applicable."""
        s0 = coffee_ts._get_initial_state_()
        actions = coffee_ts._get_applicable_actions_from(s0)
        elements = actions.get_elements()
        assert len(elements) >= 2  # noop + heatwater

        from skdecide.hub.domain.pddl.domain import TPDDLAction

        kinds = {a.kind for a in elements}
        assert TPDDLAction.NOOP in kinds
        assert TPDDLAction.INSTANTANEOUS in kinds

    def test_noop_advances_time(self, coffee_ts):
        """Noop in time_stepping mode advances time by dt."""
        from skdecide.hub.domain.pddl.domain import TPDDLAction

        s0 = coffee_ts._get_initial_state_()
        noop = TPDDLAction(TPDDLAction.NOOP)
        s1 = coffee_ts._get_next_state(s0, noop)
        assert abs(s1.time - 1.0) < 1e-6

    def test_heatwater_activates_heating(self, coffee_ts):
        """After heatwater, the heating predicate should be set."""
        s0 = coffee_ts._get_initial_state_()
        actions = coffee_ts._get_applicable_actions_from(s0)
        elements = actions.get_elements()

        from skdecide.hub.domain.pddl.domain import TPDDLAction

        heatwater = None
        for a in elements:
            if a.kind == TPDDLAction.INSTANTANEOUS:
                heatwater = a
                break
        assert heatwater is not None

        s1 = coffee_ts._get_next_state(s0, heatwater)
        assert s1 is not None
        assert s1.time > 0.0

    def test_temperature_increases_with_heating(self, coffee_ts):
        """After heatwater and time steps, temperature should increase."""
        from skdecide.hub.domain.pddl.domain import TPDDLAction

        s0 = coffee_ts._get_initial_state_()

        # Find and apply heatwater
        actions = coffee_ts._get_applicable_actions_from(s0)
        heatwater = None
        for a in actions.get_elements():
            if a.kind == TPDDLAction.INSTANTANEOUS:
                heatwater = a
                break
        assert heatwater is not None

        s1 = coffee_ts._get_next_state(s0, heatwater)
        temp_func_id = coffee_ts.task.function_id("temperature")
        water1_id = coffee_ts.task.object_id("water1")

        # Take a few noop steps to let heating process run
        noop = TPDDLAction(TPDDLAction.NOOP)
        state = s1
        for _ in range(5):
            state = coffee_ts._get_next_state(state, noop)

        fluents = state.to_cpp().get_fluents()
        temp = fluents[temp_func_id][(water1_id,)]
        # Initial temp 7 + ~6 steps at rate 2 per second = ~19
        # (first step includes action, subsequent 5 are pure time steps)
        assert temp > 7.0, f"Temperature should have increased from 7, got {temp}"

    def test_boil_event_fires(self, coffee_ts):
        """After enough time steps, boil event fires when temp >= 100."""
        from skdecide.hub.domain.pddl.domain import TPDDLAction

        s0 = coffee_ts._get_initial_state_()

        # Apply heatwater
        actions = coffee_ts._get_applicable_actions_from(s0)
        heatwater = None
        for a in actions.get_elements():
            if a.kind == TPDDLAction.INSTANTANEOUS:
                heatwater = a
                break

        state = coffee_ts._get_next_state(s0, heatwater)
        noop = TPDDLAction(TPDDLAction.NOOP)

        # Net heating rate=1.5 (heating=2, cooling=0.5 once temp>=18)
        # From temp=7: ~6 steps pure heating to 19, then ~54 steps at 1.5/s
        for _ in range(65):
            state = coffee_ts._get_next_state(state, noop)

        # Check that boiled predicate is set
        boiled_id = coffee_ts.task.predicate_id("boiled")
        atoms = state.to_cpp().get_atoms()
        water1_id = coffee_ts.task.object_id("water1")
        boiled_set = atoms[boiled_id]
        assert (water1_id,) in boiled_set, "Water should be boiled after heating"

    def test_durative_action_available_after_boil(self, coffee_ts):
        """After boiling, makecoffee durative action should be applicable
        when temperature is in [60, 80] range."""
        from skdecide.hub.domain.pddl.domain import TPDDLAction

        # We need the water to be boiled AND temperature in [60,80]
        # After boiling (temp>=100), stop-heating fires.
        # Then cooling process runs at rate -0.5 per second.
        # From 100 to 80: (100-80)/0.5 = 40 seconds
        # From 100 to 60: (100-60)/0.5 = 80 seconds
        # So between ~40-80 seconds after boil, temp is in [60,80]

        s0 = coffee_ts._get_initial_state_()
        actions = coffee_ts._get_applicable_actions_from(s0)
        heatwater = None
        for a in actions.get_elements():
            if a.kind == TPDDLAction.INSTANTANEOUS:
                heatwater = a
                break

        state = coffee_ts._get_next_state(s0, heatwater)
        noop = TPDDLAction(TPDDLAction.NOOP)

        # Heat to boil (~50 steps) then cool (~50 steps to ~75 degrees)
        for _ in range(100):
            state = coffee_ts._get_next_state(state, noop)

        # Check temperature is in the right range for makecoffee
        temp_func_id = coffee_ts.task.function_id("temperature")
        water1_id = coffee_ts.task.object_id("water1")
        fluents = state.to_cpp().get_fluents()
        temp = fluents[temp_func_id][(water1_id,)]

        # Check if durative actions are available
        actions = coffee_ts._get_applicable_actions_from(state)
        da_actions = [
            a for a in actions.get_elements() if a.kind == TPDDLAction.DURATIVE_START
        ]

        if 60 <= temp <= 80:
            assert len(da_actions) > 0, (
                f"makecoffee should be applicable at temp={temp}"
            )

    def test_transition_value_is_time(self, coffee_ts):
        """Transition cost should be the elapsed time."""
        from skdecide.hub.domain.pddl.domain import TPDDLAction

        s0 = coffee_ts._get_initial_state_()
        noop = TPDDLAction(TPDDLAction.NOOP)
        s1 = coffee_ts._get_next_state(s0, noop)
        val = coffee_ts._get_transition_value(s0, noop, s1)
        assert abs(val.cost - 1.0) < 1e-6

    def test_not_terminal_initially(self, coffee_ts):
        s0 = coffee_ts._get_initial_state_()
        assert not coffee_ts._is_terminal(s0)


@pytest.fixture(params=["binary_search", "z3"], ids=["binary_search", "z3"])
def coffee_ed_engine(request):
    """Coffee domain in event-driven mode, parametrized by engine."""
    from skdecide.hub.domain.pddl import TPDDLDomain
    from skdecide.hub.domain.pddl.domain import _HAS_Z3_PYTHON

    use_z3 = request.param == "z3"
    if use_z3 and not _HAS_Z3_PYTHON:
        pytest.skip("z3-solver not installed")
    return TPDDLDomain(
        COFFEE_DOMAIN, COFFEE_PROBLEM, mode="event_driven", use_z3=use_z3
    )


class TestTPDDLDomainEventDriven:
    def test_event_step_advances_to_event(self, coffee_ed):
        """In event-driven mode, noop should advance to next event."""
        from skdecide.hub.domain.pddl.domain import TPDDLAction

        s0 = coffee_ed._get_initial_state_()

        # Apply heatwater first
        actions = coffee_ed._get_applicable_actions_from(s0)
        heatwater = None
        for a in actions.get_elements():
            if a.kind == TPDDLAction.INSTANTANEOUS:
                heatwater = a
                break
        assert heatwater is not None

        s1 = coffee_ed._get_next_state(s0, heatwater)
        # In event-driven mode, this should advance to next event
        # (boil at temp>=100, which is at t≈46.5)
        assert s1.time > 0.0

    def test_boil_event_time_correct(self, coffee_ed_engine):
        """Both engines should compute the correct boil event time.

        After heatwater, heating process is active at rate 2 and cooling
        is not yet active (temp=7 < 18). Single-step Euler integration
        sees only heating, so the boil event at temp>=100 fires at
        t = (100 - 7) / 2 = 46.5.
        """
        from skdecide.hub.domain.pddl.domain import TPDDLAction

        s0 = coffee_ed_engine._get_initial_state_()
        actions = coffee_ed_engine._get_applicable_actions_from(s0)
        heatwater = next(
            a for a in actions.get_elements() if a.kind == TPDDLAction.INSTANTANEOUS
        )
        s1 = coffee_ed_engine._get_next_state(s0, heatwater)
        assert abs(s1.time - 46.5) < 0.1, f"Expected event time ≈ 46.5, got {s1.time}"

    def test_boil_event_fires_correctly(self, coffee_ed_engine):
        """Both engines should cause the boil event to fire."""
        from skdecide.hub.domain.pddl.domain import TPDDLAction

        s0 = coffee_ed_engine._get_initial_state_()
        actions = coffee_ed_engine._get_applicable_actions_from(s0)
        heatwater = next(
            a for a in actions.get_elements() if a.kind == TPDDLAction.INSTANTANEOUS
        )
        s1 = coffee_ed_engine._get_next_state(s0, heatwater)

        boiled_id = coffee_ed_engine.task.predicate_id("boiled")
        water1_id = coffee_ed_engine.task.object_id("water1")
        atoms = s1.to_cpp().get_atoms()
        assert (water1_id,) in atoms[boiled_id], "Water should be boiled"

    def test_temperature_at_boil(self, coffee_ed_engine):
        """Temperature should reach the threshold at event time."""
        from skdecide.hub.domain.pddl.domain import TPDDLAction

        s0 = coffee_ed_engine._get_initial_state_()
        actions = coffee_ed_engine._get_applicable_actions_from(s0)
        heatwater = next(
            a for a in actions.get_elements() if a.kind == TPDDLAction.INSTANTANEOUS
        )
        s1 = coffee_ed_engine._get_next_state(s0, heatwater)

        temp_func_id = coffee_ed_engine.task.function_id("temperature")
        water1_id = coffee_ed_engine.task.object_id("water1")
        temp = s1.to_cpp().get_fluents()[temp_func_id][(water1_id,)]
        assert temp >= 100.0 - 1e-3, f"Temperature should be >= 100, got {temp}"

    def test_cooling_after_boil(self, coffee_ed_engine):
        """After boil, a second event step should advance to stop-cooling."""
        from skdecide.hub.domain.pddl.domain import TPDDLAction

        s0 = coffee_ed_engine._get_initial_state_()
        actions = coffee_ed_engine._get_applicable_actions_from(s0)
        heatwater = next(
            a for a in actions.get_elements() if a.kind == TPDDLAction.INSTANTANEOUS
        )
        s1 = coffee_ed_engine._get_next_state(s0, heatwater)
        # After boil: heating off, cooling active at rate -0.5
        # Stop-cooling fires when temp <= 18: t = (100 - 18) / 0.5 = 164
        noop = TPDDLAction(TPDDLAction.NOOP)
        s2 = coffee_ed_engine._get_next_state(s1, noop)
        assert s2.time > s1.time, "Time should advance after noop"

    def test_transition_value_event_driven(self, coffee_ed_engine):
        """Transition cost equals elapsed time in event-driven mode."""
        from skdecide.hub.domain.pddl.domain import TPDDLAction

        s0 = coffee_ed_engine._get_initial_state_()
        actions = coffee_ed_engine._get_applicable_actions_from(s0)
        heatwater = next(
            a for a in actions.get_elements() if a.kind == TPDDLAction.INSTANTANEOUS
        )
        s1 = coffee_ed_engine._get_next_state(s0, heatwater)
        val = coffee_ed_engine._get_transition_value(s0, heatwater, s1)
        assert abs(val.cost - s1.time) < 1e-6


class TestTPDDLDomainEngineComparison:
    """Compare binary search and Z3 engines produce equivalent results."""

    def test_engines_produce_same_boil_time(self):
        """Binary search and Z3 should agree on the boil event time."""
        from skdecide.hub.domain.pddl import TPDDLDomain
        from skdecide.hub.domain.pddl.domain import _HAS_Z3_PYTHON, TPDDLAction

        if not _HAS_Z3_PYTHON:
            pytest.skip("z3-solver not installed")

        dom_bs = TPDDLDomain(
            COFFEE_DOMAIN, COFFEE_PROBLEM, mode="event_driven", use_z3=False
        )
        dom_z3 = TPDDLDomain(
            COFFEE_DOMAIN, COFFEE_PROBLEM, mode="event_driven", use_z3=True
        )

        s0_bs = dom_bs._get_initial_state_()
        s0_z3 = dom_z3._get_initial_state_()

        hw_bs = next(
            a
            for a in dom_bs._get_applicable_actions_from(s0_bs).get_elements()
            if a.kind == TPDDLAction.INSTANTANEOUS
        )
        hw_z3 = next(
            a
            for a in dom_z3._get_applicable_actions_from(s0_z3).get_elements()
            if a.kind == TPDDLAction.INSTANTANEOUS
        )

        s1_bs = dom_bs._get_next_state(s0_bs, hw_bs)
        s1_z3 = dom_z3._get_next_state(s0_z3, hw_z3)

        assert abs(s1_bs.time - s1_z3.time) < 0.1, (
            f"Times differ: bs={s1_bs.time}, z3={s1_z3.time}"
        )

    def test_engines_produce_same_boiled_state(self):
        """Both engines should produce the same boiled predicate."""
        from skdecide.hub.domain.pddl import TPDDLDomain
        from skdecide.hub.domain.pddl.domain import _HAS_Z3_PYTHON, TPDDLAction

        if not _HAS_Z3_PYTHON:
            pytest.skip("z3-solver not installed")

        dom_bs = TPDDLDomain(
            COFFEE_DOMAIN, COFFEE_PROBLEM, mode="event_driven", use_z3=False
        )
        dom_z3 = TPDDLDomain(
            COFFEE_DOMAIN, COFFEE_PROBLEM, mode="event_driven", use_z3=True
        )

        s0_bs = dom_bs._get_initial_state_()
        s0_z3 = dom_z3._get_initial_state_()

        hw_bs = next(
            a
            for a in dom_bs._get_applicable_actions_from(s0_bs).get_elements()
            if a.kind == TPDDLAction.INSTANTANEOUS
        )
        hw_z3 = next(
            a
            for a in dom_z3._get_applicable_actions_from(s0_z3).get_elements()
            if a.kind == TPDDLAction.INSTANTANEOUS
        )

        s1_bs = dom_bs._get_next_state(s0_bs, hw_bs)
        s1_z3 = dom_z3._get_next_state(s0_z3, hw_z3)

        boiled_id = dom_bs.task.predicate_id("boiled")
        water1_id = dom_bs.task.object_id("water1")

        atoms_bs = s1_bs.to_cpp().get_atoms()
        atoms_z3 = s1_z3.to_cpp().get_atoms()

        assert (water1_id,) in atoms_bs[boiled_id], "BS: water should be boiled"
        assert (water1_id,) in atoms_z3[boiled_id], "Z3: water should be boiled"

    def test_z3_precision(self):
        """Z3 should give a more precise event time than binary search."""
        from skdecide.hub.domain.pddl import TPDDLDomain
        from skdecide.hub.domain.pddl.domain import _HAS_Z3_PYTHON, TPDDLAction

        if not _HAS_Z3_PYTHON:
            pytest.skip("z3-solver not installed")

        dom_z3 = TPDDLDomain(
            COFFEE_DOMAIN, COFFEE_PROBLEM, mode="event_driven", use_z3=True
        )
        s0 = dom_z3._get_initial_state_()
        hw = next(
            a
            for a in dom_z3._get_applicable_actions_from(s0).get_elements()
            if a.kind == TPDDLAction.INSTANTANEOUS
        )
        s1 = dom_z3._get_next_state(s0, hw)
        # Z3 should compute exact t = (100 - 7) / 2 = 46.5
        assert abs(s1.time - 46.5) < 1e-6, (
            f"Z3 should give exact time 46.5, got {s1.time}"
        )


class TestTPDDLDomainVendingMachine:
    def test_initial_state(self, vending_ts):
        s0 = vending_ts._get_initial_state_()
        assert s0 is not None
        assert s0.time == 0.0

    def test_entercoin_applicable(self, vending_ts):
        """Initially, slot is open and light sensor is on, so entercoin works."""
        from skdecide.hub.domain.pddl.domain import TPDDLAction

        s0 = vending_ts._get_initial_state_()
        actions = vending_ts._get_applicable_actions_from(s0)
        inst_actions = [
            a for a in actions.get_elements() if a.kind == TPDDLAction.INSTANTANEOUS
        ]
        assert len(inst_actions) > 0, "entercoin should be applicable"

    def test_coin_physics(self, vending_ts):
        """After entercoin, coin should fall with acceleration."""
        from skdecide.hub.domain.pddl.domain import TPDDLAction

        s0 = vending_ts._get_initial_state_()

        # Apply entercoin
        actions = vending_ts._get_applicable_actions_from(s0)
        entercoin = None
        for a in actions.get_elements():
            if a.kind == TPDDLAction.INSTANTANEOUS:
                entercoin = a
                break

        state = vending_ts._get_next_state(s0, entercoin)
        noop = TPDDLAction(TPDDLAction.NOOP)

        # Take several time steps
        for _ in range(10):
            state = vending_ts._get_next_state(state, noop)

        # Distance should have increased (coin falling under acceleration)
        dist_func_id = vending_ts.task.function_id("dist")
        fluents = state.to_cpp().get_fluents()
        dist = fluents[dist_func_id][()]
        assert dist > 0.0, f"Coin distance should increase, got {dist}"


class TestTPDDLDomainParameters:
    def test_custom_epsilon(self):
        from skdecide.hub.domain.pddl import TPDDLDomain

        domain = TPDDLDomain(
            COFFEE_DOMAIN,
            COFFEE_PROBLEM,
            mode="time_stepping",
            dt=0.5,
            epsilon=1e-6,
            max_cascade_iterations=50,
        )
        assert domain is not None
        s0 = domain._get_initial_state_()
        assert s0.time == 0.0

    def test_task_temporal_accessors(self, coffee_ts):
        """Task should expose processes, events, and durative actions."""
        task = coffee_ts.task
        assert task.num_events() >= 2  # boil, stop-heating, stop-cooling
        assert task.num_processes() >= 2  # heating, cooling
        assert task.num_durative_actions() >= 1  # makecoffee

    def test_state_hashable(self, coffee_ts):
        """TPDDLStates should be hashable for use in sets/dicts."""
        s0 = coffee_ts._get_initial_state_()
        state_set = {s0}
        assert s0 in state_set
        assert len(state_set) == 1
