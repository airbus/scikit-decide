# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest

PDDL_DOMAINS_DIR = os.path.join(os.path.dirname(__file__), "pddl_domains")
BLOCKS_DIR = os.path.join(PDDL_DOMAINS_DIR, "blocks")
AGRICOLA_DIR = os.path.join(PDDL_DOMAINS_DIR, "agricola-opt18")
TIREWORLD_DIR = os.path.join(PDDL_DOMAINS_DIR, "tireworld")


@pytest.fixture
def pddl_module():
    from skdecide.hub.domain.pddl import pddl

    return pddl


@pytest.fixture
def blocks_domain_file():
    return os.path.join(BLOCKS_DIR, "domain.pddl")


@pytest.fixture
def blocks_problem_file():
    return os.path.join(BLOCKS_DIR, "probBLOCKS-3-0.pddl")


@pytest.fixture
def agricola_domain_file():
    return os.path.join(AGRICOLA_DIR, "domain.pddl")


@pytest.fixture
def agricola_problem_file():
    return os.path.join(AGRICOLA_DIR, "p01.pddl")


@pytest.fixture
def tireworld_domain_file():
    return os.path.join(TIREWORLD_DIR, "domain.pddl")


@pytest.fixture
def tireworld_problem_file():
    return os.path.join(TIREWORLD_DIR, "p01.pddl")


class TestPDDLBasicLoading:
    def test_create_pddl_instance(self, pddl_module):
        p = pddl_module.PDDL()
        assert p.get_domains() == []
        assert p.get_problems() == []

    def test_load_domain_only(self, pddl_module, blocks_domain_file):
        p = pddl_module.PDDL()
        p.load([blocks_domain_file])
        assert len(p.get_domains()) == 1
        assert len(p.get_problems()) == 0

    def test_load_domain_and_problem(
        self, pddl_module, blocks_domain_file, blocks_problem_file
    ):
        p = pddl_module.PDDL()
        p.load([blocks_domain_file, blocks_problem_file])
        assert len(p.get_domains()) == 1
        assert len(p.get_problems()) == 1

    def test_load_nonexistent_file(self, pddl_module):
        p = pddl_module.PDDL()
        with pytest.raises(RuntimeError):
            p.load(["/nonexistent/path/domain.pddl"])

    def test_load_multiple_domains(
        self, pddl_module, blocks_domain_file, agricola_domain_file
    ):
        p = pddl_module.PDDL()
        p.load([blocks_domain_file, agricola_domain_file])
        assert len(p.get_domains()) == 2


class TestPDDLDomainStructure:
    @pytest.fixture
    def blocks_domain(self, pddl_module, blocks_domain_file):
        p = pddl_module.PDDL()
        p.load([blocks_domain_file])
        return p.get_domains()[0]

    def test_domain_name(self, blocks_domain):
        assert blocks_domain.get_name().lower() == "blocks"

    def test_domain_requirements(self, blocks_domain):
        reqs = blocks_domain.get_requirements()
        assert reqs is not None
        assert reqs.has_strips()

    def test_domain_predicates(self, blocks_domain):
        predicates = blocks_domain.get_predicates()
        pred_names = {p.get_name().lower() for p in predicates}
        assert pred_names == {"on", "ontable", "clear", "handempty", "holding"}

    def test_predicate_variables(self, blocks_domain):
        predicates = blocks_domain.get_predicates()
        pred_map = {p.get_name().lower(): p for p in predicates}
        on_pred = pred_map["on"]
        assert len(on_pred.get_variables()) == 2
        handempty_pred = pred_map["handempty"]
        assert len(handempty_pred.get_variables()) == 0

    def test_domain_actions_count(self, blocks_domain):
        actions = blocks_domain.get_actions()
        assert len(actions) == 4

    def test_domain_action_by_name(self, blocks_domain):
        for name in ["pick-up", "put-down", "stack", "unstack"]:
            action = blocks_domain.get_action(name)
            assert action is not None

    def test_action_parameters(self, blocks_domain):
        pickup = blocks_domain.get_action("pick-up")
        assert len(pickup.get_variables()) == 1
        stack = blocks_domain.get_action("stack")
        assert len(stack.get_variables()) == 2

    def test_action_precondition(self, blocks_domain, pddl_module):
        pickup = blocks_domain.get_action("pick-up")
        precond = pickup.get_condition()
        assert precond is not None
        assert isinstance(precond, pddl_module.ConjunctionFormula)
        formulas = precond.get_formulas()
        assert len(formulas) == 3

    def test_action_effect(self, blocks_domain, pddl_module):
        pickup = blocks_domain.get_action("pick-up")
        effect = pickup.get_effect()
        assert effect is not None
        assert isinstance(effect, pddl_module.ConjunctionEffect)
        effects = effect.get_effects()
        assert len(effects) == 4


class TestPDDLProblemStructure:
    @pytest.fixture
    def blocks_parsed(self, pddl_module, blocks_domain_file, blocks_problem_file):
        p = pddl_module.PDDL()
        p.load([blocks_domain_file, blocks_problem_file])
        return p

    def test_problem_name(self, blocks_parsed):
        problem = blocks_parsed.get_problems()[0]
        assert problem.get_name().lower() == "blocks-3-0"

    def test_problem_domain_reference(self, blocks_parsed):
        problem = blocks_parsed.get_problems()[0]
        domain = problem.get_domain()
        assert domain is not None
        assert domain.get_name().lower() == "blocks"

    def test_problem_objects(self, blocks_parsed):
        problem = blocks_parsed.get_problems()[0]
        objects = problem.get_objects()
        obj_names = {o.get_name().upper() for o in objects}
        assert obj_names == {"A", "B", "C"}

    def test_problem_initial_state(self, blocks_parsed, pddl_module):
        problem = blocks_parsed.get_problems()[0]
        init = problem.get_initial_effect()
        assert init is not None
        assert isinstance(init, pddl_module.ConjunctionEffect)
        effects = init.get_effects()
        assert len(effects) == 6

    def test_problem_goal(self, blocks_parsed, pddl_module):
        problem = blocks_parsed.get_problems()[0]
        goal = problem.get_goal()
        assert goal is not None
        assert isinstance(goal, pddl_module.PredicateFormula)
        assert goal.get_name().lower() == "on"
        assert len(goal.get_terms()) == 2


class TestPDDLTyping:
    @pytest.fixture
    def agricola_parsed(self, pddl_module, agricola_domain_file, agricola_problem_file):
        p = pddl_module.PDDL()
        p.load([agricola_domain_file, agricola_problem_file])
        return p

    def test_typed_domain_requirements(self, agricola_parsed):
        domain = agricola_parsed.get_domains()[0]
        reqs = domain.get_requirements()
        assert reqs.has_typing()
        assert reqs.has_negative_preconditions()
        assert reqs.has_action_costs()

    def test_type_hierarchy(self, agricola_parsed):
        domain = agricola_parsed.get_domains()[0]
        types = domain.get_types()
        type_names = {t.get_name().lower() for t in types}
        for expected in [
            "actiontag",
            "goods",
            "stage",
            "round",
            "worker",
            "animal",
            "vegetable",
        ]:
            assert expected in type_names, f"Missing type: {expected}"

    def test_typed_predicates(self, agricola_parsed):
        domain = agricola_parsed.get_domains()[0]
        predicates = domain.get_predicates()
        pred_map = {p.get_name().lower(): p for p in predicates}
        assert "next_stage" in pred_map
        next_stage = pred_map["next_stage"]
        assert len(next_stage.get_variables()) == 2

    def test_domain_functions(self, agricola_parsed):
        domain = agricola_parsed.get_domains()[0]
        functions = domain.get_functions()
        func_names = {f.get_name().lower() for f in functions}
        assert "total-cost" in func_names
        assert "group_worker_cost" in func_names

    def test_domain_constants(self, agricola_parsed):
        domain = agricola_parsed.get_domains()[0]
        objects = domain.get_objects()
        obj_names = {o.get_name().lower() for o in objects}
        assert "num0" in obj_names
        assert "noworker" in obj_names
        assert "sheep" in obj_names
        assert "wood" in obj_names

    def test_many_actions(self, agricola_parsed):
        domain = agricola_parsed.get_domains()[0]
        actions = domain.get_actions()
        assert len(actions) == 22

    def test_action_by_name(self, agricola_parsed):
        domain = agricola_parsed.get_domains()[0]
        action = domain.get_action("take_food")
        assert action is not None
        action = domain.get_action("plow_field")
        assert action is not None

    def test_problem_objects_typed(self, agricola_parsed):
        problem = agricola_parsed.get_problems()[0]
        objects = problem.get_objects()
        obj_names = {o.get_name().lower() for o in objects}
        assert "num1" in obj_names
        assert "stage1" in obj_names
        assert "round1" in obj_names
        assert "worker1" in obj_names

    def test_problem_metric(self, agricola_parsed, pddl_module):
        problem = agricola_parsed.get_problems()[0]
        metric = problem.get_metric()
        assert metric is not None
        assert isinstance(metric, pddl_module.MinimizeExpression)


class TestPDDLFormulas:
    @pytest.fixture
    def blocks_domain(self, pddl_module, blocks_domain_file):
        p = pddl_module.PDDL()
        p.load([blocks_domain_file])
        return p.get_domains()[0]

    def test_conjunction_formula(self, blocks_domain, pddl_module):
        action = blocks_domain.get_action("pick-up")
        precond = action.get_condition()
        assert isinstance(precond, pddl_module.ConjunctionFormula)
        formulas = precond.get_formulas()
        assert len(formulas) == 3
        for f in formulas:
            assert isinstance(f, pddl_module.PredicateFormula)

    def test_predicate_formula_name(self, blocks_domain):
        action = blocks_domain.get_action("pick-up")
        precond = action.get_condition()
        formulas = precond.get_formulas()
        pred_names = {f.get_name().lower() for f in formulas}
        assert pred_names == {"clear", "ontable", "handempty"}

    def test_predicate_formula_terms(self, blocks_domain):
        action = blocks_domain.get_action("pick-up")
        precond = action.get_condition()
        formulas = precond.get_formulas()
        for f in formulas:
            name = f.get_name().lower()
            if name == "handempty":
                assert len(f.get_terms()) == 0
            else:
                assert len(f.get_terms()) == 1


class TestPDDLEffects:
    @pytest.fixture
    def blocks_domain(self, pddl_module, blocks_domain_file):
        p = pddl_module.PDDL()
        p.load([blocks_domain_file])
        return p.get_domains()[0]

    def test_conjunction_effect(self, blocks_domain, pddl_module):
        action = blocks_domain.get_action("pick-up")
        effect = action.get_effect()
        assert isinstance(effect, pddl_module.ConjunctionEffect)

    def test_effect_contents(self, blocks_domain, pddl_module):
        action = blocks_domain.get_action("pick-up")
        effect = action.get_effect()
        effects = effect.get_effects()
        assert len(effects) == 4
        neg_count = sum(isinstance(e, pddl_module.NegationEffect) for e in effects)
        pred_count = sum(isinstance(e, pddl_module.PredicateEffect) for e in effects)
        assert neg_count == 3
        assert pred_count == 1

    def test_negation_effect_inner(self, blocks_domain, pddl_module):
        action = blocks_domain.get_action("pick-up")
        effects = action.get_effect().get_effects()
        for e in effects:
            if isinstance(e, pddl_module.NegationEffect):
                inner = e.get_effect()
                assert isinstance(inner, pddl_module.PredicateEffect)
                break

    def test_predicate_effect_name(self, blocks_domain, pddl_module):
        action = blocks_domain.get_action("pick-up")
        effects = action.get_effect().get_effects()
        for e in effects:
            if isinstance(e, pddl_module.PredicateEffect):
                assert e.get_name().lower() == "holding"

    def test_assignment_effect(self, pddl_module, agricola_domain_file):
        p = pddl_module.PDDL()
        p.load([agricola_domain_file])
        domain = p.get_domains()[0]
        action = domain.get_action("take_food")
        effect = action.get_effect()
        assert isinstance(effect, pddl_module.ConjunctionEffect)
        effects = effect.get_effects()
        has_increase = any(isinstance(e, pddl_module.IncreaseEffect) for e in effects)
        assert has_increase


class TestPDDLOptionalEffect:
    def test_action_without_effect(self, pddl_module, tmp_path):
        domain_file = tmp_path / "no_effect_domain.pddl"
        domain_file.write_text(
            "(define (domain noeffect)\n"
            "  (:requirements :strips)\n"
            "  (:predicates (p))\n"
            "  (:action noop\n"
            "    :parameters ()\n"
            "    :precondition (p)\n"
            "  )\n"
            ")"
        )
        p = pddl_module.PDDL()
        p.load([str(domain_file)])
        domain = p.get_domains()[0]
        action = domain.get_action("noop")
        effect = action.get_effect()
        assert effect is not None
        assert isinstance(effect, pddl_module.ConjunctionEffect)
        assert len(effect.get_effects()) == 0


class TestPDDLReaderWrapper:
    def test_reader_constructor_with_files(
        self, pddl_module, blocks_domain_file, blocks_problem_file
    ):
        reader = pddl_module.PDDLReader(blocks_domain_file, blocks_problem_file)
        assert len(reader.domains) == 1
        assert len(reader.problems) == 1

    def test_reader_constructor_empty(self, pddl_module):
        reader = pddl_module.PDDLReader()
        assert len(reader.domains) == 0
        assert len(reader.problems) == 0

    def test_reader_incremental_loading(
        self, pddl_module, blocks_domain_file, blocks_problem_file
    ):
        reader = pddl_module.PDDLReader(blocks_domain_file)
        assert len(reader.domains) == 1
        assert len(reader.problems) == 0
        reader.load(blocks_problem_file)
        assert len(reader.problems) == 1

    def test_reader_domain_access(self, pddl_module, blocks_domain_file):
        reader = pddl_module.PDDLReader(blocks_domain_file)
        domain = reader.domains[0]
        assert domain.get_name().lower() == "blocks"
        assert len(domain.get_actions()) == 4

    def test_reader_problem_access(
        self, pddl_module, blocks_domain_file, blocks_problem_file
    ):
        reader = pddl_module.PDDLReader(blocks_domain_file, blocks_problem_file)
        problem = reader.problems[0]
        assert problem.get_name().lower() == "blocks-3-0"
        assert problem.get_domain().get_name().lower() == "blocks"


class TestPDDLTireworld:
    """Tests for tireworld domain (uses :probabilistic-effects)."""

    @pytest.fixture
    def tireworld_parsed(
        self, pddl_module, tireworld_domain_file, tireworld_problem_file
    ):
        p = pddl_module.PDDL()
        p.load([tireworld_domain_file, tireworld_problem_file])
        return p

    def test_tireworld_domain_name(self, tireworld_parsed):
        domain = tireworld_parsed.get_domains()[0]
        assert domain.get_name().lower() == "tire"

    def test_tireworld_requirements(self, tireworld_parsed):
        domain = tireworld_parsed.get_domains()[0]
        reqs = domain.get_requirements()
        assert reqs is not None
        assert reqs.has_typing()
        assert reqs.has_strips()
        assert reqs.has_equality()
        assert reqs.has_probabilistic_effects()

    def test_tireworld_predicates(self, tireworld_parsed):
        domain = tireworld_parsed.get_domains()[0]
        predicates = domain.get_predicates()
        pred_names = {p.get_name().lower() for p in predicates}
        assert pred_names == {
            "vehicle-at",
            "spare-in",
            "road",
            "not-flattire",
            "hasspare",
        }

    def test_tireworld_move_car_action(self, tireworld_parsed):
        domain = tireworld_parsed.get_domains()[0]
        action = domain.get_action("move-car")
        assert action is not None

    def test_tireworld_all_actions(self, tireworld_parsed):
        domain = tireworld_parsed.get_domains()[0]
        actions = domain.get_actions()
        assert len(actions) == 3

    def test_tireworld_problem_objects(self, tireworld_parsed):
        problem = tireworld_parsed.get_problems()[0]
        objects = problem.get_objects()
        obj_names = {o.get_name().lower() for o in objects}
        assert len(obj_names) == 17
        assert "n0" in obj_names
        assert "n16" in obj_names

    def test_tireworld_problem_goal(self, tireworld_parsed, pddl_module):
        problem = tireworld_parsed.get_problems()[0]
        goal = problem.get_goal()
        assert goal is not None
        assert isinstance(goal, pddl_module.PredicateFormula)
        assert goal.get_name().lower() == "vehicle-at"


class TestPDDLProbabilisticEffects:
    """Tests for probabilistic effect structure in tireworld domain."""

    @pytest.fixture
    def tireworld_parsed(
        self, pddl_module, tireworld_domain_file, tireworld_problem_file
    ):
        p = pddl_module.PDDL()
        p.load([tireworld_domain_file, tireworld_problem_file])
        return p

    def test_move_car_probabilistic_effect(self, tireworld_parsed, pddl_module):
        domain = tireworld_parsed.get_domains()[0]
        action = domain.get_action("move-car")
        effect = action.get_effect()
        assert isinstance(effect, pddl_module.ConjunctionEffect)
        effects = effect.get_effects()
        prob_effects = [
            e for e in effects if isinstance(e, pddl_module.ProbabilisticEffect)
        ]
        assert len(prob_effects) == 1
        outcomes = prob_effects[0].get_outcomes()
        assert len(outcomes) == 1
        assert abs(outcomes[0][0] - 0.4) < 1e-9

    def test_changetire_probabilistic_effect(self, tireworld_parsed, pddl_module):
        domain = tireworld_parsed.get_domains()[0]
        action = domain.get_action("changetire")
        effect = action.get_effect()
        assert isinstance(effect, pddl_module.ProbabilisticEffect)
        outcomes = effect.get_outcomes()
        assert len(outcomes) == 1
        assert abs(outcomes[0][0] - 0.5) < 1e-9
        inner = outcomes[0][1]
        assert isinstance(inner, pddl_module.ConjunctionEffect)
        inner_effects = inner.get_effects()
        assert len(inner_effects) == 2
