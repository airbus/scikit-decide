# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO: finish work in progress
from __future__ import annotations

__all__ = [
    "Action",
    "AddExpression",
    "AlwaysFormula",
    "AlwaysWithinFormula",
    "AssignEffect",
    "AtEndEffect",
    "AtEndFormula",
    "AtMostOnceFormula",
    "AtStartEffect",
    "AtStartFormula",
    "AtTimeEffect",
    "Class",
    "ConditionalEffect",
    "ConjunctionEffect",
    "ConjunctionFormula",
    "DecreaseEffect",
    "DerivedPredicate",
    "DisjunctionEffect",
    "DisjunctionFormula",
    "DivExpression",
    "Domain",
    "DurationEffect",
    "DurationExpression",
    "DurationFormula",
    "DurativeAction",
    "Effect",
    "EqFormula",
    "EqualityFormula",
    "Event",
    "ExistentialEffect",
    "ExistentialFormula",
    "Expression",
    "Formula",
    "Function",
    "FunctionExpression",
    "GoalAchievedExpression",
    "GreaterEqFormula",
    "GreaterFormula",
    "HoldAfterFormula",
    "HoldDuringFormula",
    "ImplyFormula",
    "IncreaseEffect",
    "LessEqFormula",
    "LessFormula",
    "MaximizeExpression",
    "MinimizeExpression",
    "MinusExpression",
    "MulExpression",
    "NegationEffect",
    "NegationFormula",
    "Number",
    "NumericalExpression",
    "Object",
    "OverAllFormula",
    "PDDL",
    "PDDLReader",
    "Predicate",
    "PredicateEffect",
    "PredicateFormula",
    "Preference",
    "ProbabilisticEffect",
    "Problem",
    "Process",
    "Requirements",
    "RewardExpression",
    "ScaleDownEffect",
    "ScaleUpEffect",
    "SometimeAfterFormula",
    "SometimeBeforeFormula",
    "SometimeFormula",
    "SubExpression",
    "Term",
    "TimeExpression",
    "TotalCostExpression",
    "TotalTimeExpression",
    "Type",
    "UniversalEffect",
    "UniversalFormula",
    "Variable",
    "ViolationExpression",
    "WithinFormula",
]

try:
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_ as PDDL
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Action_ as Action
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_AddExpression_ as AddExpression
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_AlwaysFormula_ as AlwaysFormula
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_AlwaysWithinFormula_ as AlwaysWithinFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_AssignEffect_ as AssignEffect
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_AtEndEffect_ as AtEndEffect
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_AtEndFormula_ as AtEndFormula
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_AtMostOnceFormula_ as AtMostOnceFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_AtStartEffect_ as AtStartEffect
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_AtStartFormula_ as AtStartFormula
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_AtTimeEffect_ as AtTimeEffect
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Class_ as Class
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_ConditionalEffect_ as ConditionalEffect,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_ConjunctionEffect_ as ConjunctionEffect,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_ConjunctionFormula_ as ConjunctionFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_DecreaseEffect_ as DecreaseEffect
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_DerivedPredicate_ as DerivedPredicate,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_DisjunctionEffect_ as DisjunctionEffect,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_DisjunctionFormula_ as DisjunctionFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_DivExpression_ as DivExpression
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Domain_ as Domain
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_DurationEffect_ as DurationEffect
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_DurationExpression_ as DurationExpression,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_DurationFormula_ as DurationFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_DurativeAction_ as DurativeAction
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Effect_ as Effect
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_EqFormula_ as EqFormula
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_EqualityFormula_ as EqualityFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Event_ as Event
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_ExistentialEffect_ as ExistentialEffect,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_ExistentialFormula_ as ExistentialFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Expression_ as Expression
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Formula_ as Formula
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Function_ as Function
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_FunctionExpression_ as FunctionExpression,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_GoalAchievedExpression_ as GoalAchievedExpression,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_GreaterEqFormula_ as GreaterEqFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_GreaterFormula_ as GreaterFormula
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_HoldAfterFormula_ as HoldAfterFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_HoldDuringFormula_ as HoldDuringFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_ImplyFormula_ as ImplyFormula
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_IncreaseEffect_ as IncreaseEffect
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_LessEqFormula_ as LessEqFormula
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_LessFormula_ as LessFormula
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_MaximizeExpression_ as MaximizeExpression,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_MinimizeExpression_ as MinimizeExpression,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_MinusExpression_ as MinusExpression,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_MulExpression_ as MulExpression
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_NegationEffect_ as NegationEffect
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_NegationFormula_ as NegationFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Number_ as Number
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_NumericalExpression_ as NumericalExpression,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Object_ as Object
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_OverAllFormula_ as OverAllFormula
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Predicate_ as Predicate
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_PredicateEffect_ as PredicateEffect,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_PredicateFormula_ as PredicateFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Preference_ as Preference
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_ProbabilisticEffect_ as ProbabilisticEffect,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Problem_ as Problem
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Process_ as Process
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Requirements_ as Requirements
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_RewardExpression_ as RewardExpression,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_ScaleDownEffect_ as ScaleDownEffect,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_ScaleUpEffect_ as ScaleUpEffect
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_SometimeAfterFormula_ as SometimeAfterFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_SometimeBeforeFormula_ as SometimeBeforeFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_SometimeFormula_ as SometimeFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_SubExpression_ as SubExpression
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Term_ as Term
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_TimeExpression_ as TimeExpression
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_TotalCostExpression_ as TotalCostExpression,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_TotalTimeExpression_ as TotalTimeExpression,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Type_ as Type
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_UniversalEffect_ as UniversalEffect,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_UniversalFormula_ as UniversalFormula,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Variable_ as Variable
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_ViolationExpression_ as ViolationExpression,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_WithinFormula_ as WithinFormula
except ImportError:
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise


class PDDLReader:
    """Convenience wrapper around the C++ PDDL parser.

    # Parameters
    files: One or more PDDL file paths (domain and/or problem files).
    verbose: Activates parsing traces.
    """

    def __init__(self, *files: str, verbose: bool = False):
        self._pddl = PDDL()
        if files:
            self._pddl.load(list(files), verbose)

    def load(self, *files: str, verbose: bool = False):
        """Parse additional PDDL files into this reader."""
        self._pddl.load(list(files), verbose)

    @property
    def domains(self):
        """Return the list of parsed domains."""
        return self._pddl.get_domains()

    @property
    def problems(self):
        """Return the list of parsed problems."""
        return self._pddl.get_problems()
