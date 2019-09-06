import sys, os
from airlaps import hub

record_sys_path = sys.path
airlaps_cpp_extension_lib_path = os.path.join(hub._get_airlaps_home(), 'lib')
if airlaps_cpp_extension_lib_path not in sys.path:
    sys.path.append(airlaps_cpp_extension_lib_path)

try:
    from __airlaps_hub_cpp import _PDDL_ as PDDL
    from __airlaps_hub_cpp import _PDDL_Domain_ as Domain
    from __airlaps_hub_cpp import _PDDL_Requirements_ as Requirements
    from __airlaps_hub_cpp import _PDDL_Type_ as Type
    from __airlaps_hub_cpp import _PDDL_Term_ as Term
    from __airlaps_hub_cpp import _PDDL_Variable_ as Variable
    from __airlaps_hub_cpp import _PDDL_Object_ as Object
    from __airlaps_hub_cpp import _PDDL_Predicate_ as Predicate
    from __airlaps_hub_cpp import _PDDL_Function_ as Function
    from __airlaps_hub_cpp import _PDDL_DerivedPredicate_ as DerivedPredicate
    from __airlaps_hub_cpp import _PDDL_Class_ as Class
    from __airlaps_hub_cpp import _PDDL_Formula_ as Formula
    from __airlaps_hub_cpp import _PDDL_Preference_ as Preference
    from __airlaps_hub_cpp import _PDDL_PredicateFormula_ as PredicateFormula
    from __airlaps_hub_cpp import _PDDL_UniversalFormula_ as UniversalFormula
    from __airlaps_hub_cpp import _PDDL_ExistentialFormula_ as ExistentialFormula
    from __airlaps_hub_cpp import _PDDL_ConjunctionFormula_ as ConjunctionFormula
    from __airlaps_hub_cpp import _PDDL_DisjunctionFormula_ as DisjunctionFormula
    from __airlaps_hub_cpp import _PDDL_ImplyFormula_ as ImplyFormula
    from __airlaps_hub_cpp import _PDDL_NegationFormula_ as NegationFormula
    from __airlaps_hub_cpp import _PDDL_AtStartFormula_ as AtStartFormula
    from __airlaps_hub_cpp import _PDDL_AtEndFormula_ as AtEndFormula
    from __airlaps_hub_cpp import _PDDL_OverAllFormula_ as OverAllFormula
    from __airlaps_hub_cpp import _PDDL_DurationFormula_ as DurationFormula
    from __airlaps_hub_cpp import _PDDL_GreaterFormula_ as GreaterFormula
    from __airlaps_hub_cpp import _PDDL_GreaterEqFormula_ as GreaterEqFormula
    from __airlaps_hub_cpp import _PDDL_LessFormula_ as LessFormula
    from __airlaps_hub_cpp import _PDDL_LessEqFormula_ as LessEqFormula
    from __airlaps_hub_cpp import _PDDL_Expression_ as Expression
    from __airlaps_hub_cpp import _PDDL_AddExpression_ as AddExpression
    from __airlaps_hub_cpp import _PDDL_SubExpression_ as SubExpression
    from __airlaps_hub_cpp import _PDDL_MulExpression_ as MulExpression
    from __airlaps_hub_cpp import _PDDL_DivExpression_ as DivExpression
    from __airlaps_hub_cpp import _PDDL_MinusExpression_ as MinusExpression
    from __airlaps_hub_cpp import _PDDL_NumericalExpression_ as NumericalExpression
    from __airlaps_hub_cpp import _PDDL_FunctionExpression_ as FunctionExpression
    from __airlaps_hub_cpp import _PDDL_Effect_ as Effect
    from __airlaps_hub_cpp import _PDDL_PredicateEffect_ as PredicateEffect
    from __airlaps_hub_cpp import _PDDL_ConjunctionEffect_ as ConjunctionEffect
    from __airlaps_hub_cpp import _PDDL_DisjunctionEffect_ as DisjunctionEffect
    from __airlaps_hub_cpp import _PDDL_UniversalEffect_ as UniversalEffect
    from __airlaps_hub_cpp import _PDDL_ExistentialEffect_ as ExistentialEffect
    from __airlaps_hub_cpp import _PDDL_ConditionalEffect_ as ConditionalEffect
    from __airlaps_hub_cpp import _PDDL_NegationEffect_ as NegationEffect
    from __airlaps_hub_cpp import _PDDL_AtStartEffect_ as AtStartEffect
    from __airlaps_hub_cpp import _PDDL_AtEndEffect_ as AtEndEffect
    from __airlaps_hub_cpp import _PDDL_DurationEffect_ as DurationEffect
    from __airlaps_hub_cpp import _PDDL_FunctionEffect_ as FunctionEffect
    from __airlaps_hub_cpp import _PDDL_AssignEffect_ as AssignEffect
    from __airlaps_hub_cpp import _PDDL_ScaleUpEffect_ as ScaleUpEffect
    from __airlaps_hub_cpp import _PDDL_ScaleDownEffect_ as ScaleDownEffect
    from __airlaps_hub_cpp import _PDDL_IncreaseEffect_ as IncreaseEffect
    from __airlaps_hub_cpp import _PDDL_DecreaseEffect_ as DecreaseEffect
    from __airlaps_hub_cpp import _PDDL_Action_ as Action
    from __airlaps_hub_cpp import _PDDL_DurativeAction_ as DurativeAction
    from __airlaps_hub_cpp import _PDDL_Event_ as Event
    from __airlaps_hub_cpp import _PDDL_Process_ as Process
except ImportError:
    sys.path = record_sys_path
    print('AIRLAPS C++ hub library not found. Please check it is installed in your AIRLAPS_HOME.')
    raise

# class PDDL:

#     def __init__(self, domain_file, problem_file='', debug_logs=False):
#         """Constructs a PDDL object (domain and problem) from PDDL files

#         # Parameters
#         domain_file: Domain description file, must also contain the problem definition if the second argument is the empty string
#         problem_file: Problem description file, can be empty in which case the problem must be described in the domain description file
#         debug_logs: Activates parsing traces

#         """
#         self._pddl = _PDDL_(domain_file, problem_file, debug_logs)
    
#     def get_domain(self):
#         return self._pddl.get_domain()
