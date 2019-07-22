from __airlaps import _PDDL_ as PDDL
from __airlaps import _PDDL_Domain_ as Domain
from __airlaps import _PDDL_Requirements_ as Requirements
from __airlaps import _PDDL_Type_ as Type
from __airlaps import _PDDL_Term_ as Term
from __airlaps import _PDDL_Variable_ as Variable
from __airlaps import _PDDL_Object_ as Object
from __airlaps import _PDDL_Predicate_ as Predicate
from __airlaps import _PDDL_Function_ as Function
from __airlaps import _PDDL_Formula_ as Formula
from __airlaps import _PDDL_Preference_ as Preference
from __airlaps import _PDDL_PredicateFormula_ as PredicateFormula
from __airlaps import _PDDL_UniversalFormula_ as UniversalFormula
from __airlaps import _PDDL_ExistentialFormula_ as ExistentialFormula
from __airlaps import _PDDL_ConjunctionFormula_ as ConjunctionFormula
from __airlaps import _PDDL_DisjunctionFormula_ as DisjunctionFormula
from __airlaps import _PDDL_Implyformula_ as ImplyFormula
from __airlaps import _PDDL_NegationFormula_ as NegationFormula
from __airlaps import _PDDL_AtStartFormula_ as AtStartFormula
from __airlaps import _PDDL_AtEndFormula_ as AtEndFormula
from __airlaps import _PDDL_OverAllFormula_ as OverAllFormula
from __airlaps import _PDDL_GreaterFormula_ as GreaterFormula
from __airlaps import _PDDL_GreaterEqFormula_ as GreaterEqFormula
from __airlaps import _PDDL_LessFormula_ as LessFormula
from __airlaps import _PDDL_LessEqFormula_ as LessEqFormula
from __airlaps import _PDDL_AddExpression_ as AddExpression
from __airlaps import _PDDL_SubExpression_ as SubExpression
from __airlaps import _PDDL_MulExpression_ as MulExpression
from __airlaps import _PDDL_DivExpression_ as DivExpression
from __airlaps import _PDDL_MinusExpression_ as MinusExpression
from __airlaps import _PDDL_NumericalExpression_ as NumericalExpression
from __airlaps import _PDDL_FunctionExpression_ as FunctionExpression

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
