generate_template_instantiation_files(
        AGGREGATION_EFFECT_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/aggregation_effect.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "ConjunctionEffect!Con;DisjunctionEffect!Dis")

generate_template_instantiation_files(
        AGGREGATION_FORMULA_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/aggregation_formula.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "ConjunctionFormula!Con;DisjunctionFormula!Dis")

generate_template_instantiation_files(
        ASSIGNMENT_EFFECT_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/assignment_effect.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "AssignEffect!Asg;ScaleUpEffect!ScU;ScaleDownEffect!ScD;IncreaseEffect!Inc;DecreaseEffect!Dec")

generate_template_instantiation_files(
        ASSOCIATIVE_CONTAINER_DOMAIN_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/associative_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Domain!Dom"
        "Tsymbol" "Class!Cls;Predicate!Prd;DerivedPredicate!Dpr;Function!Fun;Object!Obj;Preference!Prf;Type!Typ;Action!Act;DurativeAction!Dac;Event!Evt;Process!Prc")

generate_template_instantiation_files(
        ASSOCIATIVE_CONTAINER_PROBLEM_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/associative_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Problem!Pbm"
        "Tsymbol" "Object!Obj;Preference!Prf")

generate_template_instantiation_files(
        ASSOCIATIVE_CONTAINER_TYPE_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/associative_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Type!Tpe;Object!Obj;Variable!Var"
        "Tsymbol" "Type!Tpe")

generate_template_instantiation_files(
        ASSOCIATIVE_CONTAINER_CLASS_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/associative_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Class!Cls"
        "Tsymbol" "Function!Fun")

generate_template_instantiation_files(
        BINARY_EXPRESSION_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/binary_expression.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "AddExpression!Ade;SubExpression!Sue;MulExpression!Mue;DivExpression!Die;GreaterFormula!Grf;GreaterEqFormula!Gef;LessFormula!Lsf;LessEqFormula!Lef;EqFormula!Eqf")

generate_template_instantiation_files(
        BINARY_FORMULA_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/binary_formula.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "SometimeAfterFormula!Saf;SometimeBeforeFormula!Sbf;AlwaysWithinFormula!Awf;ImplyFormula!Imf")

generate_template_instantiation_files(
        CLASS_CONTAINER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/class_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Domain!Dom")

generate_template_instantiation_files(
        COMPARISON_FORMULA_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/comparison_formula.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "GreaterFormula!GFm;GreaterEqFormula!Gef;LessEqFormula!Lef;LessFormula!Lfm;EqFormula!Efm")

generate_template_instantiation_files(
        DERIVED_PREDICATE_CONTAINER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/derived_predicate_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Domain!Dom")

generate_template_instantiation_files(
        FUNCTION_CONTAINER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/function_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Domain!Dom;Class!Cls")

generate_template_instantiation_files(
        NUMBER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/number.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tnumber" "long!long;long long!llong;int!int;unsigned int!uint;long int!lint;long unsigned int!luint;double!dbl;long double!ldbl;float!flt")

generate_template_instantiation_files(
        OBJECT_CONTAINER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/object_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Domain!Dom;Problem!Pbm")

generate_template_instantiation_files(
        OPERATOR_CONTAINER_BASE_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/operator_container_base.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Domain!Dom"
        "Toperator" "Action!Act;DurativeAction!Dac;Event!Evt;Process!Prc")

generate_template_instantiation_files(
        OPERATOR_CONTAINER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/operator_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Domain!Dom")

generate_template_instantiation_files(
        OPERATOR_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/operator.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Action!Act;DurativeAction!Dac;Event!Evt;Process!Prc")

generate_template_instantiation_files(
        PREDICATE_CONTAINER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/predicate_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Domain!Dom")

generate_template_instantiation_files(
        QUANTIFIED_EFFECT_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/quantified_effect.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "UniversalEffect!Uef;ExistentialEffect!Eef")

generate_template_instantiation_files(
        QUANTIFIED_FORMULA_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/quantified_formula.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "UniversalFormula!Ufo;ExistentialFormula!Efo")

generate_template_instantiation_files(
        TERM_CONTAINER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/term_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "EqualityFormula!Eqf;FunctionExpression!Fex;PredicateEffect!Pef;PredicateFormula!Pfm")

generate_template_instantiation_files(
        VARIABLE_CONTAINER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/variable_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Function!Fun;Action!Act;DurativeAction!Dac;Event!Evt;Process!Prc;Predicate!Prd;UniversalEffect!Uef;ExistentialEffect!Eef;UniversalFormula!Ufo;ExistentialFormula!Efo")

generate_template_instantiation_files(
        TYPE_CONTAINER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/type_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Domain!Dom;Type!Tpe;Object!Obj;Variable!Var")

generate_template_instantiation_files(
        PREFERENCE_CONTAINER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/preference_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Domain!Dom;Problem!Pbm")

generate_template_instantiation_files(
        UNARY_EFFECT_NEGATION_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/unary_effect.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "NegationEffect!Neg"
        "Teff" "PredicateEffect!Pef")

generate_template_instantiation_files(
        UNARY_EFFECT_TIMED_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/unary_effect.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "AtStartEffect!Ast;AtEndEffect!Aen;AtTimeEffect!Atm"
        "Teff" "Effect!Eff")

generate_template_instantiation_files(
        UNARY_FORMULA_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/unary_formula.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "NegationFormula!Neg;AtStartFormula!Ast;AtEndFormula!Aen;OverAllFormula!Oal;AlwaysFormula!Alw;SometimeFormula!Som;AtMostOnceFormula!Amo;WithinFormula!Wit;HoldAfterFormula!Haf;HoldDuringFormula!Hdu")

generate_template_instantiation_files(
        UNARY_EXPRESSION_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/unary_expression.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "MinusExpression!Min;MinimizeExpression!Mni;MaximizeExpression!Mxa")

generate_template_instantiation_files(
        SEQUENCE_CONTAINER_TERM_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/sequence_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "EqualityFormula!Eqf;FunctionExpression!Fex;PredicateEffect!Pef;PredicateFormula!Pfm"
        "Tsymbol" "Term!Trm")

generate_template_instantiation_files(
        SEQUENCE_CONTAINER_VARIABLE_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/sequence_container.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Tderived" "Function!Fun;Action!Act;DurativeAction!Dac;Event!Evt;Process!Prc;Predicate!Prd;UniversalEffect!Uef;ExistentialEffect!Eef;UniversalFormula!Ufo;ExistentialFormula!Efo"
        "Tsymbol" "Variable!Var")

generate_template_instantiation_files(
        PARSER_PASS_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/parser/impl/parser_pass.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "TgrammarTag" "DomainStructureTag!DStr;DomainOperatorsTag!DOps;ProblemTag!Prb"
        "TcontrolTag" "NormalControlTag!Nrm;TracerControlTag!Trc")

ADD_LIBRARY(skdecide_pddl STATIC
        type.cc
        ${TERM_CONTAINER_TEMPLATE_FILES}
        ${VARIABLE_CONTAINER_TEMPLATE_FILES}
        ${TYPE_CONTAINER_TEMPLATE_FILES}
        ${PREFERENCE_CONTAINER_TEMPLATE_FILES}
        ${UNARY_EFFECT_NEGATION_TEMPLATE_FILES}
        ${UNARY_EFFECT_TIMED_TEMPLATE_FILES}
        ${UNARY_FORMULA_TEMPLATE_FILES}
        ${UNARY_EXPRESSION_TEMPLATE_FILES}
        ${AGGREGATION_EFFECT_TEMPLATE_FILES}
        impl/aggregation_effect.cc
        ${AGGREGATION_FORMULA_TEMPLATE_FILES}
        impl/aggregation_formula.cc
        ${ASSIGNMENT_EFFECT_TEMPLATE_FILES}
        impl/assignment_effect.cc
        ${ASSOCIATIVE_CONTAINER_DOMAIN_TEMPLATE_FILES}
        ${ASSOCIATIVE_CONTAINER_PROBLEM_TEMPLATE_FILES}
        ${ASSOCIATIVE_CONTAINER_TYPE_TEMPLATE_FILES}
        ${ASSOCIATIVE_CONTAINER_CLASS_TEMPLATE_FILES}
        impl/binary_effect.cc
        ${BINARY_EXPRESSION_TEMPLATE_FILES}
        ${BINARY_FORMULA_TEMPLATE_FILES}
        ${CLASS_CONTAINER_TEMPLATE_FILES}
        impl/class.cc
        ${COMPARISON_FORMULA_TEMPLATE_FILES}
        impl/comparison_formula.cc
        impl/conditional_effect.cc
        impl/probabilistic_effect.cc
        impl/constraint_formula.cc
        ${DERIVED_PREDICATE_CONTAINER_TEMPLATE_FILES}
        impl/derived_predicate.cc
        impl/domain.cc
        impl/duration_effect.cc
        impl/duration_expression.cc
        impl/duration_formula.cc
        impl/effect.cc
        impl/equality_formula.cc
        impl/expression.cc
        impl/formula.cc
        ${FUNCTION_CONTAINER_TEMPLATE_FILES}
        impl/function_expression.cc
        impl/function.cc
        impl/identifier.cc
        impl/imply_formula.cc
        impl/minus_expression.cc
        impl/negation_effect.cc
        impl/negation_formula.cc
        ${NUMBER_TEMPLATE_FILES}
        impl/number.cc
        impl/numerical_expression.cc
        ${OBJECT_CONTAINER_TEMPLATE_FILES}
        impl/object.cc
        impl/operation_expression.cc
        ${OPERATOR_CONTAINER_BASE_TEMPLATE_FILES}
        ${OPERATOR_CONTAINER_TEMPLATE_FILES}
        ${OPERATOR_TEMPLATE_FILES}
        impl/operator.cc
        impl/optimization_expression.cc
        impl/pddl.cc
        ${PREDICATE_CONTAINER_TEMPLATE_FILES}
        impl/preference.cc
        impl/problem.cc
        ${QUANTIFIED_EFFECT_TEMPLATE_FILES}
        impl/quantified_effect.cc
        ${QUANTIFIED_FORMULA_TEMPLATE_FILES}
        impl/quantified_formula.cc
        impl/requirements.cc
        ${SEQUENCE_CONTAINER_TERM_TEMPLATE_FILES}
        ${SEQUENCE_CONTAINER_VARIABLE_TEMPLATE_FILES}
        parser/parser.cc
        ${PARSER_PASS_TEMPLATE_FILES}
        impl/predicate_effect.cc
        impl/predicate_formula.cc
        impl/predicate.cc
        impl/variable.cc
        impl/timed_effect.cc
        impl/timed_formula.cc
        impl/term.cc
        impl/effect_semantics.cc
        impl/formula_semantics.cc
        impl/expression_semantics.cc
        semantics/task.cc
        semantics/applicable_actions_generator.cc
        semantics/successor_generator.cc
        semantics/goal_checker.cc
        semantics/determinize_effects.cc
        semantics/temporal_simulator.cc
        heuristics/delete_relaxation.cc
        heuristics/ff_heuristic.cc)

TARGET_INCLUDE_DIRECTORIES(skdecide_pddl PRIVATE
        ${INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR})
TARGET_LINK_LIBRARIES(skdecide_pddl PUBLIC libclingo spdlog::spdlog)

# Optional Z3 for analytical event time detection
find_package(Z3 QUIET)
if(Z3_FOUND)
    message(STATUS "Z3 found: ${Z3_VERSION} — enabling analytical event time detection")
    target_compile_definitions(skdecide_pddl PRIVATE SKDECIDE_HAS_Z3)
    target_include_directories(skdecide_pddl PRIVATE ${Z3_CXX_INCLUDE_DIRS})
    target_link_libraries(skdecide_pddl PRIVATE z3::libz3)
else()
    message(STATUS "Z3 not found — event time detection will use binary search only")
endif()

IF(NOT ONLY_PYTHON)
        CMAKE_POLICY(SET CMP0079 NEW)
        TARGET_LINK_LIBRARIES(skdecide PRIVATE skdecide_pddl)
ENDIF()

IF(BUILD_PYTHON_BINDING OR ONLY_PYTHON)
        ADD_LIBRARY(py_pddl OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/py_pddl.cc)
        TARGET_INCLUDE_DIRECTORIES(py_pddl PRIVATE
                ${INCLUDE_DIRS}
                ${CMAKE_CURRENT_SOURCE_DIR})
        CMAKE_POLICY(SET CMP0079 NEW)
        TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE
                skdecide_pddl
                $<TARGET_OBJECTS:py_pddl>)
ENDIF()
