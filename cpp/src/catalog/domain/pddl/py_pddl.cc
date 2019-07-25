#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <sstream>

#include "pddl.hh"

namespace py = pybind11;
using namespace airlaps::pddl;

template <typename Instance>
void inherit_identifier(Instance& instance) {
    using Identifier = typename Instance::type;
    instance.def("get_name", &Identifier::get_name,
                 py::return_value_policy::reference_internal);
}

template <typename Instance>
void inherit_type_container(Instance& instance) {
    using TypeContainer = typename Instance::type;
    instance.def("add_type", (const Domain::TypePtr& (TypeContainer::*)(const std::string&)) &TypeContainer::add_type,
                              py::arg("type"), py::return_value_policy::reference_internal)
            .def("add_type", (const Domain::TypePtr& (TypeContainer::*)(const Domain::TypePtr&)) &TypeContainer::add_type,
                              py::arg("type"), py::return_value_policy::reference_internal)
            .def("remove_type", (void (TypeContainer::*)(const std::string&)) &TypeContainer::remove_type,
                              py::arg("type"))
            .def("remove_type", (void (TypeContainer::*)(const Domain::TypePtr&)) &TypeContainer::remove_type,
                              py::arg("type"))
            .def("get_type", (const Domain::TypePtr& (TypeContainer::*)(const std::string&) const) &TypeContainer::get_type,
                              py::arg("type"), py::return_value_policy::reference_internal)
            .def("get_type", (const Domain::TypePtr& (TypeContainer::*)(const Domain::TypePtr&) const) &TypeContainer::get_type,
                              py::arg("type"), py::return_value_policy::reference_internal)
            .def("get_types", &TypeContainer::get_types,
                              py::return_value_policy::reference_internal)
            .def("__str__", (std::string (TypeContainer::*)() const) &TypeContainer::print);
}

template <typename Instance>
void inherit_object_container(Instance& instance) {
    using ObjectContainer = typename Instance::type;
    instance.def("add_object", (const Domain::ObjectPtr& (ObjectContainer::*)(const std::string&)) &ObjectContainer::add_object,
                                     py::arg("object"), py::return_value_policy::reference_internal)
            .def("add_object", (const Domain::ObjectPtr& (ObjectContainer::*)(const Domain::ObjectPtr&)) &ObjectContainer::add_object,
                                     py::arg("object"), py::return_value_policy::reference_internal)
            .def("remove_object", (void (ObjectContainer::*)(const std::string&)) &ObjectContainer::remove_object,
                                     py::arg("object"))
            .def("remove_object", (void (ObjectContainer::*)(const Domain::ObjectPtr&)) &ObjectContainer::remove_object,
                                     py::arg("object"))
            .def("get_object", (const Domain::ObjectPtr& (ObjectContainer::*)(const std::string&) const) &ObjectContainer::get_object,
                                  py::arg("object"), py::return_value_policy::reference_internal)
            .def("get_object", (const Domain::ObjectPtr& (ObjectContainer::*)(const Domain::ObjectPtr&) const) &ObjectContainer::get_object,
                                  py::arg("object"), py::return_value_policy::reference_internal)
            .def("get_objects", &ObjectContainer::get_objects,
                                  py::return_value_policy::reference_internal)
            .def("__str__", (std::string (ObjectContainer::*)() const) &ObjectContainer::print);
}

template <typename Instance>
void inherit_variable_container(Instance& instance) {
    using VariableContainer = typename Instance::type;
    instance.def("append_variable", (const Predicate::VariablePtr& (VariableContainer::*)(const std::string&)) &VariableContainer::append_variable,
                                     py::arg("variable"), py::return_value_policy::reference_internal)
            .def("append_variable", (const Predicate::VariablePtr& (VariableContainer::*)(const Predicate::VariablePtr&)) &VariableContainer::append_variable,
                                     py::arg("variable"), py::return_value_policy::reference_internal)
            .def("remove_variable", (void (VariableContainer::*)(const std::string&)) &VariableContainer::remove_variable,
                                     py::arg("variable"))
            .def("remove_variable", (void (VariableContainer::*)(const Predicate::VariablePtr&)) &VariableContainer::remove_variable,
                                     py::arg("variable"))
            .def("get_variable", (Predicate::VariableVector (VariableContainer::*)(const std::string&) const) &VariableContainer::get_variable,
                                  py::arg("variable"), py::return_value_policy::reference_internal)
            .def("get_variable", (Predicate::VariableVector (VariableContainer::*)(const Predicate::VariablePtr&) const) &VariableContainer::get_variable,
                                  py::arg("variable"), py::return_value_policy::reference_internal)
            .def("variable_at", (const Predicate::VariablePtr& (VariableContainer::*)(const std::size_t&) const) &VariableContainer::variable_at,
                                 py::arg("index"), py::return_value_policy::reference_internal)
            .def("get_variables", &VariableContainer::get_variables,
                                  py::return_value_policy::reference_internal)
            .def("__str__", (std::string (VariableContainer::*)() const) &VariableContainer::print);
}

template <typename Instance>
void inherit_term_container(Instance& instance) {
    using TermContainer = typename Instance::type;
    instance.def("append_term", (const PredicateFormula::TermPtr& (TermContainer::*)(const PredicateFormula::TermPtr&)) &TermContainer::append_term,
                                     py::arg("term"), py::return_value_policy::reference_internal)
            .def("remove_term", (void (TermContainer::*)(const std::string&)) &TermContainer::remove_term,
                                     py::arg("term"))
            .def("remove_term", (void (TermContainer::*)(const PredicateFormula::TermPtr&)) &TermContainer::remove_term,
                                     py::arg("term"))
            .def("get_term", (PredicateFormula::TermVector (TermContainer::*)(const std::string&) const) &TermContainer::get_term,
                                  py::arg("term"), py::return_value_policy::reference_internal)
            .def("get_term", (PredicateFormula::TermVector (TermContainer::*)(const PredicateFormula::TermPtr&) const) &TermContainer::get_term,
                                  py::arg("term"), py::return_value_policy::reference_internal)
            .def("term_at", (const PredicateFormula::TermPtr& (TermContainer::*)(const std::size_t&) const) &TermContainer::term_at,
                                 py::arg("index"), py::return_value_policy::reference_internal)
            .def("get_terms", &TermContainer::get_terms,
                                  py::return_value_policy::reference_internal)
            .def("__str__", (std::string (TermContainer::*)() const) &TermContainer::print);
}

template <typename Instance>
void inherit_predicate_container(Instance& instance) {
    using PredicateContainer = typename Instance::type;
    instance.def("add_predicate", (const Domain::PredicatePtr& (PredicateContainer::*)(const std::string&)) &PredicateContainer::add_predicate,
                                     py::arg("predicate"), py::return_value_policy::reference_internal)
            .def("add_predicate", (const Domain::PredicatePtr& (PredicateContainer::*)(const Domain::PredicatePtr&)) &PredicateContainer::add_predicate,
                                     py::arg("predicate"), py::return_value_policy::reference_internal)
            .def("remove_predicate", (void (PredicateContainer::*)(const std::string&)) &PredicateContainer::remove_predicate,
                                     py::arg("predicate"))
            .def("remove_predicate", (void (PredicateContainer::*)(const Domain::PredicatePtr&)) &PredicateContainer::remove_predicate,
                                     py::arg("predicate"))
            .def("get_predicate", (const Domain::PredicatePtr& (PredicateContainer::*)(const std::string&) const) &PredicateContainer::get_predicate,
                                  py::arg("predicate"), py::return_value_policy::reference_internal)
            .def("get_predicate", (const Domain::PredicatePtr& (PredicateContainer::*)(const Domain::PredicatePtr&) const) &PredicateContainer::get_predicate,
                                  py::arg("predicate"), py::return_value_policy::reference_internal)
            .def("get_predicates", &PredicateContainer::get_predicates, py::return_value_policy::reference_internal)
            .def("__str__", (std::string (PredicateContainer::*)() const) &PredicateContainer::print);
}

template <typename Instance>
void inherit_derived_predicate_container(Instance& instance) {
    using DerivedPredicateContainer = typename Instance::type;
    instance.def("add_derived_predicate", (const Domain::DerivedPredicatePtr& (DerivedPredicateContainer::*)(const std::string&)) &DerivedPredicateContainer::add_derived_predicate,
                                           py::arg("derived_predicate"), py::return_value_policy::reference_internal)
            .def("add_derived_predicate", (const Domain::DerivedPredicatePtr& (DerivedPredicateContainer::*)(const Domain::DerivedPredicatePtr&)) &DerivedPredicateContainer::add_derived_predicate,
                                           py::arg("derived_predicate"), py::return_value_policy::reference_internal)
            .def("remove_derived_predicate", (void (DerivedPredicateContainer::*)(const std::string&)) &DerivedPredicateContainer::remove_derived_predicate,
                                              py::arg("derived_predicate"))
            .def("remove_derived_predicate", (void (DerivedPredicateContainer::*)(const Domain::DerivedPredicatePtr&)) &DerivedPredicateContainer::remove_derived_predicate,
                                              py::arg("derived_predicate"))
            .def("get_derived_predicate", (const Domain::DerivedPredicatePtr& (DerivedPredicateContainer::*)(const std::string&) const) &DerivedPredicateContainer::get_derived_predicate,
                                           py::arg("derived_predicate"), py::return_value_policy::reference_internal)
            .def("get_derived_predicate", (const Domain::DerivedPredicatePtr& (DerivedPredicateContainer::*)(const Domain::DerivedPredicatePtr&) const) &DerivedPredicateContainer::get_derived_predicate,
                                           py::arg("derived_predicate"), py::return_value_policy::reference_internal)
            .def("get_derived_predicates", &DerivedPredicateContainer::get_derived_predicates, py::return_value_policy::reference_internal)
            .def("__str__", (std::string (DerivedPredicateContainer::*)() const) &DerivedPredicateContainer::print);
}

template <typename Instance>
void inherit_function_container(Instance& instance) {
    using FunctionContainer = typename Instance::type;
    instance.def("add_function", (const Domain::FunctionPtr& (FunctionContainer::*)(const std::string&)) &FunctionContainer::add_function,
                                     py::arg("function"), py::return_value_policy::reference_internal)
            .def("add_function", (const Domain::FunctionPtr& (FunctionContainer::*)(const Domain::FunctionPtr&)) &FunctionContainer::add_function,
                                     py::arg("function"), py::return_value_policy::reference_internal)
            .def("remove_function", (void (FunctionContainer::*)(const std::string&)) &FunctionContainer::remove_function,
                                     py::arg("function"))
            .def("remove_function", (void (FunctionContainer::*)(const Domain::FunctionPtr&)) &FunctionContainer::remove_function,
                                     py::arg("function"))
            .def("get_function", (const Domain::FunctionPtr& (FunctionContainer::*)(const std::string&) const) &FunctionContainer::get_function,
                                  py::arg("function"), py::return_value_policy::reference_internal)
            .def("get_function", (const Domain::FunctionPtr& (FunctionContainer::*)(const Domain::FunctionPtr&) const) &FunctionContainer::get_function,
                                  py::arg("function"), py::return_value_policy::reference_internal)
            .def("get_functions", &FunctionContainer::get_functions, py::return_value_policy::reference_internal)
            .def("__str__", (std::string (FunctionContainer::*)() const) &FunctionContainer::print);
}

template <typename Instance>
void inherit_class_container(Instance& instance) {
    using ClassContainer = typename Instance::type;
    instance.def("add_class", (const Domain::ClassPtr& (ClassContainer::*)(const std::string&)) &ClassContainer::add_class,
                                     py::arg("class"), py::return_value_policy::reference_internal)
            .def("add_class", (const Domain::ClassPtr& (ClassContainer::*)(const Domain::ClassPtr&)) &ClassContainer::add_class,
                                     py::arg("class"), py::return_value_policy::reference_internal)
            .def("remove_class", (void (ClassContainer::*)(const std::string&)) &ClassContainer::remove_class,
                                     py::arg("class"))
            .def("remove_class", (void (ClassContainer::*)(const Domain::ClassPtr&)) &ClassContainer::remove_class,
                                     py::arg("class"))
            .def("get_class", (const Domain::ClassPtr& (ClassContainer::*)(const std::string&) const) &ClassContainer::get_class,
                                  py::arg("class"), py::return_value_policy::reference_internal)
            .def("get_class", (const Domain::ClassPtr& (ClassContainer::*)(const Domain::ClassPtr&) const) &ClassContainer::get_class,
                                  py::arg("class"), py::return_value_policy::reference_internal)
            .def("get_classes", &ClassContainer::get_classes, py::return_value_policy::reference_internal)
            .def("__str__", (std::string (ClassContainer::*)() const) &ClassContainer::print);
}

template <typename Instance>
void inherit_unary_formula(Instance& instance) {
    using UnaryFormula = typename Instance::type;
    instance.def("set_formula", (void (UnaryFormula::*)(const Formula::Ptr&)) &UnaryFormula::set_formula, py::arg("formula"))
            .def("get_formula", (const Formula::Ptr& (UnaryFormula::*)()) &UnaryFormula::get_formula, py::return_value_policy::reference_internal)
            .def("__str__", (std::string (UnaryFormula::*)() const) &UnaryFormula::print);
}

template <typename Instance>
void inherit_binary_formula(Instance& instance) {
    using BinaryFormula = typename Instance::type;
    instance.def("set_left_formula", (void (BinaryFormula::*)(const Formula::Ptr&)) &BinaryFormula::set_left_formula, py::arg("formula"))
            .def("get_left_formula", (const Formula::Ptr& (BinaryFormula::*)()) &BinaryFormula::get_left_formula, py::return_value_policy::reference_internal)
            .def("set_right_formula", (void (BinaryFormula::*)(const Formula::Ptr&)) &BinaryFormula::set_right_formula, py::arg("formula"))
            .def("get_right_formula", (const Formula::Ptr& (BinaryFormula::*)()) &BinaryFormula::get_right_formula, py::return_value_policy::reference_internal)
            .def("__str__", (std::string (BinaryFormula::*)() const) &BinaryFormula::print);
}

template <typename Instance>
void inherit_unary_expression(Instance& instance) {
    using UnaryExpression = typename Instance::type;
    instance.def("set_expression", (void (UnaryExpression::*)(const Formula::Ptr&)) &UnaryExpression::set_expression, py::arg("expression"))
            .def("get_expression", (const Formula::Ptr& (UnaryExpression::*)()) &UnaryExpression::get_expression, py::return_value_policy::reference_internal)
            .def("__str__", (std::string (UnaryExpression::*)() const) &UnaryExpression::print);
}

template <typename Instance>
void inherit_binary_expression(Instance& instance) {
    using BinaryExpression = typename Instance::type;
    instance.def("set_left_expression", (void (BinaryExpression::*)(const Expression::Ptr&)) &BinaryExpression::set_left_expression, py::arg("expression"))
            .def("get_left_expression", (const Expression::Ptr& (BinaryExpression::*)()) &BinaryExpression::get_left_expression, py::return_value_policy::reference_internal)
            .def("set_right_expression", (void (BinaryExpression::*)(const Expression::Ptr&)) &BinaryExpression::set_right_expression, py::arg("expression"))
            .def("get_right_expression", (const Expression::Ptr& (BinaryExpression::*)()) &BinaryExpression::get_right_expression, py::return_value_policy::reference_internal)
            .def("__str__", (std::string (BinaryExpression::*)() const) &BinaryExpression::print);
}

template <typename Instance>
void inherit_unary_effect(Instance& instance) {
    using UnaryEffect = typename Instance::type;
    instance.def("set_effect", (void (UnaryEffect::*)(const Effect::Ptr&)) &UnaryEffect::set_effect, py::arg("effect"))
            .def("get_effect", (const Effect::Ptr& (UnaryEffect::*)()) &UnaryEffect::get_effect, py::return_value_policy::reference_internal)
            .def("__str__", (std::string (UnaryEffect::*)() const) &UnaryEffect::print);
}

template <typename Instance>
void inherit_binary_effect(Instance& instance) {
    using BinaryEffect = typename Instance::type;
    instance.def("set_condition", (void (BinaryEffect::*)(const Formula::Ptr&)) &BinaryEffect::set_condition, py::arg("condition"))
            .def("get_condition", (const Formula::Ptr& (BinaryEffect::*)()) &BinaryEffect::get_condition, py::return_value_policy::reference_internal)
            .def("set_effect", (void (BinaryEffect::*)(const Effect::Ptr&)) &BinaryEffect::set_effect, py::arg("effect"))
            .def("get_effect", (const Effect::Ptr& (BinaryEffect::*)()) &BinaryEffect::get_effect, py::return_value_policy::reference_internal)
            .def("__str__", (std::string (BinaryEffect::*)() const) &BinaryEffect::print);
}

template <typename Instance>
void inherit_assignment_effect(Instance& instance) {
    using AssignmentEffect = typename Instance::type;
    instance.def("set_function", (void (AssignmentEffect::*)(const FunctionEffect::Ptr&)) &AssignmentEffect::set_function, py::arg("function"))
            .def("get_function", (const FunctionEffect::Ptr& (AssignmentEffect::*)()) &AssignmentEffect::get_function, py::return_value_policy::reference_internal)
            .def("set_expression", (void (AssignmentEffect::*)(const Expression::Ptr&)) &AssignmentEffect::set_expression, py::arg("expression"))
            .def("get_expression", (const Expression::Ptr& (AssignmentEffect::*)()) &AssignmentEffect::get_expression, py::return_value_policy::reference_internal)
            .def("__str__", (std::string (AssignmentEffect::*)() const) &AssignmentEffect::print);
}

void init_pypddl(py::module& m) {
    py::class_<PDDL> py_pddl(m, "_PDDL_");
        py_pddl
            .def(py::init<>())
            .def("load", &airlaps::pddl::PDDL::load,
                py::arg("domain"),
                py::arg("problem")=std::string(""),
                py::arg("debug_logs")=false,
                py::call_guard<py::scoped_ostream_redirect,
                               py::scoped_estream_redirect>())
            .def("get_domain", &airlaps::pddl::PDDL::get_domain,
                               py::return_value_policy::reference_internal)
        ;
    
    py::class_<Domain> py_domain(m, "_PDDL_Domain_");
    inherit_identifier(py_domain);
    inherit_type_container(py_domain);
    inherit_object_container(py_domain);
    inherit_predicate_container(py_domain);
    inherit_derived_predicate_container(py_domain);
    inherit_function_container(py_domain);
    inherit_class_container(py_domain);
        py_domain
            .def(py::init<>())
            .def("set_name", &Domain::set_name, py::arg("name"))
            .def("set_requirements", &Domain::set_requirements, py::arg("requirements"))
            .def("get_requirements", &Domain::get_requirements,
                                     py::return_value_policy::reference_internal)
            .def("__str__", &Domain::print)
        ;
    
    py::class_<Requirements> py_requirements(m, "_PDDL_Requirements_");
        py_requirements
            .def(py::init<>())
            .def("set_equality", &Requirements::set_equality)
            .def("has_equality", &Requirements::has_equality)
            .def("set_strips", &Requirements::set_strips)
            .def("has_strips", &Requirements::has_strips)
            .def("set_typing", &Requirements::set_typing)
            .def("has_typing", &Requirements::has_typing)
            .def("set_negative_preconditions", &Requirements::set_negative_preconditions)
            .def("has_negative_preconditions", &Requirements::has_negative_preconditions)
            .def("set_disjunctive_preconditions", &Requirements::set_disjunctive_preconditions)
            .def("has_disjunctive_preconditions", &Requirements::has_disjunctive_preconditions)
            .def("set_existential_preconditions", &Requirements::set_existential_preconditions)
            .def("has_existential_preconditions", &Requirements::has_existential_preconditions)
            .def("set_universal_preconditions", &Requirements::set_universal_preconditions)
            .def("has_universal_preconditions", &Requirements::has_universal_preconditions)
            .def("set_conditional_effects", &Requirements::set_conditional_effects)
            .def("has_conditional_effects", &Requirements::has_conditional_effects)
            .def("set_fluents", &Requirements::set_fluents)
            .def("has_fluents", &Requirements::has_fluents)
            .def("set_durative_actions", &Requirements::set_durative_actions)
            .def("has_durative_actions", &Requirements::has_durative_actions)
            .def("set_time", &Requirements::set_time)
            .def("has_time", &Requirements::has_time)
            .def("set_action_costs", &Requirements::set_action_costs)
            .def("has_action_costs", &Requirements::has_action_costs)
            .def("set_object_fluents", &Requirements::set_object_fluents)
            .def("has_object_fluents", &Requirements::has_object_fluents)
            .def("set_numeric_fluents", &Requirements::set_numeric_fluents)
            .def("has_numeric_fluents", &Requirements::has_numeric_fluents)
            .def("set_modules", &Requirements::set_modules)
            .def("has_modules", &Requirements::has_modules)
            .def("set_adl", &Requirements::set_adl)
            .def("has_adl", &Requirements::has_adl)
            .def("set_quantified_preconditions", &Requirements::set_quantified_preconditions)
            .def("has_quantified_preconditions", &Requirements::has_quantified_preconditions)
            .def("set_duration_inequalities", &Requirements::set_duration_inequalities)
            .def("has_duration_inequalities", &Requirements::has_duration_inequalities)
            .def("set_continuous_effects", &Requirements::set_continuous_effects)
            .def("has_continuous_effects", &Requirements::has_continuous_effects)
            .def("set_derived_predicates", &Requirements::set_derived_predicates)
            .def("has_derived_predicates", &Requirements::has_derived_predicates)
            .def("set_timed_initial_literals", &Requirements::set_timed_initial_literals)
            .def("has_timed_initial_literals", &Requirements::has_timed_initial_literals)
            .def("set_preferences", &Requirements::set_preferences)
            .def("has_preferences", &Requirements::has_preferences)
            .def("set_constraints", &Requirements::set_constraints)
            .def("has_constraints", &Requirements::has_constraints)
            .def("__str__", &Requirements::print)
        ;
    
    py::class_<Type, Type::Ptr> py_type(m, "_PDDL_Type_");
    inherit_identifier(py_type);
    inherit_type_container(py_type);
        py_type
            .def(py::init<std::string>(), py::arg("name"))
        ;
    
    // All term containers contain the same type of term smart pointers
    // so just get it from proposition formulas
    py::class_<Term, PredicateFormula::TermPtr> py_term(m, "_PDDL_Term_");
    
    py::class_<Object, Domain::ObjectPtr> py_object(m, "_PDDL_Object_", py_term);
    inherit_identifier(py_object);
    inherit_type_container(py_object);
        py_object
            .def(py::init<std::string>(), py::arg("name"))
        ;
    
    // All variable containers contain the same type of variable smart pointers
    // so just get it from predicates
    py::class_<Variable, Predicate::VariablePtr> py_variable(m, "_PDDL_Variable_", py_term);
    inherit_identifier(py_variable);
    inherit_type_container(py_variable);
        py_variable
            .def(py::init<std::string>(), py::arg("name"))
        ;
    
    py::class_<Predicate, Domain::PredicatePtr> py_predicate(m, "_PDDL_Predicate_");
    inherit_identifier(py_predicate);
    inherit_variable_container(py_predicate);
        py_predicate
            .def(py::init<std::string>(), py::arg("name"))
        ;
    
    py::class_<Function, Domain::FunctionPtr> py_function(m, "_PDDL_Function_");
    inherit_identifier(py_function);
    inherit_variable_container(py_function);
        py_function
            .def(py::init<std::string>(), py::arg("name"))
        ;
    
    py::class_<DerivedPredicate, DerivedPredicate::Ptr> py_derived_predicate(m, "_PDDL_DerivedPredicate_");
        py_derived_predicate
            .def(py::init<>())
            .def("set_predicate", &DerivedPredicate::set_predicate, py::arg("predicate"))
            .def("get_predicate", &DerivedPredicate::get_predicate, py::return_value_policy::reference_internal)
            .def("set_formula", &DerivedPredicate::set_formula, py::arg("formula"))
            .def("get_formula", &DerivedPredicate::get_formula, py::return_value_policy::reference_internal)
        ;
    
    py::class_<Class, Class::Ptr> py_class(m, "_PDDL_Class_");
    inherit_identifier(py_class);
    inherit_function_container(py_class);
        py_class
            .def(py::init<std::string>(), py::arg("name"))
        ;
    
    py::class_<Formula, Formula::Ptr> py_formula(m, "_PDDL_Formula_");

    py::enum_<ConstraintFormula::Sort>(m, "_PDDL_ConstraintFormulaSort_", py::arithmetic())
        .value("ATEND", ConstraintFormula::Sort::E_ATEND)
        .value("ALWAYS", ConstraintFormula::Sort::E_ALWAYS)
        .value("SOMETIME", ConstraintFormula::Sort::E_SOMETIME)
        .value("WITHIN", ConstraintFormula::Sort::E_WITHIN)
        .value("ATMOSTONCE", ConstraintFormula::Sort::E_ATMOSTONCE)
        .value("SOMETIMEAFTER", ConstraintFormula::Sort::E_SOMETIMEAFTER)
        .value("SOMETIMEBEFORE", ConstraintFormula::Sort::E_SOMETIMEBEFORE)
        .value("ALWAYSWITHIN", ConstraintFormula::Sort::E_ALWAYSWITHIN)
        .value("HOLDDURING", ConstraintFormula::Sort::E_HOLDDURING)
        .value("HOLDAFTER", ConstraintFormula::Sort::E_HOLDAFTER)
        .export_values();
    
    py::class_<ConstraintFormula, ConstraintFormula::Ptr> py_constraint_formula(m, "_PDDL_ConstraintFormula_", py_formula);
        py_constraint_formula
            .def(py::init<const ConstraintFormula::Sort&>(), py::arg("sort"))
            .def("get_sort", &ConstraintFormula::get_sort)
            .def("set_requirement", &ConstraintFormula::set_requirement, py::arg("requirement"))
            .def("get_requirement", &ConstraintFormula::get_requirement)
            .def("set_trigger", &ConstraintFormula::set_trigger, py::arg("trigger"))
            .def("get_trigger", &ConstraintFormula::get_trigger)
            .def("set_from", &ConstraintFormula::set_from, py::arg("from"))
            .def("get_from", &ConstraintFormula::get_from)
            .def("set_deadline", &ConstraintFormula::set_deadline, py::arg("deadline"))
            .def("get_deadline", &ConstraintFormula::get_deadline)
            .def("__str__", (std::string (ConstraintFormula::*)() const) &ConstraintFormula::print)
        ;
    
    py::class_<Preference, Preference::Ptr> py_preference(m, "_PDDL_Preference_", py_formula);
    inherit_identifier(py_preference);
        py_preference
            .def(py::init<>())
            .def("set_formula", &Preference::set_formula)
            .def("get_formula", &Preference::get_formula)
            .def("__str__", (std::string (Preference::*)() const) &Preference::print)
        ;
    
    py::class_<PredicateFormula, PredicateFormula::Ptr> py_predicate_formula(m, "_PDDL_PredicateFormula_", py_formula);
    inherit_term_container(py_predicate_formula);
        py_predicate_formula
            .def(py::init<>())
            .def("set_predicate", &PredicateFormula::set_predicate, py::arg("predicate"))
            .def("get_predicate", &PredicateFormula::get_predicate)
            .def("get_name", &PredicateFormula::get_name)
            .def("__str__", (std::string (PredicateFormula::*)() const) &PredicateFormula::print)
        ;
    
    py::class_<UniversalFormula, UniversalFormula::Ptr> py_universal_formula(m, "_PDDL_UniversalFormula_", py_formula);
        py_universal_formula
            .def(py::init<>())
            .def("set_formula", &UniversalFormula::set_formula, py::arg("formula"))
            .def("get_formula", &UniversalFormula::get_formula)
            .def("__str__", (std::string (UniversalFormula::*)() const) &UniversalFormula::print)
        ;
    
    py::class_<ExistentialFormula, ExistentialFormula::Ptr> py_existential_formula(m, "_PDDL_ExistentialFormula_", py_formula);
        py_existential_formula
            .def(py::init<>())
            .def("set_formula", &ExistentialFormula::set_formula, py::arg("formula"))
            .def("get_formula", &ExistentialFormula::get_formula)
            .def("__str__", (std::string (ExistentialFormula::*)() const) &ExistentialFormula::print)
        ;
    
    py::class_<ConjunctionFormula, ConjunctionFormula::Ptr> py_conjunction_formula(m, "_PDDL_ConjunctionFormula_", py_formula);
        py_conjunction_formula
            .def(py::init<>())
            .def("append_formula", &ConjunctionFormula::append_formula, py::arg("formula"))
            .def("remove_formula", &ConjunctionFormula::remove_formula)
            .def("formula_at", &ConjunctionFormula::formula_at, py::arg("index"))
            .def("get_formulas", &ConjunctionFormula::get_formulas)
            .def("__str__", (std::string (ConjunctionFormula::*)() const) &ConjunctionFormula::print)
        ;
    
    py::class_<DisjunctionFormula, DisjunctionFormula::Ptr> py_disjunction_formula(m, "_PDDL_DisjunctionFormula_", py_formula);
        py_disjunction_formula
            .def(py::init<>())
            .def("append_formula", &DisjunctionFormula::append_formula, py::arg("formula"))
            .def("remove_formula", &DisjunctionFormula::remove_formula)
            .def("formula_at", &DisjunctionFormula::formula_at, py::arg("index"))
            .def("get_formulas", &DisjunctionFormula::get_formulas)
            .def("__str__", (std::string (DisjunctionFormula::*)() const) &DisjunctionFormula::print)
        ;
    
    py::class_<ImplyFormula, ImplyFormula::Ptr> py_imply_formula(m, "_PDDL_ImplyFormula_", py_formula);
    inherit_binary_formula(py_imply_formula);
        py_imply_formula
            .def(py::init<>())
        ;
    
    py::class_<NegationFormula, NegationFormula::Ptr> py_negation_formula(m, "_PDDL_NegationFormula_", py_formula);
    inherit_unary_formula(py_negation_formula);
        py_negation_formula
            .def(py::init<>())
        ;
    
    py::class_<AtStartFormula, AtStartFormula::Ptr> py_atstart_formula(m, "_PDDL_AtStartFormula_", py_formula);
    inherit_unary_formula(py_atstart_formula);
        py_atstart_formula
            .def(py::init<>())
        ;
    
    py::class_<AtEndFormula, AtEndFormula::Ptr> py_atend_formula(m, "_PDDL_AtEndFormula_", py_formula);
    inherit_unary_formula(py_atend_formula);
        py_atend_formula
            .def(py::init<>())
        ;
    
    py::class_<OverAllFormula, OverAllFormula::Ptr> py_overall_formula(m, "_PDDL_OverAllFormula_", py_formula);
    inherit_unary_formula(py_overall_formula);
        py_overall_formula
            .def(py::init<>())
        ;
    
    py::class_<DurationFormula, DurationFormula::Ptr> py_duration_formula(m, "_PDDL_DurationFormula_", py_formula);
        py_duration_formula
            .def(py::init<>())
            .def("set_durative_action", &DurationFormula::set_durative_action, py::arg("durative_action"))
            .def("get_durative_action", &DurationFormula::get_durative_action, py::return_value_policy::reference_internal)
        ;
    
    py::class_<GreaterFormula, GreaterFormula::Ptr> py_greater_formula(m, "_PDDL_GreaterFormula_", py_formula);
    inherit_binary_expression(py_greater_formula);
        py_greater_formula
            .def(py::init<>())
        ;
    
    py::class_<GreaterEqFormula, GreaterEqFormula::Ptr> py_greatereq_formula(m, "_PDDL_GreaterEqFormula_", py_formula);
    inherit_binary_expression(py_greatereq_formula);
        py_greatereq_formula
            .def(py::init<>())
        ;
    
    py::class_<LessEqFormula, LessEqFormula::Ptr> py_lesseq_formula(m, "_PDDL_LessEqFormula_", py_formula);
    inherit_binary_expression(py_lesseq_formula);
        py_lesseq_formula
            .def(py::init<>())
        ;
    
    py::class_<LessFormula, LessFormula::Ptr> py_less_formula(m, "_PDDL_LessFormula_", py_formula);
    inherit_binary_expression(py_less_formula);
        py_less_formula
            .def(py::init<>())
        ;
    
    py::class_<Expression, Expression::Ptr> py_expression(m, "_PDDL_Expression_");

    py::class_<AddExpression, AddExpression::Ptr> py_add_expression(m, "_PDDL_AddExpression_", py_expression);
    inherit_binary_expression(py_add_expression);
        py_add_expression
            .def(py::init<>())
        ;
    
    py::class_<SubExpression, SubExpression::Ptr> py_sub_expression(m, "_PDDL_SubExpression_", py_expression);
    inherit_binary_expression(py_sub_expression);
        py_sub_expression
            .def(py::init<>())
        ;
    
    py::class_<MulExpression, MulExpression::Ptr> py_mul_expression(m, "_PDDL_MulExpression_", py_expression);
    inherit_binary_expression(py_mul_expression);
        py_mul_expression
            .def(py::init<>())
        ;
    
    py::class_<DivExpression, DivExpression::Ptr> py_div_expression(m, "_PDDL_DivExpression_", py_expression);
    inherit_binary_expression(py_div_expression);
        py_div_expression
            .def(py::init<>())
        ;
    
    py::class_<MinusExpression, MinusExpression::Ptr> py_minus_expression(m, "_PDDL_MinusExpression_", py_expression);
    inherit_unary_expression(py_minus_expression);
        py_minus_expression
            .def(py::init<>())
        ;
    
    py::class_<NumericalExpression, NumericalExpression::Ptr> py_numerical_expression(m, "_PDDL_NumericalExpression_", py_expression);
        py_numerical_expression
            .def(py::init<>())
            .def("set_number", &NumericalExpression::set_number, py::arg("number"))
            .def("get_number", &NumericalExpression::get_number)
            .def("__str__", (std::string (NumericalExpression::*)() const) &NumericalExpression::print)
        ;
    
    py::class_<FunctionExpression, FunctionExpression::Ptr> py_function_expression(m, "_PDDL_FunctionExpression_", py_expression);
    inherit_term_container(py_function_expression);
        py_function_expression
            .def(py::init<>())
            .def("set_function", &FunctionExpression::set_function, py::arg("function"))
            .def("get_function", &FunctionExpression::get_function)
            .def("get_name", &FunctionExpression::get_name)
            .def("__str__", (std::string (FunctionExpression::*)() const) &FunctionExpression::print)
        ;
    
    py::class_<Effect, Effect::Ptr> py_effect(m, "_PDDL_Effect_");

    py::class_<PredicateEffect, PredicateEffect::Ptr> py_predicate_effect(m, "_PDDL_PredicateEffect_", py_effect);
    inherit_term_container(py_predicate_effect);
        py_predicate_effect
            .def(py::init<>())
            .def("set_predicate", &PredicateEffect::set_predicate, py::arg("predicate"))
            .def("get_predicate", &PredicateEffect::get_predicate)
            .def("get_name", &PredicateEffect::get_name)
            .def("__str__", (std::string (PredicateEffect::*)() const) &PredicateEffect::print)
        ;
    
    py::class_<ConjunctionEffect, ConjunctionEffect::Ptr> py_conjunction_effect(m, "_PDDL_ConjunctionEffect_", py_effect);
        py_conjunction_effect
            .def(py::init<>())
            .def("append_effect", &ConjunctionEffect::append_effect, py::arg("effect"))
            .def("remove_effect", &ConjunctionEffect::remove_effect)
            .def("effect_at", &ConjunctionEffect::effect_at, py::arg("index"))
            .def("get_effects", &ConjunctionEffect::get_effects)
            .def("__str__", (std::string (ConjunctionEffect::*)() const) &ConjunctionEffect::print)
        ;
    
    py::class_<DisjunctionEffect, DisjunctionEffect::Ptr> py_disjunction_effect(m, "_PDDL_DisjunctionEffect_", py_effect);
        py_disjunction_effect
            .def(py::init<>())
            .def("append_effect", &DisjunctionEffect::append_effect, py::arg("effect"))
            .def("remove_effect", &DisjunctionEffect::remove_effect)
            .def("effect_at", &DisjunctionEffect::effect_at, py::arg("index"))
            .def("get_effects", &DisjunctionEffect::get_effects)
            .def("__str__", (std::string (DisjunctionEffect::*)() const) &DisjunctionEffect::print)
        ;
    
    py::class_<UniversalEffect, UniversalEffect::Ptr> py_universal_effect(m, "_PDDL_UniversalEffect_", py_effect);
        py_universal_effect
            .def(py::init<>())
            .def("set_effect", &UniversalEffect::set_effect, py::arg("effect"))
            .def("get_effect", &UniversalEffect::get_effect)
            .def("__str__", (std::string (UniversalEffect::*)() const) &UniversalEffect::print)
        ;
    
    py::class_<ExistentialEffect, ExistentialEffect::Ptr> py_existential_effect(m, "_PDDL_ExistentialEffect_", py_effect);
        py_existential_effect
            .def(py::init<>())
            .def("set_effect", &ExistentialEffect::set_effect, py::arg("effect"))
            .def("get_effect", &ExistentialEffect::get_effect)
            .def("__str__", (std::string (ExistentialEffect::*)() const) &ExistentialEffect::print)
        ;
    
    py::class_<ConditionalEffect, ConditionalEffect::Ptr> py_conditional_effect(m, "_PDDL_ConditionalEffect_", py_effect);
    inherit_binary_effect(py_conditional_effect);
        py_conditional_effect
            .def(py::init<>())
        ;
    
    py::class_<NegationEffect, NegationEffect::Ptr> py_negation_effect(m, "_PDDL_NegationEffect_", py_effect);
    inherit_unary_effect(py_negation_effect);
        py_negation_effect
            .def(py::init<>())
        ;
    
    py::class_<AtStartEffect, AtStartEffect::Ptr> py_atstart_effect(m, "_PDDL_AtStartEffect_", py_effect);
    inherit_unary_effect(py_atstart_effect);
        py_atstart_effect
            .def(py::init<>())
        ;
    
    py::class_<AtEndEffect, AtEndEffect::Ptr> py_atend_effect(m, "_PDDL_AtEndEffect_", py_effect);
    inherit_unary_effect(py_atend_effect);
        py_atend_effect
            .def(py::init<>())
        ;
    
    py::class_<DurationEffect, DurationEffect::Ptr> py_duration_effect(m, "_PDDL_DurationEffect_", py_effect);
        py_duration_effect
            .def(py::init<>())
            .def("set_durative_action", &DurationEffect::set_durative_action, py::arg("durative_action"))
            .def("get_durative_action", &DurationEffect::get_durative_action, py::return_value_policy::reference_internal)
        ;
    
    py::class_<FunctionEffect, FunctionEffect::Ptr> py_function_effect(m, "_PDDL_FunctionEffect_", py_effect);
    inherit_term_container(py_function_effect);
        py_function_effect
            .def(py::init<>())
            .def("set_function", &FunctionEffect::set_function, py::arg("function"))
            .def("get_function", &FunctionEffect::get_function)
            .def("get_name", &FunctionEffect::get_name)
            .def("__str__", (std::string (FunctionEffect::*)() const) &FunctionEffect::print)
        ;
    
    py::class_<AssignEffect, AssignEffect::Ptr> py_assign_effect(m, "_PDDL_AssignEffect_", py_effect);
    inherit_assignment_effect(py_assign_effect);
        py_assign_effect
            .def(py::init<>())
        ;
    
    py::class_<ScaleUpEffect, ScaleUpEffect::Ptr> py_scaleup_effect(m, "_PDDL_ScaleUpEffect_", py_effect);
    inherit_assignment_effect(py_scaleup_effect);
        py_scaleup_effect
            .def(py::init<>())
        ;
    
    py::class_<ScaleDownEffect, ScaleDownEffect::Ptr> py_scaledown_effect(m, "_PDDL_ScaleDownEffect_", py_effect);
    inherit_assignment_effect(py_scaledown_effect);
        py_scaledown_effect
            .def(py::init<>())
        ;
    
    py::class_<IncreaseEffect, IncreaseEffect::Ptr> py_increase_effect(m, "_PDDL_IncreaseEffect_", py_effect);
    inherit_assignment_effect(py_increase_effect);
        py_increase_effect
            .def(py::init<>())
        ;
    
    py::class_<DecreaseEffect, DecreaseEffect::Ptr> py_decrease_effect(m, "_PDDL_DecreaseEffect_", py_effect);
    inherit_assignment_effect(py_decrease_effect);
        py_decrease_effect
            .def(py::init<>())
        ;
    
    py::class_<Action, Action::Ptr> py_action(m, "_PDDL_Action_");
    inherit_variable_container(py_action);
    inherit_binary_effect(py_action);
        py_action
            .def(py::init<std::string>(), py::arg("name"))
        ;
    
    py::class_<DurativeAction, DurativeAction::Ptr> py_durative_action(m, "_PDDL_DurativeAction_", py_action);
        py_durative_action
            .def(py::init<std::string>(), py::arg("name"))
            .def("set_duration_constraint", &DurativeAction::set_duration_constraint)
            .def("get_duration_constraint", &DurativeAction::get_duration_constraint, py::return_value_policy::reference_internal)
        ;
    
    py::class_<Event, Event::Ptr> py_event(m, "_PDDL_Event_");
    inherit_variable_container(py_event);
    inherit_binary_effect(py_event);
        py_event
            .def(py::init<std::string>(), py::arg("name"))
        ;
    
    py::class_<Process, Process::Ptr> py_process(m, "_PDDL_Process_");
    inherit_variable_container(py_process);
    inherit_binary_effect(py_process);
        py_process
            .def(py::init<std::string>(), py::arg("name"))
        ;
}
