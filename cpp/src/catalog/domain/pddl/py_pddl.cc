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
    instance.def("append_term", (const PropositionFormula::TermPtr& (TermContainer::*)(const PropositionFormula::TermPtr&)) &TermContainer::append_term,
                                     py::arg("term"), py::return_value_policy::reference_internal)
            .def("remove_term", (void (TermContainer::*)(const std::string&)) &TermContainer::remove_term,
                                     py::arg("term"))
            .def("remove_term", (void (TermContainer::*)(const PropositionFormula::TermPtr&)) &TermContainer::remove_term,
                                     py::arg("term"))
            .def("get_term", (PropositionFormula::TermVector (TermContainer::*)(const std::string&) const) &TermContainer::get_term,
                                  py::arg("term"), py::return_value_policy::reference_internal)
            .def("get_term", (PropositionFormula::TermVector (TermContainer::*)(const PropositionFormula::TermPtr&) const) &TermContainer::get_term,
                                  py::arg("term"), py::return_value_policy::reference_internal)
            .def("term_at", (const PropositionFormula::TermPtr& (TermContainer::*)(const std::size_t&) const) &TermContainer::term_at,
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
            .def("get_predicates", &PredicateContainer::get_predicates,
                                  py::return_value_policy::reference_internal)
            .def("__str__", (std::string (PredicateContainer::*)() const) &PredicateContainer::print);
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
            .def("get_functions", &FunctionContainer::get_functions,
                                  py::return_value_policy::reference_internal)
            .def("__str__", (std::string (FunctionContainer::*)() const) &FunctionContainer::print);
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
    inherit_function_container(py_domain);
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
    py::class_<Term, PropositionFormula::TermPtr> py_term(m, "_PDDL_Term_");
    
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
    
    py::class_<PropositionFormula, PropositionFormula::Ptr> py_proposition_formula(m, "_PDDL_PropositionFormula_", py_formula);
    inherit_identifier(py_proposition_formula);
    inherit_term_container(py_proposition_formula);
        py_proposition_formula
            .def(py::init<>())
            .def("__str__", (std::string (PropositionFormula::*)() const) &PropositionFormula::print)
        ;
    
    py::enum_<QuantifiedFormula::Quantifier>(m, "_PDDL_QuantifiedFormulaQuantifier_", py::arithmetic())
        .value("FORALL", QuantifiedFormula::Quantifier::E_FORALL)
        .value("EXISTS", QuantifiedFormula::Quantifier::E_EXISTS)
        .export_values();
    
    py::class_<QuantifiedFormula, QuantifiedFormula::Ptr> py_quantified_formula(m, "_PDDL_QuantifiedFormula_", py_formula);
        py_quantified_formula
            .def(py::init<const QuantifiedFormula::Quantifier&>(), py::arg("quantifier"))
            .def("get_quantifier", &QuantifiedFormula::get_quantifier)
            .def("set_formula", &QuantifiedFormula::set_formula, py::arg("formula"))
            .def("get_formula", &QuantifiedFormula::get_formula)
            .def("__str__", (std::string (QuantifiedFormula::*)() const) &QuantifiedFormula::print)
        ;
    
    py::enum_<AggregatedFormula::Operator>(m, "_PDDL_AggregatedFormulaOperator_", py::arithmetic())
        .value("AND", AggregatedFormula::Operator::E_AND)
        .value("OR", AggregatedFormula::Operator::E_OR)
        .export_values();
    
    py::class_<AggregatedFormula, AggregatedFormula::Ptr> py_aggregated_formula(m, "_PDDL_AggregatedFormula_", py_formula);
        py_aggregated_formula
            .def(py::init<const AggregatedFormula::Operator&>(), py::arg("operator"))
            .def("get_operator", &AggregatedFormula::get_operator)
            .def("append_formula", &AggregatedFormula::append_formula, py::arg("formula"))
            .def("remove_formula", &AggregatedFormula::remove_formula)
            .def("formula_at", &AggregatedFormula::formula_at, py::arg("formula"))
            .def("get_formulas", &AggregatedFormula::get_formulas)
            .def("__str__", (std::string (AggregatedFormula::*)() const) &AggregatedFormula::print)
        ;
}
