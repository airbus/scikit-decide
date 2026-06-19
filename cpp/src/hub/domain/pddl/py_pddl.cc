/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <sstream>

#include "pddl.hh"
#include "heuristics/delete_relaxation.hh"
#include "heuristics/ff_heuristic.hh"
#include "semantics/applicable_actions_generator.hh"
#include "semantics/goal_checker.hh"
#include "semantics/state.hh"
#include "semantics/successor_generator.hh"
#include "semantics/task.hh"
#include "semantics/temporal_simulator.hh"

namespace py = pybind11;
using namespace skdecide::pddl;

template <typename Instance> void inherit_identifier(Instance &instance) {
  using IdentifierType = typename Instance::type;
  instance.def("get_name", &IdentifierType::get_name,
               py::return_value_policy::reference_internal);
}

template <typename Instance> void inherit_type_container(Instance &instance) {
  using TypeContainerType = TypeContainer<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def(
          "add_type",
          (const Domain::TypePtr &(TypeContainerType::*)(const std::string &)) &
              TypeContainerType::add_type,
          py::arg("type"), py::return_value_policy::reference_internal)
      .def("add_type",
           (const Domain::TypePtr &(
               TypeContainerType::*)(const Domain::TypePtr &)) &
               TypeContainerType::add_type,
           py::arg("type"), py::return_value_policy::reference_internal)
      .def("remove_type",
           (void(TypeContainerType::*)(const std::string &)) &
               TypeContainerType::remove_type,
           py::arg("type"))
      .def("remove_type",
           (void(TypeContainerType::*)(const Domain::TypePtr &)) &
               TypeContainerType::remove_type,
           py::arg("type"))
      .def("get_type",
           (const Domain::TypePtr &(TypeContainerType::*)(const std::string &)
                const) &
               TypeContainerType::get_type,
           py::arg("type"), py::return_value_policy::reference_internal)
      .def("get_type",
           (const Domain::TypePtr &(
               TypeContainerType::*)(const Domain::TypePtr &) const) &
               TypeContainerType::get_type,
           py::arg("type"), py::return_value_policy::reference_internal)
      .def("get_types", &TypeContainerType::get_types,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance> void inherit_object_container(Instance &instance) {
  using ObjectContainerType = ObjectContainer<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("add_object",
           (const Domain::ObjectPtr &(
               ObjectContainerType::*)(const std::string &)) &
               ObjectContainerType::add_object,
           py::arg("object"), py::return_value_policy::reference_internal)
      .def("add_object",
           (const Domain::ObjectPtr &(
               ObjectContainerType::*)(const Domain::ObjectPtr &)) &
               ObjectContainerType::add_object,
           py::arg("object"), py::return_value_policy::reference_internal)
      .def("remove_object",
           (void(ObjectContainerType::*)(const std::string &)) &
               ObjectContainerType::remove_object,
           py::arg("object"))
      .def("remove_object",
           (void(ObjectContainerType::*)(const Domain::ObjectPtr &)) &
               ObjectContainerType::remove_object,
           py::arg("object"))
      .def("get_object",
           (const Domain::ObjectPtr &(
               ObjectContainerType::*)(const std::string &) const) &
               ObjectContainerType::get_object,
           py::arg("object"), py::return_value_policy::reference_internal)
      .def("get_object",
           (const Domain::ObjectPtr &(
               ObjectContainerType::*)(const Domain::ObjectPtr &) const) &
               ObjectContainerType::get_object,
           py::arg("object"), py::return_value_policy::reference_internal)
      .def("get_objects", &ObjectContainerType::get_objects,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance>
void inherit_variable_container(Instance &instance) {
  using VariableContainerType = VariableContainer<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("append_variable",
           (const Predicate::VariablePtr &(
               VariableContainerType::*)(const std::string &)) &
               VariableContainerType::append_variable,
           py::arg("variable"), py::return_value_policy::reference_internal)
      .def("append_variable",
           (const Predicate::VariablePtr &(
               VariableContainerType::*)(const Predicate::VariablePtr &)) &
               VariableContainerType::append_variable,
           py::arg("variable"), py::return_value_policy::reference_internal)
      .def("remove_variable",
           (void(VariableContainerType::*)(const std::string &)) &
               VariableContainerType::remove_variable,
           py::arg("variable"))
      .def("remove_variable",
           (void(VariableContainerType::*)(const Predicate::VariablePtr &)) &
               VariableContainerType::remove_variable,
           py::arg("variable"))
      .def("get_variable",
           (Predicate::VariableVector(VariableContainerType::*)(
               const std::string &) const) &
               VariableContainerType::get_variable,
           py::arg("variable"), py::return_value_policy::reference_internal)
      .def("get_variable",
           (Predicate::VariableVector(VariableContainerType::*)(
               const Predicate::VariablePtr &) const) &
               VariableContainerType::get_variable,
           py::arg("variable"), py::return_value_policy::reference_internal)
      .def("variable_at",
           (const Predicate::VariablePtr &(
               VariableContainerType::*)(const std::size_t &) const) &
               VariableContainerType::variable_at,
           py::arg("index"), py::return_value_policy::reference_internal)
      .def("get_variables", &VariableContainerType::get_variables,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance> void inherit_term_container(Instance &instance) {
  using TermContainerType = TermContainer<typename Instance::type>;
  instance
      .def("append_term",
           (const PredicateFormula::TermPtr &(
               TermContainerType::*)(const PredicateFormula::TermPtr &)) &
               TermContainerType::append_term,
           py::arg("term"), py::return_value_policy::reference_internal)
      .def("remove_term",
           (void(TermContainerType::*)(const std::string &)) &
               TermContainerType::remove_term,
           py::arg("term"))
      .def("remove_term",
           (void(TermContainerType::*)(const PredicateFormula::TermPtr &)) &
               TermContainerType::remove_term,
           py::arg("term"))
      .def("get_term",
           (PredicateFormula::TermVector(TermContainerType::*)(
               const std::string &) const) &
               TermContainerType::get_term,
           py::arg("term"), py::return_value_policy::reference_internal)
      .def("get_term",
           (PredicateFormula::TermVector(TermContainerType::*)(
               const PredicateFormula::TermPtr &) const) &
               TermContainerType::get_term,
           py::arg("term"), py::return_value_policy::reference_internal)
      .def("term_at",
           (const PredicateFormula::TermPtr &(
               TermContainerType::*)(const std::size_t &) const) &
               TermContainerType::term_at,
           py::arg("index"), py::return_value_policy::reference_internal)
      .def("get_terms", &TermContainerType::get_terms,
           py::return_value_policy::reference_internal)
      .def("__str__", (std::string(TermContainerType::*)() const) &
                          TermContainerType::print);
}

template <typename Instance>
void inherit_predicate_container(Instance &instance) {
  using PredicateContainerType = PredicateContainer<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("add_predicate",
           (const Domain::PredicatePtr &(
               PredicateContainerType::*)(const std::string &)) &
               PredicateContainerType::add_predicate,
           py::arg("predicate"), py::return_value_policy::reference_internal)
      .def("add_predicate",
           (const Domain::PredicatePtr &(
               PredicateContainerType::*)(const Domain::PredicatePtr &)) &
               PredicateContainerType::add_predicate,
           py::arg("predicate"), py::return_value_policy::reference_internal)
      .def("remove_predicate",
           (void(PredicateContainerType::*)(const std::string &)) &
               PredicateContainerType::remove_predicate,
           py::arg("predicate"))
      .def("remove_predicate",
           (void(PredicateContainerType::*)(const Domain::PredicatePtr &)) &
               PredicateContainerType::remove_predicate,
           py::arg("predicate"))
      .def("get_predicate",
           (const Domain::PredicatePtr &(
               PredicateContainerType::*)(const std::string &) const) &
               PredicateContainerType::get_predicate,
           py::arg("predicate"), py::return_value_policy::reference_internal)
      .def("get_predicate",
           (const Domain::PredicatePtr &(
               PredicateContainerType::*)(const Domain::PredicatePtr &) const) &
               PredicateContainerType::get_predicate,
           py::arg("predicate"), py::return_value_policy::reference_internal)
      .def("get_predicates", &PredicateContainerType::get_predicates,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance>
void inherit_derived_predicate_container(Instance &instance) {
  using DerivedPredicateContainerType =
      DerivedPredicateContainer<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("add_derived_predicate",
           (const Domain::DerivedPredicatePtr &(
               DerivedPredicateContainerType::*)(const std::string &)) &
               DerivedPredicateContainerType::add_derived_predicate,
           py::arg("derived_predicate"),
           py::return_value_policy::reference_internal)
      .def("add_derived_predicate",
           (const Domain::DerivedPredicatePtr &(
               DerivedPredicateContainerType::
                   *)(const Domain::DerivedPredicatePtr &)) &
               DerivedPredicateContainerType::add_derived_predicate,
           py::arg("derived_predicate"),
           py::return_value_policy::reference_internal)
      .def("remove_derived_predicate",
           (void(DerivedPredicateContainerType::*)(const std::string &)) &
               DerivedPredicateContainerType::remove_derived_predicate,
           py::arg("derived_predicate"))
      .def("remove_derived_predicate",
           (void(DerivedPredicateContainerType::*)(
               const Domain::DerivedPredicatePtr &)) &
               DerivedPredicateContainerType::remove_derived_predicate,
           py::arg("derived_predicate"))
      .def("get_derived_predicate",
           (const Domain::DerivedPredicatePtr &(
               DerivedPredicateContainerType::*)(const std::string &) const) &
               DerivedPredicateContainerType::get_derived_predicate,
           py::arg("derived_predicate"),
           py::return_value_policy::reference_internal)
      .def("get_derived_predicate",
           (const Domain::DerivedPredicatePtr &(
               DerivedPredicateContainerType::
                   *)(const Domain::DerivedPredicatePtr &) const) &
               DerivedPredicateContainerType::get_derived_predicate,
           py::arg("derived_predicate"),
           py::return_value_policy::reference_internal)
      .def("get_derived_predicates",
           &DerivedPredicateContainerType::get_derived_predicates,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance>
void inherit_function_container(Instance &instance) {
  using FunctionContainerType = FunctionContainer<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("add_function",
           (const Domain::FunctionPtr &(
               FunctionContainerType::*)(const std::string &)) &
               FunctionContainerType::add_function,
           py::arg("function"), py::return_value_policy::reference_internal)
      .def("add_function",
           (const Domain::FunctionPtr &(
               FunctionContainerType::*)(const Domain::FunctionPtr &)) &
               FunctionContainerType::add_function,
           py::arg("function"), py::return_value_policy::reference_internal)
      .def("remove_function",
           (void(FunctionContainerType::*)(const std::string &)) &
               FunctionContainerType::remove_function,
           py::arg("function"))
      .def("remove_function",
           (void(FunctionContainerType::*)(const Domain::FunctionPtr &)) &
               FunctionContainerType::remove_function,
           py::arg("function"))
      .def("get_function",
           (const Domain::FunctionPtr &(
               FunctionContainerType::*)(const std::string &) const) &
               FunctionContainerType::get_function,
           py::arg("function"), py::return_value_policy::reference_internal)
      .def("get_function",
           (const Domain::FunctionPtr &(
               FunctionContainerType::*)(const Domain::FunctionPtr &) const) &
               FunctionContainerType::get_function,
           py::arg("function"), py::return_value_policy::reference_internal)
      .def("get_functions", &FunctionContainerType::get_functions,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance> void inherit_class_container(Instance &instance) {
  using ClassContainerType = ClassContainer<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("add_class",
           (const Domain::ClassPtr &(
               ClassContainerType::*)(const std::string &)) &
               ClassContainerType::add_class,
           py::arg("class"), py::return_value_policy::reference_internal)
      .def("add_class",
           (const Domain::ClassPtr &(
               ClassContainerType::*)(const Domain::ClassPtr &)) &
               ClassContainerType::add_class,
           py::arg("class"), py::return_value_policy::reference_internal)
      .def("remove_class",
           (void(ClassContainerType::*)(const std::string &)) &
               ClassContainerType::remove_class,
           py::arg("class"))
      .def("remove_class",
           (void(ClassContainerType::*)(const Domain::ClassPtr &)) &
               ClassContainerType::remove_class,
           py::arg("class"))
      .def("get_class",
           (const Domain::ClassPtr &(ClassContainerType::*)(const std::string &)
                const) &
               ClassContainerType::get_class,
           py::arg("class"), py::return_value_policy::reference_internal)
      .def("get_class",
           (const Domain::ClassPtr &(
               ClassContainerType::*)(const Domain::ClassPtr &) const) &
               ClassContainerType::get_class,
           py::arg("class"), py::return_value_policy::reference_internal)
      .def("get_classes", &ClassContainerType::get_classes,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance>
void inherit_preference_container(Instance &instance) {
  using PreferenceContainerType = PreferenceContainer<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("add_preference",
           (const Domain::PreferencePtr &(
               PreferenceContainerType::*)(const std::string &)) &
               PreferenceContainerType::add_preference,
           py::arg("preference"), py::return_value_policy::reference_internal)
      .def("add_preference",
           (const Domain::PreferencePtr &(
               PreferenceContainerType::*)(const Domain::PreferencePtr &)) &
               PreferenceContainerType::add_preference,
           py::arg("preference"), py::return_value_policy::reference_internal)
      .def("remove_preference",
           (void(PreferenceContainerType::*)(const std::string &)) &
               PreferenceContainerType::remove_preference,
           py::arg("preference"))
      .def("remove_preference",
           (void(PreferenceContainerType::*)(const Domain::PreferencePtr &)) &
               PreferenceContainerType::remove_preference,
           py::arg("preference"))
      .def("get_preference",
           (const Domain::PreferencePtr &(
               PreferenceContainerType::*)(const std::string &) const) &
               PreferenceContainerType::get_preference,
           py::arg("preference"), py::return_value_policy::reference_internal)
      .def("get_preference",
           (const Domain::PreferencePtr &(
               PreferenceContainerType::*)(const Domain::PreferencePtr &)
                const) &
               PreferenceContainerType::get_preference,
           py::arg("preference"), py::return_value_policy::reference_internal)
      .def("get_preferences", &PreferenceContainerType::get_preferences,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance> void inherit_action_container(Instance &instance) {
  using ActionContainerType = ActionContainer<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("add_action",
           (const Domain::ActionPtr &(
               ActionContainerType::*)(const std::string &)) &
               ActionContainerType::add_action,
           py::arg("action"), py::return_value_policy::reference_internal)
      .def("add_action",
           (const Domain::ActionPtr &(
               ActionContainerType::*)(const Domain::ActionPtr &)) &
               ActionContainerType::add_action,
           py::arg("action"), py::return_value_policy::reference_internal)
      .def("remove_action",
           (void(ActionContainerType::*)(const std::string &)) &
               ActionContainerType::remove_action,
           py::arg("action"))
      .def("remove_action",
           (void(ActionContainerType::*)(const Domain::ActionPtr &)) &
               ActionContainerType::remove_action,
           py::arg("action"))
      .def("get_action",
           (const Domain::ActionPtr &(
               ActionContainerType::*)(const std::string &) const) &
               ActionContainerType::get_action,
           py::arg("action"), py::return_value_policy::reference_internal)
      .def("get_action",
           (const Domain::ActionPtr &(
               ActionContainerType::*)(const Domain::ActionPtr &) const) &
               ActionContainerType::get_action,
           py::arg("action"), py::return_value_policy::reference_internal)
      .def("get_actions", &ActionContainerType::get_actions,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance> void inherit_event_container(Instance &instance) {
  using EventContainerType = EventContainer<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("add_event",
           (const Domain::EventPtr &(
               EventContainerType::*)(const std::string &)) &
               EventContainerType::add_event,
           py::arg("event"), py::return_value_policy::reference_internal)
      .def("add_event",
           (const Domain::EventPtr &(
               EventContainerType::*)(const Domain::EventPtr &)) &
               EventContainerType::add_event,
           py::arg("event"), py::return_value_policy::reference_internal)
      .def("remove_event",
           (void(EventContainerType::*)(const std::string &)) &
               EventContainerType::remove_event,
           py::arg("event"))
      .def("remove_event",
           (void(EventContainerType::*)(const Domain::EventPtr &)) &
               EventContainerType::remove_event,
           py::arg("event"))
      .def("get_event",
           (const Domain::EventPtr &(EventContainerType::*)(const std::string &)
                const) &
               EventContainerType::get_event,
           py::arg("event"), py::return_value_policy::reference_internal)
      .def("get_event",
           (const Domain::EventPtr &(
               EventContainerType::*)(const Domain::EventPtr &) const) &
               EventContainerType::get_event,
           py::arg("event"), py::return_value_policy::reference_internal)
      .def("get_events", &EventContainerType::get_events,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance>
void inherit_process_container(Instance &instance) {
  using ProcessContainerType = ProcessContainer<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("add_process",
           (const Domain::ProcessPtr &(
               ProcessContainerType::*)(const std::string &)) &
               ProcessContainerType::add_process,
           py::arg("process"), py::return_value_policy::reference_internal)
      .def("add_process",
           (const Domain::ProcessPtr &(
               ProcessContainerType::*)(const Domain::ProcessPtr &)) &
               ProcessContainerType::add_process,
           py::arg("process"), py::return_value_policy::reference_internal)
      .def("remove_process",
           (void(ProcessContainerType::*)(const std::string &)) &
               ProcessContainerType::remove_process,
           py::arg("process"))
      .def("remove_process",
           (void(ProcessContainerType::*)(const Domain::ProcessPtr &)) &
               ProcessContainerType::remove_process,
           py::arg("process"))
      .def("get_process",
           (const Domain::ProcessPtr &(
               ProcessContainerType::*)(const std::string &) const) &
               ProcessContainerType::get_process,
           py::arg("process"), py::return_value_policy::reference_internal)
      .def("get_process",
           (const Domain::ProcessPtr &(
               ProcessContainerType::*)(const Domain::ProcessPtr &) const) &
               ProcessContainerType::get_process,
           py::arg("process"), py::return_value_policy::reference_internal)
      .def("get_processes", &ProcessContainerType::get_processes,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance>
void inherit_durative_action_container(Instance &instance) {
  using DurativeActionContainerType =
      DurativeActionContainer<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("add_durative_action",
           (const Domain::DurativeActionPtr &(
               DurativeActionContainerType::*)(const std::string &)) &
               DurativeActionContainerType::add_durative_action,
           py::arg("durative_action"),
           py::return_value_policy::reference_internal)
      .def("add_durative_action",
           (const Domain::DurativeActionPtr &(
               DurativeActionContainerType::*)(const Domain::DurativeActionPtr
                                                   &)) &
               DurativeActionContainerType::add_durative_action,
           py::arg("durative_action"),
           py::return_value_policy::reference_internal)
      .def("remove_durative_action",
           (void(DurativeActionContainerType::*)(const std::string &)) &
               DurativeActionContainerType::remove_durative_action,
           py::arg("durative_action"))
      .def("remove_durative_action",
           (void(DurativeActionContainerType::*)(
               const Domain::DurativeActionPtr &)) &
               DurativeActionContainerType::remove_durative_action,
           py::arg("durative_action"))
      .def("get_durative_action",
           (const Domain::DurativeActionPtr &(
               DurativeActionContainerType::*)(const std::string &) const) &
               DurativeActionContainerType::get_durative_action,
           py::arg("durative_action"),
           py::return_value_policy::reference_internal)
      .def(
          "get_durative_action",
          (const Domain::DurativeActionPtr &(
              DurativeActionContainerType::*)(const Domain::DurativeActionPtr &)
               const) &
              DurativeActionContainerType::get_durative_action,
          py::arg("durative_action"),
          py::return_value_policy::reference_internal)
      .def("get_durative_actions",
           &DurativeActionContainerType::get_durative_actions,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance> void inherit_unary_formula(Instance &instance) {
  using UnaryFormulaType = UnaryFormula<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("set_formula",
           (void(UnaryFormulaType::*)(const Formula::Ptr &)) &
               UnaryFormulaType::set_formula,
           py::arg("formula"))
      .def("get_formula",
           (const Formula::Ptr &(UnaryFormulaType::*)()) &
               UnaryFormulaType::get_formula,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance> void inherit_binary_formula(Instance &instance) {
  using BinaryFormulaType = BinaryFormula<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("set_left_formula",
           (void(BinaryFormulaType::*)(const Formula::Ptr &)) &
               BinaryFormulaType::set_left_formula,
           py::arg("formula"))
      .def("get_left_formula",
           (const Formula::Ptr &(BinaryFormulaType::*)()) &
               BinaryFormulaType::get_left_formula,
           py::return_value_policy::reference_internal)
      .def("set_right_formula",
           (void(BinaryFormulaType::*)(const Formula::Ptr &)) &
               BinaryFormulaType::set_right_formula,
           py::arg("formula"))
      .def("get_right_formula",
           (const Formula::Ptr &(BinaryFormulaType::*)()) &
               BinaryFormulaType::get_right_formula,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance> void inherit_unary_expression(Instance &instance) {
  using UnaryExpressionType = UnaryExpression<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("set_expression",
           (void(UnaryExpressionType::*)(const Formula::Ptr &)) &
               UnaryExpressionType::set_expression,
           py::arg("expression"))
      .def("get_expression",
           (const Formula::Ptr &(UnaryExpressionType::*)()) &
               UnaryExpressionType::get_expression,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance>
void inherit_binary_expression(Instance &instance) {
  using BinaryExpressionType = BinaryExpression<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("set_left_expression",
           (void(BinaryExpressionType::*)(const Expression::Ptr &)) &
               BinaryExpressionType::set_left_expression,
           py::arg("expression"))
      .def("get_left_expression",
           (const Expression::Ptr &(BinaryExpressionType::*)()) &
               BinaryExpressionType::get_left_expression,
           py::return_value_policy::reference_internal)
      .def("set_right_expression",
           (void(BinaryExpressionType::*)(const Expression::Ptr &)) &
               BinaryExpressionType::set_right_expression,
           py::arg("expression"))
      .def("get_right_expression",
           (const Expression::Ptr &(BinaryExpressionType::*)()) &
               BinaryExpressionType::get_right_expression,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance, typename Eff = Effect>
void inherit_unary_effect(Instance &instance) {
  using UnaryEffectType = UnaryEffect<typename Instance::type, Eff>;
  using InstanceType = typename Instance::type;
  instance
      .def("set_effect",
           (void(UnaryEffectType::*)(const typename Eff::Ptr &)) &
               UnaryEffectType::set_effect,
           py::arg("effect"))
      .def("get_effect",
           (const typename Eff::Ptr &(UnaryEffectType::*)()) &
               UnaryEffectType::get_effect,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance> void inherit_binary_effect(Instance &instance) {
  using BinaryEffectType = BinaryEffect;
  using InstanceType = typename Instance::type;
  instance
      .def("set_condition",
           (void(BinaryEffectType::*)(const Formula::Ptr &)) &
               BinaryEffectType::set_condition,
           py::arg("condition"))
      .def("get_condition",
           (const Formula::Ptr &(BinaryEffectType::*)()) &
               BinaryEffectType::get_condition,
           py::return_value_policy::reference_internal)
      .def("set_effect",
           (void(BinaryEffectType::*)(const Effect::Ptr &)) &
               BinaryEffectType::set_effect,
           py::arg("effect"))
      .def("get_effect",
           (const Effect::Ptr &(BinaryEffectType::*)()) &
               BinaryEffectType::get_effect,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

template <typename Instance>
void inherit_assignment_effect(Instance &instance) {
  using AssignmentEffectType = AssignmentEffect<typename Instance::type>;
  using InstanceType = typename Instance::type;
  instance
      .def("set_function",
           (void(AssignmentEffectType::*)(const FunctionExpression::Ptr &)) &
               AssignmentEffectType::set_function,
           py::arg("function"))
      .def("get_function",
           (const FunctionExpression::Ptr &(AssignmentEffectType::*)()) &
               AssignmentEffectType::get_function,
           py::return_value_policy::reference_internal)
      .def("set_expression",
           (void(AssignmentEffectType::*)(const Expression::Ptr &)) &
               AssignmentEffectType::set_expression,
           py::arg("expression"))
      .def("get_expression",
           (const Expression::Ptr &(AssignmentEffectType::*)()) &
               AssignmentEffectType::get_expression,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(InstanceType::*)() const) & InstanceType::print);
}

void init_pypddl(py::module &m) {
  py::class_<PDDL> py_pddl(m, "_PDDL_");
  py_pddl.def(py::init<>())
      .def("load", &skdecide::pddl::PDDL::load, py::arg("files"),
           py::arg("verbose") = false,
           py::call_guard<py::scoped_ostream_redirect,
                          py::scoped_estream_redirect>())
      .def("get_domains", &skdecide::pddl::PDDL::get_domains,
           py::return_value_policy::reference_internal)
      .def("get_problems", &skdecide::pddl::PDDL::get_problems,
           py::return_value_policy::reference_internal);

  py::class_<Domain, std::shared_ptr<Domain>> py_domain(m, "_PDDL_Domain_");
  inherit_identifier(py_domain);
  inherit_type_container(py_domain);
  inherit_object_container(py_domain);
  inherit_predicate_container(py_domain);
  inherit_derived_predicate_container(py_domain);
  inherit_function_container(py_domain);
  inherit_class_container(py_domain);
  inherit_preference_container(py_domain);
  inherit_action_container(py_domain);
  inherit_event_container(py_domain);
  inherit_process_container(py_domain);
  inherit_durative_action_container(py_domain);
  py_domain.def(py::init<std::string>(), py::arg("name"))
      .def("set_requirements", &Domain::set_requirements,
           py::arg("requirements"))
      .def("get_requirements", &Domain::get_requirements,
           py::return_value_policy::reference_internal)
      .def("set_constraints", &Domain::set_constraints, py::arg("constraints"))
      .def("get_constraints", &Domain::get_constraints,
           py::return_value_policy::reference_internal)
      .def("__str__", &Domain::print);

  py::class_<Requirements> py_requirements(m, "_PDDL_Requirements_");
  py_requirements.def(py::init<>())
      .def("set_equality", &Requirements::set_equality)
      .def("has_equality", &Requirements::has_equality)
      .def("set_strips", &Requirements::set_strips)
      .def("has_strips", &Requirements::has_strips)
      .def("set_typing", &Requirements::set_typing)
      .def("has_typing", &Requirements::has_typing)
      .def("set_negative_preconditions",
           &Requirements::set_negative_preconditions)
      .def("has_negative_preconditions",
           &Requirements::has_negative_preconditions)
      .def("set_disjunctive_preconditions",
           &Requirements::set_disjunctive_preconditions)
      .def("has_disjunctive_preconditions",
           &Requirements::has_disjunctive_preconditions)
      .def("set_existential_preconditions",
           &Requirements::set_existential_preconditions)
      .def("has_existential_preconditions",
           &Requirements::has_existential_preconditions)
      .def("set_universal_preconditions",
           &Requirements::set_universal_preconditions)
      .def("has_universal_preconditions",
           &Requirements::has_universal_preconditions)
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
      .def("set_quantified_preconditions",
           &Requirements::set_quantified_preconditions)
      .def("has_quantified_preconditions",
           &Requirements::has_quantified_preconditions)
      .def("set_duration_inequalities",
           &Requirements::set_duration_inequalities)
      .def("has_duration_inequalities",
           &Requirements::has_duration_inequalities)
      .def("set_continuous_effects", &Requirements::set_continuous_effects)
      .def("has_continuous_effects", &Requirements::has_continuous_effects)
      .def("set_derived_predicates", &Requirements::set_derived_predicates)
      .def("has_derived_predicates", &Requirements::has_derived_predicates)
      .def("set_timed_initial_literals",
           &Requirements::set_timed_initial_literals)
      .def("has_timed_initial_literals",
           &Requirements::has_timed_initial_literals)
      .def("set_preferences", &Requirements::set_preferences)
      .def("has_preferences", &Requirements::has_preferences)
      .def("set_constraints", &Requirements::set_constraints)
      .def("has_constraints", &Requirements::has_constraints)
      .def("set_probabilistic_effects",
           &Requirements::set_probabilistic_effects)
      .def("has_probabilistic_effects",
           &Requirements::has_probabilistic_effects)
      .def("set_rewards", &Requirements::set_rewards)
      .def("has_rewards", &Requirements::has_rewards)
      .def("set_mdp", &Requirements::set_mdp)
      .def("has_mdp", &Requirements::has_mdp)
      .def("__str__", &Requirements::print);

  py::class_<Type, Type::Ptr> py_type(m, "_PDDL_Type_");
  inherit_identifier(py_type);
  inherit_type_container(py_type);
  py_type.def(py::init<std::string>(), py::arg("name"));

  // All term containers contain the same type of term smart pointers
  // so just get it from proposition formulas
  py::class_<Term, PredicateFormula::TermPtr> py_term(m, "_PDDL_Term_");

  py::class_<Object, Domain::ObjectPtr> py_object(m, "_PDDL_Object_", py_term);
  inherit_identifier(py_object);
  inherit_type_container(py_object);
  py_object.def(py::init<std::string>(), py::arg("name"));

  // All variable containers contain the same type of variable smart pointers
  // so just get it from predicates
  py::class_<Variable, Predicate::VariablePtr> py_variable(m, "_PDDL_Variable_",
                                                           py_term);
  inherit_identifier(py_variable);
  inherit_type_container(py_variable);
  py_variable.def(py::init<std::string>(), py::arg("name"));

  py::class_<Predicate, Domain::PredicatePtr> py_predicate(m,
                                                           "_PDDL_Predicate_");
  inherit_identifier(py_predicate);
  inherit_variable_container(py_predicate);
  py_predicate.def(py::init<std::string>(), py::arg("name"));

  py::class_<Function, Domain::FunctionPtr> py_function(m, "_PDDL_Function_");
  inherit_identifier(py_function);
  inherit_variable_container(py_function);
  py_function.def(py::init<std::string>(), py::arg("name"));

  py::class_<DerivedPredicate, DerivedPredicate::Ptr> py_derived_predicate(
      m, "_PDDL_DerivedPredicate_", py_predicate);
  py_derived_predicate.def(py::init<std::string>(), py::arg("name"))
      .def("set_formula", &DerivedPredicate::set_formula, py::arg("formula"))
      .def("get_formula", &DerivedPredicate::get_formula,
           py::return_value_policy::reference_internal);

  py::class_<Class, Class::Ptr> py_class(m, "_PDDL_Class_");
  inherit_identifier(py_class);
  inherit_function_container(py_class);
  py_class.def(py::init<std::string>(), py::arg("name"));

  py::class_<Formula, Formula::Ptr> py_formula(m, "_PDDL_Formula_");

  py::class_<AlwaysFormula, AlwaysFormula::Ptr> py_always_formula(
      m, "_PDDL_AlwaysFormula_", py_formula);
  inherit_unary_formula(py_always_formula);
  py_always_formula.def(py::init<>());

  py::class_<SometimeFormula, SometimeFormula::Ptr> py_sometime_formula(
      m, "_PDDL_SometimeFormula_", py_formula);
  inherit_unary_formula(py_sometime_formula);
  py_sometime_formula.def(py::init<>());

  py::class_<AtMostOnceFormula, AtMostOnceFormula::Ptr> py_at_most_once_formula(
      m, "_PDDL_AtMostOnceFormula_", py_formula);
  inherit_unary_formula(py_at_most_once_formula);
  py_at_most_once_formula.def(py::init<>());

  py::class_<WithinFormula, WithinFormula::Ptr> py_within_formula(
      m, "_PDDL_WithinFormula_", py_formula);
  inherit_unary_formula(py_within_formula);
  py_within_formula.def(py::init<>())
      .def("set_deadline", &WithinFormula::set_deadline, py::arg("deadline"))
      .def("get_deadline", &WithinFormula::get_deadline,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(WithinFormula::*)() const) & WithinFormula::print);

  py::class_<HoldAfterFormula, HoldAfterFormula::Ptr> py_hold_after_formula(
      m, "_PDDL_HoldAfterFormula_", py_formula);
  inherit_unary_formula(py_hold_after_formula);
  py_hold_after_formula.def(py::init<>())
      .def("set_from", &HoldAfterFormula::set_from, py::arg("from"))
      .def("get_from", &HoldAfterFormula::get_from,
           py::return_value_policy::reference_internal)
      .def("__str__", (std::string(HoldAfterFormula::*)() const) &
                          HoldAfterFormula::print);

  py::class_<HoldDuringFormula, HoldDuringFormula::Ptr> py_hold_during_formula(
      m, "_PDDL_HoldDuringFormula_", py_formula);
  inherit_unary_formula(py_hold_during_formula);
  py_hold_during_formula.def(py::init<>())
      .def("set_from", &HoldDuringFormula::set_from, py::arg("from"))
      .def("get_from", &HoldDuringFormula::get_from,
           py::return_value_policy::reference_internal)
      .def("set_deadline", &HoldDuringFormula::set_deadline,
           py::arg("deadline"))
      .def("get_deadline", &HoldDuringFormula::get_deadline,
           py::return_value_policy::reference_internal)
      .def("__str__", (std::string(HoldDuringFormula::*)() const) &
                          HoldDuringFormula::print);

  py::class_<SometimeAfterFormula, SometimeAfterFormula::Ptr>
      py_sometime_after_formula(m, "_PDDL_SometimeAfterFormula_", py_formula);
  inherit_binary_formula(py_sometime_after_formula);
  py_sometime_after_formula.def(py::init<>());

  py::class_<SometimeBeforeFormula, SometimeBeforeFormula::Ptr>
      py_sometime_before_formula(m, "_PDDL_SometimeBeforeFormula_", py_formula);
  inherit_binary_formula(py_sometime_before_formula);
  py_sometime_before_formula.def(py::init<>());

  py::class_<AlwaysWithinFormula, AlwaysWithinFormula::Ptr>
      py_always_within_formula(m, "_PDDL_AlwaysWithinFormula_", py_formula);
  inherit_binary_formula(py_always_within_formula);
  py_always_within_formula.def(py::init<>())
      .def("set_deadline", &AlwaysWithinFormula::set_deadline,
           py::arg("deadline"))
      .def("get_deadline", &AlwaysWithinFormula::get_deadline,
           py::return_value_policy::reference_internal)
      .def("__str__", (std::string(AlwaysWithinFormula::*)() const) &
                          AlwaysWithinFormula::print);

  py::class_<Preference, Preference::Ptr> py_preference(m, "_PDDL_Preference_",
                                                        py_formula);
  inherit_identifier(py_preference);
  py_preference.def(py::init<>())
      .def("set_formula", &Preference::set_formula)
      .def("get_formula", &Preference::get_formula)
      .def("__str__", (std::string(Preference::*)() const) & Preference::print);

  py::class_<PredicateFormula, PredicateFormula::Ptr> py_predicate_formula(
      m, "_PDDL_PredicateFormula_", py_formula);
  inherit_term_container(py_predicate_formula);
  py_predicate_formula.def(py::init<>())
      .def("set_predicate", &PredicateFormula::set_predicate,
           py::arg("predicate"))
      .def("get_predicate", &PredicateFormula::get_predicate)
      .def("get_name", &PredicateFormula::get_name)
      .def("__str__", (std::string(PredicateFormula::*)() const) &
                          PredicateFormula::print);

  py::class_<EqualityFormula, EqualityFormula::Ptr> py_equality_formula(
      m, "_PDDL_EqualityFormula_", py_formula);
  inherit_term_container(py_equality_formula);
  py_equality_formula.def(py::init<>())
      .def("get_name", &EqualityFormula::get_name)
      .def("__str__",
           (std::string(EqualityFormula::*)() const) & EqualityFormula::print);

  py::class_<UniversalFormula, UniversalFormula::Ptr> py_universal_formula(
      m, "_PDDL_UniversalFormula_", py_formula);
  py_universal_formula.def(py::init<>())
      .def("set_formula", &UniversalFormula::set_formula, py::arg("formula"))
      .def("get_formula", &UniversalFormula::get_formula)
      .def("__str__", (std::string(UniversalFormula::*)() const) &
                          UniversalFormula::print);

  py::class_<ExistentialFormula, ExistentialFormula::Ptr>
      py_existential_formula(m, "_PDDL_ExistentialFormula_", py_formula);
  py_existential_formula.def(py::init<>())
      .def("set_formula", &ExistentialFormula::set_formula, py::arg("formula"))
      .def("get_formula", &ExistentialFormula::get_formula)
      .def("__str__", (std::string(ExistentialFormula::*)() const) &
                          ExistentialFormula::print);

  py::class_<ConjunctionFormula, ConjunctionFormula::Ptr>
      py_conjunction_formula(m, "_PDDL_ConjunctionFormula_", py_formula);
  py_conjunction_formula.def(py::init<>())
      .def("append_formula", &ConjunctionFormula::append_formula,
           py::arg("formula"))
      .def("remove_formula", &ConjunctionFormula::remove_formula)
      .def("formula_at", &ConjunctionFormula::formula_at, py::arg("index"))
      .def("get_formulas", &ConjunctionFormula::get_formulas)
      .def("__str__", (std::string(ConjunctionFormula::*)() const) &
                          ConjunctionFormula::print);

  py::class_<DisjunctionFormula, DisjunctionFormula::Ptr>
      py_disjunction_formula(m, "_PDDL_DisjunctionFormula_", py_formula);
  py_disjunction_formula.def(py::init<>())
      .def("append_formula", &DisjunctionFormula::append_formula,
           py::arg("formula"))
      .def("remove_formula", &DisjunctionFormula::remove_formula)
      .def("formula_at", &DisjunctionFormula::formula_at, py::arg("index"))
      .def("get_formulas", &DisjunctionFormula::get_formulas)
      .def("__str__", (std::string(DisjunctionFormula::*)() const) &
                          DisjunctionFormula::print);

  py::class_<ImplyFormula, ImplyFormula::Ptr> py_imply_formula(
      m, "_PDDL_ImplyFormula_", py_formula);
  inherit_binary_formula(py_imply_formula);
  py_imply_formula.def(py::init<>());

  py::class_<NegationFormula, NegationFormula::Ptr> py_negation_formula(
      m, "_PDDL_NegationFormula_", py_formula);
  inherit_unary_formula(py_negation_formula);
  py_negation_formula.def(py::init<>());

  py::class_<AtStartFormula, AtStartFormula::Ptr> py_atstart_formula(
      m, "_PDDL_AtStartFormula_", py_formula);
  inherit_unary_formula(py_atstart_formula);
  py_atstart_formula.def(py::init<>());

  py::class_<AtEndFormula, AtEndFormula::Ptr> py_atend_formula(
      m, "_PDDL_AtEndFormula_", py_formula);
  inherit_unary_formula(py_atend_formula);
  py_atend_formula.def(py::init<>());

  py::class_<OverAllFormula, OverAllFormula::Ptr> py_overall_formula(
      m, "_PDDL_OverAllFormula_", py_formula);
  inherit_unary_formula(py_overall_formula);
  py_overall_formula.def(py::init<>());

  py::class_<DurationFormula, DurationFormula::Ptr> py_duration_formula(
      m, "_PDDL_DurationFormula_", py_formula);
  py_duration_formula.def(py::init<>())
      .def("set_durative_action", &DurationFormula::set_durative_action,
           py::arg("durative_action"))
      .def("get_durative_action", &DurationFormula::get_durative_action,
           py::return_value_policy::reference_internal);

  py::class_<GreaterFormula, GreaterFormula::Ptr> py_greater_formula(
      m, "_PDDL_GreaterFormula_", py_formula);
  inherit_binary_expression(py_greater_formula);
  py_greater_formula.def(py::init<>());

  py::class_<GreaterEqFormula, GreaterEqFormula::Ptr> py_greatereq_formula(
      m, "_PDDL_GreaterEqFormula_", py_formula);
  inherit_binary_expression(py_greatereq_formula);
  py_greatereq_formula.def(py::init<>());

  py::class_<LessEqFormula, LessEqFormula::Ptr> py_lesseq_formula(
      m, "_PDDL_LessEqFormula_", py_formula);
  inherit_binary_expression(py_lesseq_formula);
  py_lesseq_formula.def(py::init<>());

  py::class_<LessFormula, LessFormula::Ptr> py_less_formula(
      m, "_PDDL_LessFormula_", py_formula);
  inherit_binary_expression(py_less_formula);
  py_less_formula.def(py::init<>());

  py::class_<EqFormula, EqFormula::Ptr> py_eq_formula(m, "_PDDL_EqFormula_",
                                                      py_formula);
  inherit_binary_expression(py_eq_formula);
  py_eq_formula.def(py::init<>());

  py::class_<Expression, Expression::Ptr> py_expression(m, "_PDDL_Expression_");

  py::class_<AddExpression, AddExpression::Ptr> py_add_expression(
      m, "_PDDL_AddExpression_", py_expression);
  inherit_binary_expression(py_add_expression);
  py_add_expression.def(py::init<>());

  py::class_<SubExpression, SubExpression::Ptr> py_sub_expression(
      m, "_PDDL_SubExpression_", py_expression);
  inherit_binary_expression(py_sub_expression);
  py_sub_expression.def(py::init<>());

  py::class_<MulExpression, MulExpression::Ptr> py_mul_expression(
      m, "_PDDL_MulExpression_", py_expression);
  inherit_binary_expression(py_mul_expression);
  py_mul_expression.def(py::init<>());

  py::class_<DivExpression, DivExpression::Ptr> py_div_expression(
      m, "_PDDL_DivExpression_", py_expression);
  inherit_binary_expression(py_div_expression);
  py_div_expression.def(py::init<>());

  py::class_<MinusExpression, MinusExpression::Ptr> py_minus_expression(
      m, "_PDDL_MinusExpression_", py_expression);
  inherit_unary_expression(py_minus_expression);
  py_minus_expression.def(py::init<>());

  py::class_<Number, Number::Ptr> py_number(m, "_PDDL_Number_");
  py_number.def(py::init<long>())
      .def(py::init<double>())
      .def("is_double", &Number::is_double)
      .def("as_double", &Number::as_double)
      .def("as_long", &Number::as_long)
      .def("__str__", (std::string(Number::*)() const) & Number::print);

  py::class_<NumericalExpression, NumericalExpression::Ptr>
      py_numerical_expression(m, "_PDDL_NumericalExpression_", py_expression);
  py_numerical_expression.def(py::init<>())
      .def("set_number", &NumericalExpression::set_number, py::arg("number"))
      .def("get_number", &NumericalExpression::get_number)
      .def("__str__", (std::string(NumericalExpression::*)() const) &
                          NumericalExpression::print);

  py::class_<FunctionExpression, FunctionExpression::Ptr>
      py_function_expression(m, "_PDDL_FunctionExpression_", py_expression);
  inherit_term_container(py_function_expression);
  py_function_expression.def(py::init<>())
      .def("set_function", &FunctionExpression::set_function,
           py::arg("function"))
      .def("get_function", &FunctionExpression::get_function)
      .def("get_name", &FunctionExpression::get_name)
      .def("__str__", (std::string(FunctionExpression::*)() const) &
                          FunctionExpression::print);

  py::class_<TimeExpression, TimeExpression::Ptr> py_time_expression(
      m, "_PDDL_TimeExpression_");
  py_time_expression.def(py::init<>())
      .def("__str__",
           (std::string(TimeExpression::*)() const) & TimeExpression::print);

  py::class_<DurationExpression, DurationExpression::Ptr>
      py_duration_expression(m, "_PDDL_DurationExpression_");
  py_duration_expression.def(py::init<>())
      .def("__str__", (std::string(DurationExpression::*)() const) &
                          DurationExpression::print);

  py::class_<MinimizeExpression, MinimizeExpression::Ptr>
      py_minimize_expression(m, "_PDDL_MinimizeExpression_", py_expression);
  inherit_unary_expression(py_minimize_expression);
  py_minimize_expression.def(py::init<>());

  py::class_<MaximizeExpression, MaximizeExpression::Ptr>
      py_maximize_expression(m, "_PDDL_MaximizeExpression_", py_expression);
  inherit_unary_expression(py_maximize_expression);
  py_maximize_expression.def(py::init<>());

  py::class_<TotalTimeExpression, TotalTimeExpression::Ptr>
      py_total_time_expression(m, "_PDDL_TotalTimeExpression_");
  py_total_time_expression.def(py::init<>())
      .def("__str__", (std::string(TotalTimeExpression::*)() const) &
                          TotalTimeExpression::print);

  py::class_<TotalCostExpression, TotalCostExpression::Ptr>
      py_total_cost_expression(m, "_PDDL_TotalCostExpression_");
  py_total_cost_expression.def(py::init<>())
      .def("__str__", (std::string(TotalCostExpression::*)() const) &
                          TotalCostExpression::print);

  py::class_<ViolationExpression, ViolationExpression::Ptr>
      py_violation_expression(m, "_PDDL_ViolationExpression_");
  py_violation_expression.def(py::init<>())
      .def("set_preference", &ViolationExpression::set_preference,
           py::arg("preference"))
      .def("get_preference", &ViolationExpression::get_preference)
      .def("__str__", (std::string(ViolationExpression::*)() const) &
                          ViolationExpression::print);

  py::class_<RewardExpression, RewardExpression::Ptr> py_reward_expression(
      m, "_PDDL_RewardExpression_", py_expression);
  py_reward_expression.def(py::init<>())
      .def("__str__", (std::string(RewardExpression::*)() const) &
                          RewardExpression::print);

  py::class_<GoalAchievedExpression, GoalAchievedExpression::Ptr>
      py_goal_achieved_expression(m, "_PDDL_GoalAchievedExpression_",
                                  py_expression);
  py_goal_achieved_expression.def(py::init<>())
      .def("__str__", (std::string(GoalAchievedExpression::*)() const) &
                          GoalAchievedExpression::print);

  py::class_<Effect, Effect::Ptr> py_effect(m, "_PDDL_Effect_");

  py::class_<PredicateEffect, PredicateEffect::Ptr> py_predicate_effect(
      m, "_PDDL_PredicateEffect_", py_effect);
  inherit_term_container(py_predicate_effect);
  py_predicate_effect.def(py::init<>())
      .def("set_predicate", &PredicateEffect::set_predicate,
           py::arg("predicate"))
      .def("get_predicate", &PredicateEffect::get_predicate)
      .def("get_name", &PredicateEffect::get_name)
      .def("__str__",
           (std::string(PredicateEffect::*)() const) & PredicateEffect::print);

  py::class_<ConjunctionEffect, ConjunctionEffect::Ptr> py_conjunction_effect(
      m, "_PDDL_ConjunctionEffect_", py_effect);
  py_conjunction_effect.def(py::init<>())
      .def("append_effect", &ConjunctionEffect::append_effect,
           py::arg("effect"))
      .def("remove_effect", &ConjunctionEffect::remove_effect)
      .def("effect_at", &ConjunctionEffect::effect_at, py::arg("index"))
      .def("get_effects", &ConjunctionEffect::get_effects)
      .def("__str__", (std::string(ConjunctionEffect::*)() const) &
                          ConjunctionEffect::print);

  py::class_<DisjunctionEffect, DisjunctionEffect::Ptr> py_disjunction_effect(
      m, "_PDDL_DisjunctionEffect_", py_effect);
  py_disjunction_effect.def(py::init<>())
      .def("append_effect", &DisjunctionEffect::append_effect,
           py::arg("effect"))
      .def("remove_effect", &DisjunctionEffect::remove_effect)
      .def("effect_at", &DisjunctionEffect::effect_at, py::arg("index"))
      .def("get_effects", &DisjunctionEffect::get_effects)
      .def("__str__", (std::string(DisjunctionEffect::*)() const) &
                          DisjunctionEffect::print);

  py::class_<UniversalEffect, UniversalEffect::Ptr> py_universal_effect(
      m, "_PDDL_UniversalEffect_", py_effect);
  py_universal_effect.def(py::init<>())
      .def("set_effect", &UniversalEffect::set_effect, py::arg("effect"))
      .def("get_effect", &UniversalEffect::get_effect)
      .def("__str__",
           (std::string(UniversalEffect::*)() const) & UniversalEffect::print);

  py::class_<ExistentialEffect, ExistentialEffect::Ptr> py_existential_effect(
      m, "_PDDL_ExistentialEffect_", py_effect);
  py_existential_effect.def(py::init<>())
      .def("set_effect", &ExistentialEffect::set_effect, py::arg("effect"))
      .def("get_effect", &ExistentialEffect::get_effect)
      .def("__str__", (std::string(ExistentialEffect::*)() const) &
                          ExistentialEffect::print);

  py::class_<ConditionalEffect, ConditionalEffect::Ptr> py_conditional_effect(
      m, "_PDDL_ConditionalEffect_", py_effect);
  inherit_binary_effect(py_conditional_effect);
  py_conditional_effect.def(py::init<>());

  py::class_<NegationEffect, NegationEffect::Ptr> py_negation_effect(
      m, "_PDDL_NegationEffect_", py_effect);
  inherit_unary_effect<decltype(py_negation_effect), PredicateEffect>(
      py_negation_effect);
  py_negation_effect.def(py::init<>());

  py::class_<AtStartEffect, AtStartEffect::Ptr> py_atstart_effect(
      m, "_PDDL_AtStartEffect_", py_effect);
  inherit_unary_effect(py_atstart_effect);
  py_atstart_effect.def(py::init<>());

  py::class_<AtEndEffect, AtEndEffect::Ptr> py_atend_effect(
      m, "_PDDL_AtEndEffect_", py_effect);
  inherit_unary_effect(py_atend_effect);
  py_atend_effect.def(py::init<>());

  py::class_<AtTimeEffect, AtTimeEffect::Ptr> py_attime_effect(
      m, "_PDDL_AtTimeEffect_", py_effect);
  inherit_unary_effect(py_attime_effect);
  py_attime_effect.def(py::init<>())
      .def("set_time", &AtTimeEffect::set_time, py::arg("time"))
      .def("get_time", &AtTimeEffect::get_time,
           py::return_value_policy::reference_internal)
      .def("__str__",
           (std::string(AtTimeEffect::*)() const) & AtTimeEffect::print);

  py::class_<DurationEffect, DurationEffect::Ptr> py_duration_effect(
      m, "_PDDL_DurationEffect_", py_effect);
  py_duration_effect.def(py::init<>())
      .def("set_durative_action", &DurationEffect::set_durative_action,
           py::arg("durative_action"))
      .def("get_durative_action", &DurationEffect::get_durative_action,
           py::return_value_policy::reference_internal);

  py::class_<AssignEffect, AssignEffect::Ptr> py_assign_effect(
      m, "_PDDL_AssignEffect_", py_effect);
  inherit_assignment_effect(py_assign_effect);
  py_assign_effect.def(py::init<>());

  py::class_<ScaleUpEffect, ScaleUpEffect::Ptr> py_scaleup_effect(
      m, "_PDDL_ScaleUpEffect_", py_effect);
  inherit_assignment_effect(py_scaleup_effect);
  py_scaleup_effect.def(py::init<>());

  py::class_<ScaleDownEffect, ScaleDownEffect::Ptr> py_scaledown_effect(
      m, "_PDDL_ScaleDownEffect_", py_effect);
  inherit_assignment_effect(py_scaledown_effect);
  py_scaledown_effect.def(py::init<>());

  py::class_<IncreaseEffect, IncreaseEffect::Ptr> py_increase_effect(
      m, "_PDDL_IncreaseEffect_", py_effect);
  inherit_assignment_effect(py_increase_effect);
  py_increase_effect.def(py::init<>());

  py::class_<DecreaseEffect, DecreaseEffect::Ptr> py_decrease_effect(
      m, "_PDDL_DecreaseEffect_", py_effect);
  inherit_assignment_effect(py_decrease_effect);
  py_decrease_effect.def(py::init<>());

  py::class_<ProbabilisticEffect, ProbabilisticEffect::Ptr>
      py_probabilistic_effect(m, "_PDDL_ProbabilisticEffect_", py_effect);
  py_probabilistic_effect.def(py::init<>())
      .def("append_outcome", &ProbabilisticEffect::append_outcome,
           py::arg("probability"), py::arg("effect"))
      .def("remove_outcome", &ProbabilisticEffect::remove_outcome,
           py::arg("index"))
      .def("outcome_at", &ProbabilisticEffect::outcome_at, py::arg("index"))
      .def("get_outcomes", &ProbabilisticEffect::get_outcomes)
      .def("__str__", (std::string(ProbabilisticEffect::*)() const) &
                          ProbabilisticEffect::print);

  py::class_<Action, Action::Ptr> py_action(m, "_PDDL_Action_");
  inherit_identifier(py_action);
  inherit_variable_container(py_action);
  inherit_binary_effect(py_action);
  py_action.def(py::init<std::string>(), py::arg("name"))
      .def("print", &Action::print);

  py::class_<DurativeAction, DurativeAction::Ptr> py_durative_action(
      m, "_PDDL_DurativeAction_", py_action);
  py_durative_action.def(py::init<std::string>(), py::arg("name"))
      .def("set_duration_constraint", &DurativeAction::set_duration_constraint)
      .def("get_duration_constraint", &DurativeAction::get_duration_constraint,
           py::return_value_policy::reference_internal)
      .def("print", &DurativeAction::print);

  py::class_<Event, Event::Ptr> py_event(m, "_PDDL_Event_");
  inherit_identifier(py_event);
  inherit_variable_container(py_event);
  inherit_binary_effect(py_event);
  py_event.def(py::init<std::string>(), py::arg("name"))
      .def("print", &Event::print);

  py::class_<Process, Process::Ptr> py_process(m, "_PDDL_Process_");
  inherit_identifier(py_process);
  inherit_variable_container(py_process);
  inherit_binary_effect(py_process);
  py_process.def(py::init<std::string>(), py::arg("name"))
      .def("print", &Process::print);

  py::class_<Problem, std::shared_ptr<Problem>> py_problem(m, "_PDDL_Problem_");
  inherit_identifier(py_problem);
  inherit_object_container(py_problem);
  inherit_preference_container(py_problem);
  py_problem.def(py::init<std::string>(), py::arg("name"))
      .def("set_domain", &Problem::set_domain, py::arg("domain"))
      .def("get_domain", &Problem::get_domain,
           py::return_value_policy::reference_internal)
      .def("set_requirements", &Problem::set_requirements,
           py::arg("requirements"))
      .def("get_requirements", &Problem::get_requirements,
           py::return_value_policy::reference_internal)
      .def("set_initial_effect", &Problem::set_initial_effect,
           py::arg("initial_effect"))
      .def("get_initial_effect", &Problem::get_initial_effect,
           py::return_value_policy::reference_internal)
      .def("set_goal", &Problem::set_goal, py::arg("goal"))
      .def("get_goal", &Problem::get_goal,
           py::return_value_policy::reference_internal)
      .def("set_constraints", &Problem::set_constraints, py::arg("constraints"))
      .def("get_constraints", &Problem::get_constraints,
           py::return_value_policy::reference_internal)
      .def("set_metric", &Problem::set_metric, py::arg("metric"))
      .def("get_metric", &Problem::get_metric,
           py::return_value_policy::reference_internal)
      .def("set_goal_reward", &Problem::set_goal_reward, py::arg("goal_reward"))
      .def("get_goal_reward", &Problem::get_goal_reward,
           py::return_value_policy::reference_internal)
      .def("__str__", &Problem::print);

  // === Semantics bindings ===

  py::class_<State>(m, "_PDDL_State_")
      .def(py::init<>())
      .def("copy", &State::copy)
      .def("get_atoms",
           [](const State &s) {
             py::list result;
             for (auto &ts : s.atoms) {
               py::set pyset;
               for (auto &tuple : ts) {
                 pyset.add(py::tuple(py::cast(tuple)));
               }
               result.append(pyset);
             }
             return result;
           })
      .def("get_fluents",
           [](const State &s) {
             py::list result;
             for (auto &fm : s.fluents) {
               py::dict pydict;
               for (auto &[k, v] : fm) {
                 pydict[py::tuple(py::cast(k))] = v;
               }
               result.append(pydict);
             }
             return result;
           })
      .def_readwrite("time", &State::time)
      .def("get_active_durative_actions",
           [](const State &s) {
             py::list result;
             for (auto &ada : s.active_durative_actions) {
               py::dict d;
               d["action_id"] = ada.action_id;
               d["start_time"] = ada.start_time;
               d["end_time"] = ada.end_time;
               py::dict binding;
               for (auto &[k, v] : ada.binding) {
                 binding[py::cast(k)] = v;
               }
               d["binding"] = binding;
               result.append(d);
             }
             return result;
           })
      .def("__hash__", &State::hash)
      .def("__eq__", &State::operator==);

  py::class_<GroundAction>(m, "_PDDL_GroundAction_")
      .def(py::init<>())
      .def_readonly("action_id", &GroundAction::action_id)
      .def_readonly("arguments", &GroundAction::arguments)
      .def("get_binding",
           [](const GroundAction &ga) {
             py::dict result;
             for (auto &[k, v] : ga.binding) {
               result[py::cast(k)] = v;
             }
             return result;
           })
      .def("__hash__", &GroundAction::hash)
      .def("__eq__", &GroundAction::operator==);

  py::class_<Task>(m, "_PDDL_Task_")
      .def(py::init<const Task::DomainPtr &, const Task::ProblemPtr &>(),
           py::arg("domain"), py::arg("problem"))
      .def("num_objects", &Task::num_objects)
      .def("object_id", &Task::object_id, py::arg("name"))
      .def("object_name", &Task::object_name, py::arg("id"),
           py::return_value_policy::reference_internal)
      .def("num_predicates", &Task::num_predicates)
      .def("predicate_id", &Task::predicate_id, py::arg("name"))
      .def("predicate_name", &Task::predicate_name, py::arg("id"),
           py::return_value_policy::reference_internal)
      .def("num_functions", &Task::num_functions)
      .def("function_id", &Task::function_id, py::arg("name"))
      .def("function_name", &Task::function_name, py::arg("id"),
           py::return_value_policy::reference_internal)
      .def("total_cost_function", &Task::total_cost_function)
      .def("reward_function", &Task::reward_function)
      .def("initial_state", &Task::initial_state,
           py::return_value_policy::reference_internal)
      .def("action_name", &Task::action_name, py::arg("action_id"),
           py::return_value_policy::reference_internal)
      .def("num_events", &Task::num_events)
      .def("event_name", &Task::event_name, py::arg("event_id"),
           py::return_value_policy::reference_internal)
      .def("num_processes", &Task::num_processes)
      .def("process_name", &Task::process_name, py::arg("process_id"),
           py::return_value_policy::reference_internal)
      .def("num_durative_actions", &Task::num_durative_actions)
      .def("durative_action_name", &Task::durative_action_name,
           py::arg("da_id"), py::return_value_policy::reference_internal)
      .def("domain", &Task::domain,
           py::return_value_policy::reference_internal);

  py::class_<Successor>(m, "_PDDL_Successor_")
      .def_readonly("state", &Successor::state)
      .def_readonly("probability", &Successor::probability);

  py::class_<ApplicableActionsGenerator>(m, "_PDDL_ApplicableActionsGenerator_")
      .def(py::init<const Task &>(), py::arg("task"), py::keep_alive<1, 2>())
      .def("get_applicable_actions",
           &ApplicableActionsGenerator::get_applicable_actions,
           py::arg("state"), py::arg("check_numeric") = true);

  py::class_<SuccessorGenerator>(m, "_PDDL_SuccessorGenerator_")
      .def(py::init<const Task &>(), py::arg("task"), py::keep_alive<1, 2>())
      .def("get_successors", &SuccessorGenerator::get_successors,
           py::arg("state"), py::arg("action"));

  py::class_<GoalChecker>(m, "_PDDL_GoalChecker_")
      .def(py::init<const Task &>(), py::arg("task"), py::keep_alive<1, 2>())
      .def("is_goal", &GoalChecker::is_goal, py::arg("state"));

  py::class_<TemporalSimulator>(m, "_PDDL_TemporalSimulator_")
      .def(py::init([](const Task &task, double epsilon,
                       double max_event_lookahead, int max_cascade_iterations,
                       py::object event_time_finder) {
             TemporalSimulator::EventTimeFinderFn fn = nullptr;
             if (!event_time_finder.is_none()) {
               fn = [event_time_finder](const State &s) -> double {
                 py::gil_scoped_acquire acquire;
                 return event_time_finder(s).cast<double>();
               };
             }
             return new TemporalSimulator(task, epsilon, max_event_lookahead,
                                          max_cascade_iterations,
                                          std::move(fn));
           }),
           py::arg("task"), py::arg("epsilon") = 1e-9,
           py::arg("max_event_lookahead") = 1e6,
           py::arg("max_cascade_iterations") = 100,
           py::arg("event_time_finder") = py::none(), py::keep_alive<1, 2>())
      .def("apply_action", &TemporalSimulator::apply_action, py::arg("state"),
           py::arg("action"))
      .def("integrate_processes", &TemporalSimulator::integrate_processes,
           py::arg("state"), py::arg("dt"))
      .def("fire_events", &TemporalSimulator::fire_events, py::arg("state"))
      .def("start_durative_action", &TemporalSimulator::start_durative_action,
           py::arg("state"), py::arg("da_action"))
      .def("end_durative_action", &TemporalSimulator::end_durative_action,
           py::arg("state"), py::arg("active_index"))
      .def("check_invariants", &TemporalSimulator::check_invariants,
           py::arg("state"))
      .def(
          "time_step",
          [](const TemporalSimulator &sim, const State &state, double dt,
             py::object action) {
            if (action.is_none()) {
              return sim.time_step(state, dt);
            }
            const GroundAction &ga = action.cast<const GroundAction &>();
            return sim.time_step(state, dt, &ga);
          },
          py::arg("state"), py::arg("dt"), py::arg("action") = py::none())
      .def(
          "event_step",
          [](const TemporalSimulator &sim, const State &state,
             py::object action) {
            if (action.is_none()) {
              return sim.event_step(state);
            }
            const GroundAction &ga = action.cast<const GroundAction &>();
            return sim.event_step(state, &ga);
          },
          py::arg("state"), py::arg("action") = py::none())
      .def("get_applicable_actions", &TemporalSimulator::get_applicable_actions,
           py::arg("state"))
      .def("get_applicable_durative_actions",
           &TemporalSimulator::get_applicable_durative_actions,
           py::arg("state"))
      .def("get_active_processes", &TemporalSimulator::get_active_processes,
           py::arg("state"))
      .def("get_triggered_events", &TemporalSimulator::get_triggered_events,
           py::arg("state"))
      .def("is_goal", &TemporalSimulator::is_goal, py::arg("state"))
      .def("get_next_event_time", &TemporalSimulator::get_next_event_time,
           py::arg("state"))
      .def_static("has_z3", &TemporalSimulator::has_z3);

  // === Delete-relaxation heuristics ===

  py::class_<FlatAtomKey>(m, "_PDDL_FlatAtomKey_")
      .def_readonly("predicate_id", &FlatAtomKey::predicate_id)
      .def_readonly("args", &FlatAtomKey::args)
      .def("__eq__",
           [](const FlatAtomKey &a, const FlatAtomKey &b) { return a == b; })
      .def("__hash__",
           [](const FlatAtomKey &k) { return FlatAtomKeyHash{}(k); })
      .def("__repr__", [](const FlatAtomKey &k) {
        std::string s =
            "FlatAtomKey(pred=" + std::to_string(k.predicate_id) + ", args=[";
        for (std::size_t i = 0; i < k.args.size(); ++i) {
          if (i > 0)
            s += ", ";
          s += std::to_string(k.args[i]);
        }
        s += "])";
        return s;
      });

  py::enum_<HeuristicMode>(m, "_PDDL_HeuristicMode_")
      .value("HADD", HeuristicMode::HADD)
      .value("HMAX", HeuristicMode::HMAX);

  py::class_<DeleteRelaxationHeuristic>(m, "_PDDL_DeleteRelaxationHeuristic_")
      .def(py::init<const Task &, HeuristicMode, double, double, bool>(),
           py::arg("task"), py::arg("mode"), py::arg("discount_factor") = 1.0,
           py::arg("dead_end_cost") = 1e9, py::arg("verbose") = false,
           py::keep_alive<1, 2>())
      .def("compute", &DeleteRelaxationHeuristic::compute, py::arg("state"))
      .def(
          "compute_detailed",
          [](const DeleteRelaxationHeuristic &h, const State &state) {
            auto r = h.compute_detailed(state);
            py::dict d;
            d["heuristic_value"] = r.heuristic_value;
            py::list ac;
            for (auto &p : r.atom_costs) {
              ac.append(py::make_tuple(p.first, p.second));
            }
            d["atom_costs"] = ac;
            py::list gc;
            for (auto &p : r.goal_atom_costs) {
              gc.append(py::make_tuple(p.first, p.second));
            }
            d["goal_atom_costs"] = gc;
            return d;
          },
          py::arg("state"))
      .def("mode", &DeleteRelaxationHeuristic::mode)
      .def("discount_factor", &DeleteRelaxationHeuristic::discount_factor)
      .def("num_atoms", &DeleteRelaxationHeuristic::num_atoms)
      .def("num_relaxed_actions",
           &DeleteRelaxationHeuristic::num_relaxed_actions);

  // === FF heuristic ===

  py::class_<FFHeuristic>(m, "_PDDL_FFHeuristic_")
      .def(py::init<const Task &, double, double, bool>(), py::arg("task"),
           py::arg("discount_factor") = 1.0, py::arg("dead_end_cost") = 1e9,
           py::arg("verbose") = false, py::keep_alive<1, 2>())
      .def("compute", &FFHeuristic::compute, py::arg("state"))
      .def("compute_with_helpful", &FFHeuristic::compute_with_helpful,
           py::arg("state"))
      .def(
          "compute_detailed",
          [](const FFHeuristic &h, const State &state) {
            auto r = h.compute_detailed(state);
            py::dict d;
            d["heuristic_value"] = r.heuristic_value;
            py::list ac;
            for (auto &p : r.atom_costs) {
              ac.append(py::make_tuple(p.first, p.second));
            }
            d["atom_costs"] = ac;
            py::list gc;
            for (auto &p : r.goal_atom_costs) {
              gc.append(py::make_tuple(p.first, p.second));
            }
            d["goal_atom_costs"] = gc;
            py::list rpa;
            for (auto &a : r.relaxed_plan_actions) {
              rpa.append(a);
            }
            d["relaxed_plan_actions"] = rpa;
            py::list ha;
            for (auto &a : r.helpful_actions) {
              ha.append(a);
            }
            d["helpful_actions"] = ha;
            py::list ma;
            for (auto &a : r.marked_atoms) {
              ma.append(a);
            }
            d["marked_atoms"] = ma;
            return d;
          },
          py::arg("state"))
      .def("discount_factor", &FFHeuristic::discount_factor)
      .def("num_atoms", &FFHeuristic::num_atoms)
      .def("num_relaxed_actions", &FFHeuristic::num_relaxed_actions);
}
