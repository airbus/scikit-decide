#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <sstream>

#include "pddl.hh"

namespace py = pybind11;
using namespace airlaps::pddl;

template <typename Instance>
void inherit_type_container(Instance& instance) {
    using TypeContainer = typename Instance::type;
    instance.def("get_name", &TypeContainer::get_name,
                             py::return_value_policy::reference_internal)
            .def("add_type", (const Domain::TypePtr& (TypeContainer::*)(const std::string&)) &TypeContainer::add_type,
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
    inherit_type_container(py_domain);
        py_domain
            .def(py::init<>())
            .def("set_name", &Domain::set_name, py::arg("name"))
            .def("get_name", &Domain::get_name,
                             py::return_value_policy::reference_internal)
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
    inherit_type_container(py_type);
        py_type
            .def(py::init<std::string>(), py::arg("name"))
        ;
    
    py::class_<Object, Domain::ObjectPtr> py_object(m, "_PDDL_Object_");
    inherit_type_container(py_object);
        py_object
            .def(py::init<std::string>(), py::arg("name"))
        ;
    
    // py::class_<Variable, Domain::VariablePtr> py_variable(m, "_PDDL_Variable_");
    // inherit_type_container(py_variable);
    //     py_variable
    //         .def(py::init<std::string>(), py::arg("name"))
    //     ;
}
