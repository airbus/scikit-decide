#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <sstream>

#include "pddl.hh"

namespace py = pybind11;

void init_pypddl(py::module& m) {
    py::class_<airlaps::pddl::PDDL> py_pddl(m, "_PDDL_");
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
    
    py::class_<airlaps::pddl::Domain> py_domain(m, "_PDDL_Domain_");
        py_domain
            .def(py::init<>())
            .def("set_name", &airlaps::pddl::Domain::set_name, py::arg("name"))
            .def("get_name", &airlaps::pddl::Domain::get_name,
                             py::return_value_policy::reference_internal)
            .def("set_requirements", &airlaps::pddl::Domain::set_requirements, py::arg("requirements"))
            .def("get_requirements", &airlaps::pddl::Domain::get_requirements,
                                     py::return_value_policy::reference_internal)
            .def("add_type", [](airlaps::pddl::Domain& d, const py::object& t) {
                if (py::isinstance<py::str>(t)) {
                    return d.add_type(py::cast<std::string>(t));
                } else {
                    return d.add_type(py::cast<airlaps::pddl::Type::Ptr>(t));
                }
            }, py::arg("type"), py::return_value_policy::reference_internal)
            .def("remove_type", [](airlaps::pddl::Domain& d, const py::object& t) {
                if (py::isinstance<py::str>(t)) {
                    d.remove_type(py::cast<std::string>(t));
                } else {
                    d.remove_type(py::cast<airlaps::pddl::Type::Ptr>(t));
                }
            }, py::arg("type"))
            .def("get_type", &airlaps::pddl::Domain::get_type, py::arg("type"),
                             py::return_value_policy::reference_internal)
            .def("get_types", &airlaps::pddl::Domain::get_types,
                              py::return_value_policy::reference_internal)
            .def("__str__", &airlaps::pddl::Domain::print)
        ;
    
    py::class_<airlaps::pddl::Requirements> py_requirements(m, "_PDDL_Requirements_");
        py_requirements
            .def(py::init<>())
            .def("set_equality", &airlaps::pddl::Requirements::set_equality)
            .def("has_equality", &airlaps::pddl::Requirements::has_equality)
            .def("set_strips", &airlaps::pddl::Requirements::set_strips)
            .def("has_strips", &airlaps::pddl::Requirements::has_strips)
            .def("set_typing", &airlaps::pddl::Requirements::set_typing)
            .def("has_typing", &airlaps::pddl::Requirements::has_typing)
            .def("set_negative_preconditions", &airlaps::pddl::Requirements::set_negative_preconditions)
            .def("has_negative_preconditions", &airlaps::pddl::Requirements::has_negative_preconditions)
            .def("set_disjunctive_preconditions", &airlaps::pddl::Requirements::set_disjunctive_preconditions)
            .def("has_disjunctive_preconditions", &airlaps::pddl::Requirements::has_disjunctive_preconditions)
            .def("set_existential_preconditions", &airlaps::pddl::Requirements::set_existential_preconditions)
            .def("has_existential_preconditions", &airlaps::pddl::Requirements::has_existential_preconditions)
            .def("set_universal_preconditions", &airlaps::pddl::Requirements::set_universal_preconditions)
            .def("has_universal_preconditions", &airlaps::pddl::Requirements::has_universal_preconditions)
            .def("set_conditional_effects", &airlaps::pddl::Requirements::set_conditional_effects)
            .def("has_conditional_effects", &airlaps::pddl::Requirements::has_conditional_effects)
            .def("set_fluents", &airlaps::pddl::Requirements::set_fluents)
            .def("has_fluents", &airlaps::pddl::Requirements::has_fluents)
            .def("set_durative_actions", &airlaps::pddl::Requirements::set_durative_actions)
            .def("has_durative_actions", &airlaps::pddl::Requirements::has_durative_actions)
            .def("set_time", &airlaps::pddl::Requirements::set_time)
            .def("has_time", &airlaps::pddl::Requirements::has_time)
            .def("set_action_costs", &airlaps::pddl::Requirements::set_action_costs)
            .def("has_action_costs", &airlaps::pddl::Requirements::has_action_costs)
            .def("set_object_fluents", &airlaps::pddl::Requirements::set_object_fluents)
            .def("has_object_fluents", &airlaps::pddl::Requirements::has_object_fluents)
            .def("set_numeric_fluents", &airlaps::pddl::Requirements::set_numeric_fluents)
            .def("has_numeric_fluents", &airlaps::pddl::Requirements::has_numeric_fluents)
            .def("set_modules", &airlaps::pddl::Requirements::set_modules)
            .def("has_modules", &airlaps::pddl::Requirements::has_modules)
            .def("set_adl", &airlaps::pddl::Requirements::set_adl)
            .def("has_adl", &airlaps::pddl::Requirements::has_adl)
            .def("set_quantified_preconditions", &airlaps::pddl::Requirements::set_quantified_preconditions)
            .def("has_quantified_preconditions", &airlaps::pddl::Requirements::has_quantified_preconditions)
            .def("set_duration_inequalities", &airlaps::pddl::Requirements::set_duration_inequalities)
            .def("has_duration_inequalities", &airlaps::pddl::Requirements::has_duration_inequalities)
            .def("set_continuous_effects", &airlaps::pddl::Requirements::set_continuous_effects)
            .def("has_continuous_effects", &airlaps::pddl::Requirements::has_continuous_effects)
            .def("set_derived_predicates", &airlaps::pddl::Requirements::set_derived_predicates)
            .def("has_derived_predicates", &airlaps::pddl::Requirements::has_derived_predicates)
            .def("set_timed_initial_literals", &airlaps::pddl::Requirements::set_timed_initial_literals)
            .def("has_timed_initial_literals", &airlaps::pddl::Requirements::has_timed_initial_literals)
            .def("set_preferences", &airlaps::pddl::Requirements::set_preferences)
            .def("has_preferences", &airlaps::pddl::Requirements::has_preferences)
            .def("set_constraints", &airlaps::pddl::Requirements::set_constraints)
            .def("has_constraints", &airlaps::pddl::Requirements::has_constraints)
            .def("__str__", &airlaps::pddl::Requirements::print)
        ;
    
    py::class_<airlaps::pddl::Type, airlaps::pddl::Type::Ptr> py_type(m, "_PDDL_Type_");
        py_type
            .def(py::init<std::string>(), py::arg("name"))
            .def("get_name", &airlaps::pddl::Type::get_name,
                             py::return_value_policy::reference_internal)
            .def("add_parent", &airlaps::pddl::Type::add_parent, py::arg("type"))
            .def("remove_parent", &airlaps::pddl::Type::remove_parent, py::arg("type"))
            .def("get_parent", &airlaps::pddl::Type::get_parent, py::arg("type"),
                               py::return_value_policy::reference_internal)
            .def("get_parents", &airlaps::pddl::Type::get_parents,
                                py::return_value_policy::reference_internal)
            .def("__str__", &airlaps::pddl::Type::print)
        ;
    
    py::class_<airlaps::pddl::Object, airlaps::pddl::Object::Ptr> py_object(m, "_PDDL_Object_");
        py_object
            .def(py::init<std::string>(), py::arg("name"))
            .def("get_name", &airlaps::pddl::Object::get_name,
                             py::return_value_policy::reference_internal)
            .def("add_type", &airlaps::pddl::Object::add_type, py::arg("type"))
            .def("remove_type", &airlaps::pddl::Object::remove_type, py::arg("type"))
            .def("get_type", &airlaps::pddl::Object::get_type, py::arg("type"),
                               py::return_value_policy::reference_internal)
            .def("get_types", &airlaps::pddl::Object::get_types,
                                py::return_value_policy::reference_internal)
            .def("__str__", &airlaps::pddl::Object::print)
        ;
}
