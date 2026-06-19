/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_ehc.hh"

void init_pyehc(py::module &m) {
  py::class_<skdecide::PyEHCSolver> py_ehc_solver(m, "_EHCSolver_");
  py_ehc_solver
      .def(py::init<py::object &, py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    bool, const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"), py::arg("preferred_actions"),
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PyEHCSolver::close)
      .def("clear", &skdecide::PyEHCSolver::clear)
      .def("solve", &skdecide::PyEHCSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyEHCSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyEHCSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyEHCSolver::get_utility, py::arg("state"))
      .def("get_nb_explored_states",
           &skdecide::PyEHCSolver::get_nb_explored_states)
      .def("get_explored_states", &skdecide::PyEHCSolver::get_explored_states)
      .def("get_solving_time", &skdecide::PyEHCSolver::get_solving_time)
      .def("get_plan", &skdecide::PyEHCSolver::get_plan)
      .def("get_policy", &skdecide::PyEHCSolver::get_policy);
}
