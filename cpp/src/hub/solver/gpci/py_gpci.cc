/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_gpci.hh"

void init_pygpci(py::module &m) {
  py::class_<skdecide::PyGPCISolver> py_gpci_solver(m, "_GPCISolver_");
  py_gpci_solver
      .def(py::init<py::object &, py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    double, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("epsilon") = 0.001, py::arg("parallel") = false,
           py::arg("callback") = nullptr, py::arg("verbose") = false)
      .def("close", &skdecide::PyGPCISolver::close)
      .def("clear", &skdecide::PyGPCISolver::clear)
      .def("solve", &skdecide::PyGPCISolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyGPCISolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyGPCISolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyGPCISolver::get_utility,
           py::arg("state"))
      .def("get_goal_probability",
           &skdecide::PyGPCISolver::get_goal_probability, py::arg("state"))
      .def("get_goal_cost", &skdecide::PyGPCISolver::get_goal_cost,
           py::arg("state"))
      .def("get_nb_explored_states",
           &skdecide::PyGPCISolver::get_nb_explored_states)
      .def("get_nb_prob_iterations",
           &skdecide::PyGPCISolver::get_nb_prob_iterations)
      .def("get_nb_cost_iterations",
           &skdecide::PyGPCISolver::get_nb_cost_iterations)
      .def("get_current_phase", &skdecide::PyGPCISolver::get_current_phase)
      .def("get_solving_time", &skdecide::PyGPCISolver::get_solving_time)
      .def("get_explored_states", &skdecide::PyGPCISolver::get_explored_states)
      .def("get_policy", &skdecide::PyGPCISolver::get_policy);
}
