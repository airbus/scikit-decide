/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_bfws.hh"

namespace py = pybind11;

void init_pybfws(py::module &m) {
  py::class_<skdecide::PyBFWSSolver> py_bfws_solver(m, "_BFWSSolver_");
  py_bfws_solver
      .def(py::init<py::object &, // Python solver
                    py::object &, // Python domain
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    bool, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("state_features"),
           py::arg("heuristic"), py::arg("termination_checker"),
           py::arg("use_state_feature_hash") = false,
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PyBFWSSolver::close)
      .def("clear", &skdecide::PyBFWSSolver::clear)
      .def("solve", &skdecide::PyBFWSSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyBFWSSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyBFWSSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyBFWSSolver::get_utility,
           py::arg("state"))
      .def("get_nb_explored_states",
           &skdecide::PyBFWSSolver::get_nb_explored_states)
      .def("get_explored_states", &skdecide::PyBFWSSolver::get_explored_states)
      .def("get_nb_tip_states", &skdecide::PyBFWSSolver::get_nb_tip_states)
      .def("get_top_tip_state", &skdecide::PyBFWSSolver::get_top_tip_state)
      .def("get_solving_time", &skdecide::PyBFWSSolver::get_solving_time)
      .def("get_plan", &skdecide::PyBFWSSolver::get_plan)
      .def("get_policy", &skdecide::PyBFWSSolver::get_policy);
}
