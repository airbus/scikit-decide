/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_aostar.hh"

void init_pyaostar(py::module &m) {
  py::class_<skdecide::PyAOStarSolver> py_aostar_solver(m, "_AOStarSolver_");
  py_aostar_solver
      .def(py::init<py::object &, // Python solver
                    py::object &, // Python domain
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    double, std::size_t, bool, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"), py::arg("discount") = 1.0,
           py::arg("max_tip_expansions") = 1, py::arg("detect_cycles") = false,
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PyAOStarSolver::close)
      .def("clear", &skdecide::PyAOStarSolver::clear)
      .def("solve", &skdecide::PyAOStarSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyAOStarSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyAOStarSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyAOStarSolver::get_utility,
           py::arg("state"))
      .def("get_nb_explored_states",
           &skdecide::PyAOStarSolver::get_nb_explored_states)
      .def("get_explored_states",
           &skdecide::PyAOStarSolver::get_explored_states)
      .def("get_nb_tip_states", &skdecide::PyAOStarSolver::get_nb_tip_states)
      .def("get_top_tip_state", &skdecide::PyAOStarSolver::get_top_tip_state)
      .def("get_solving_time", &skdecide::PyAOStarSolver::get_solving_time)
      .def("get_policy", &skdecide::PyAOStarSolver::get_policy);
}
