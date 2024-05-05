/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "hub/solver/lrtdp/lrtdp.hh"
#include "py_lrtdp.hh"

void init_pylrtdp(py::module &m) {
  py::class_<skdecide::PyLRTDPSolver> py_lrtdp_solver(m, "_LRTDPSolver_");
  py_lrtdp_solver
      .def(py::init<
               py::object &, // Python solver
               py::object &, // Python domain
               const std::function<py::object(
                   py::object &, const py::object &,
                   const py::object &)> // last arg used for optional thread_id
                   &,
               const std::function<py::object(
                   py::object &, const py::object &,
                   const py::object &)> // last arg used for optional thread_id
                   &,
               bool, std::size_t, std::size_t, std::size_t, std::size_t, double,
               double, bool, bool, bool,
               const std::function<py::bool_(
                   py::object &, const py::object &,
                   const py::object &)> // last arg used for optional thread_id
                   &>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"), py::arg("use_labels") = false,
           py::arg("time_budget") = 3600000, py::arg("rollout_budget") = 100000,
           py::arg("max_depth") = 1000,
           py::arg("epsilon_moving_average_window") = 100,
           py::arg("epsilon") = 0.001, py::arg("discount") = 1.0,
           py::arg("online_node_garbage") = false, py::arg("parallel") = false,
           py::arg("debug_logs") = false, py::arg("callback") = nullptr)
      .def("close", &skdecide::PyLRTDPSolver::close)
      .def("clear", &skdecide::PyLRTDPSolver::clear)
      .def("solve", &skdecide::PyLRTDPSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyLRTDPSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyLRTDPSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyLRTDPSolver::get_utility,
           py::arg("state"))
      .def("get_nb_of_explored_states",
           &skdecide::PyLRTDPSolver::get_nb_of_explored_states)
      .def("get_nb_rollouts", &skdecide::PyLRTDPSolver::get_nb_rollouts)
      .def("get_residual_moving_average",
           &skdecide::PyLRTDPSolver::get_residual_moving_average)
      .def("get_solving_time", &skdecide::PyLRTDPSolver::get_solving_time)
      .def("get_policy", &skdecide::PyLRTDPSolver::get_policy);
}
