/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_lrtdp.hh"

void init_pylrtdp(py::module &m) {
  py::class_<skdecide::PyLRTDPSolver> py_lrtdp_solver(m, "_LRTDPSolver_");
  py_lrtdp_solver
      .def(
          py::init<py::object &, // Python solver
                   py::object &, // Python domain
                   const std::function<py::object(
                       py::object &, const py::object &,
                       const py::object &)> // last arg used for
                                            // optional thread_id
                       &,
                   const std::function<py::object(
                       py::object &, const py::object &,
                       const py::object &)> // last arg used for
                                            // optional thread_id
                       &,
                   const std::function<py::object(const py::object &)> &, bool,
                   std::size_t, std::size_t, std::size_t, std::size_t, double,
                   double, bool, bool,
                   const std::function<py::bool_(
                       const py::object &, const py::object &)> // last arg used
                                                                // for optional
                                                                // thread_id
                       &,
                   bool>(),
          py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
          py::arg("heuristic"), py::arg("terminal_value") = nullptr,
          py::arg("use_labels") = true, py::arg("time_budget") = 3600000,
          py::arg("rollout_budget") = 100000, py::arg("max_depth") = 1000,
          py::arg("residual_moving_average_window") = 100,
          py::arg("epsilon") = 0.001, py::arg("discount") = 1.0,
          py::arg("online_node_garbage") = false, py::arg("parallel") = false,
          py::arg("callback") = nullptr, py::arg("verbose") = false)
      .def("close", &skdecide::PyLRTDPSolver::close)
      .def("clear", &skdecide::PyLRTDPSolver::clear)
      .def("solve", &skdecide::PyLRTDPSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyLRTDPSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyLRTDPSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyLRTDPSolver::get_utility,
           py::arg("state"))
      .def("get_nb_explored_states",
           &skdecide::PyLRTDPSolver::get_nb_explored_states)
      .def("get_nb_rollouts", &skdecide::PyLRTDPSolver::get_nb_rollouts)
      .def("get_residual_moving_average",
           &skdecide::PyLRTDPSolver::get_residual_moving_average)
      .def("get_solving_time", &skdecide::PyLRTDPSolver::get_solving_time)
      .def("get_explored_states", &skdecide::PyLRTDPSolver::get_explored_states)
      .def("get_solved_states", &skdecide::PyLRTDPSolver::get_solved_states)
      .def("get_policy", &skdecide::PyLRTDPSolver::get_policy);

  py::class_<skdecide::PyLRTAstarSolver, skdecide::PyLRTDPSolver>
      py_lrtastar_solver(m, "_LRTAstarSolver_");
  py_lrtastar_solver
      .def(
          py::init<py::object &, py::object &,
                   const std::function<py::object(
                       py::object &, const py::object &, const py::object &)> &,
                   const std::function<py::object(
                       py::object &, const py::object &, const py::object &)> &,
                   std::size_t, std::size_t, std::size_t, bool,
                   const std::function<py::bool_(const py::object &,
                                                 const py::object &)> &,
                   bool>(),
          py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
          py::arg("heuristic"), py::arg("time_budget") = 3600000,
          py::arg("rollout_budget") = 100000, py::arg("max_depth") = 1000,
          py::arg("parallel") = false, py::arg("callback") = nullptr,
          py::arg("verbose") = false)
      .def("get_plan", &skdecide::PyLRTAstarSolver::get_plan, py::arg("state"));
}
