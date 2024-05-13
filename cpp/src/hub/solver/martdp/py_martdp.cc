/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_martdp.hh"

namespace py = pybind11;

void init_pymartdp(py::module &m) {
  py::class_<skdecide::PyMARTDPSolver> py_martdp_solver(m, "_MARTDPSolver_");
  py_martdp_solver
      .def(py::init<py::object &, // Python solver
                    py::object &, // Python domain
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    std::size_t, std::size_t, std::size_t, std::size_t, double,
                    std::size_t, double, double, double, double, bool, bool,
                    const std::function<py::bool_(const py::object &)> &>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"), py::arg("time_budget") = 3600000,
           py::arg("rollout_budget") = 100000, py::arg("max_depth") = 1000,
           py::arg("max_feasibility_trials") = 0,
           py::arg("graph_expansion_rate") = 0.1,
           py::arg("residual_moving_average_window") = 100,
           py::arg("epsilon") = 0.0, // not a stopping criterion by default
           py::arg("discount") = 1.0, py::arg("action_choice_noise") = 0.1,
           py::arg("dead_end_cost") = 10e4,
           py::arg("online_node_garbage") = false,
           py::arg("debug_logs") = false, py::arg("callback") = nullptr)
      .def("close", &skdecide::PyMARTDPSolver::close)
      .def("clear", &skdecide::PyMARTDPSolver::clear)
      .def("solve", &skdecide::PyMARTDPSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyMARTDPSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyMARTDPSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyMARTDPSolver::get_utility,
           py::arg("state"))
      .def("get_nb_explored_states",
           &skdecide::PyMARTDPSolver::get_nb_explored_states)
      .def("get_state_nb_actions",
           &skdecide::PyMARTDPSolver::get_state_nb_actions, py::arg("state"))
      .def("get_nb_rollouts", &skdecide::PyMARTDPSolver::get_nb_rollouts)
      .def("get_residual_moving_average",
           &skdecide::PyMARTDPSolver::get_residual_moving_average)
      .def("get_solving_time", &skdecide::PyMARTDPSolver::get_solving_time)
      .def("get_policy", &skdecide::PyMARTDPSolver::get_policy);
}
