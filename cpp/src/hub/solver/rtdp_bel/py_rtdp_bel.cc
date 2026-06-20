/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_rtdp_bel.hh"

void init_pyrtdp_bel(py::module &m) {
  py::class_<skdecide::PyRTDPBelSolver> py_rtdp_bel_solver(m,
                                                           "_RTDPBelSolver_");
  py_rtdp_bel_solver
      .def(py::init<py::object &, py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    std::size_t, std::size_t, std::size_t, std::size_t, double,
                    double, bool,
                    const std::function<py::bool_(const py::object &,
                                                  const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"), py::arg("discretization") = 10,
           py::arg("time_budget") = 3600000, py::arg("rollout_budget") = 100000,
           py::arg("max_depth") = 1000, py::arg("epsilon") = 0.001,
           py::arg("discount") = 1.0, py::arg("parallel") = false,
           py::arg("callback") = nullptr, py::arg("verbose") = false)
      .def("close", &skdecide::PyRTDPBelSolver::close)
      .def("clear", &skdecide::PyRTDPBelSolver::clear)
      .def("solve", &skdecide::PyRTDPBelSolver::solve, py::arg("distribution"))
      .def("get_next_action", &skdecide::PyRTDPBelSolver::get_next_action,
           py::arg("observation"))
      .def("get_utility", &skdecide::PyRTDPBelSolver::get_utility,
           py::arg("observation"))
      .def("is_solution_defined_for",
           &skdecide::PyRTDPBelSolver::is_solution_defined_for,
           py::arg("observation"))
      .def("get_policy", &skdecide::PyRTDPBelSolver::get_policy,
           py::arg("observation"))
      .def("reset_belief", &skdecide::PyRTDPBelSolver::reset_belief)
      .def("get_nb_explored_beliefs",
           &skdecide::PyRTDPBelSolver::get_nb_explored_beliefs)
      .def("get_explored_beliefs",
           &skdecide::PyRTDPBelSolver::get_explored_beliefs)
      .def("get_nb_rollouts", &skdecide::PyRTDPBelSolver::get_nb_rollouts)
      .def("get_solving_time", &skdecide::PyRTDPBelSolver::get_solving_time)
      .def("get_last_trajectory",
           &skdecide::PyRTDPBelSolver::get_last_trajectory)
      .def("get_belief_policy", &skdecide::PyRTDPBelSolver::get_belief_policy)
      .def("get_next_action_from_belief",
           &skdecide::PyRTDPBelSolver::get_next_action_from_belief,
           py::arg("belief"))
      .def("get_utility_from_belief",
           &skdecide::PyRTDPBelSolver::get_utility_from_belief,
           py::arg("belief"))
      .def("is_solution_defined_for_from_belief",
           &skdecide::PyRTDPBelSolver::is_solution_defined_for_from_belief,
           py::arg("belief"));
}
