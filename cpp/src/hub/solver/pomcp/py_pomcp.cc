/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "py_pomcp.hh"

namespace py = pybind11;

void init_pypomcp(py::module &m) {
  py::class_<skdecide::PyPOMCPSolver> py_pomcp_solver(m, "_POMCPSolver_");
  py_pomcp_solver
      .def(py::init<py::object &, py::object &, double, double, std::size_t,
                    std::size_t, double, std::size_t, std::size_t, double, bool,
                    const std::function<py::bool_(const py::object &,
                                                  const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"),
           py::arg("exploration_constant") = 1.0 / std::sqrt(2.0),
           py::arg("discount") = 0.95, py::arg("num_simulations") = 1000,
           py::arg("max_depth") = 100, py::arg("epsilon") = 0.001,
           py::arg("time_budget") = 0,
           py::arg("num_particles_belief_update") = 500,
           py::arg("ess_threshold_ratio") = 2.0, py::arg("parallel") = false,
           py::arg("callback") = nullptr, py::arg("verbose") = false)
      .def("close", &skdecide::PyPOMCPSolver::close)
      .def("clear", &skdecide::PyPOMCPSolver::clear)
      .def("solve", &skdecide::PyPOMCPSolver::solve, py::arg("distribution"))
      .def("get_next_action", &skdecide::PyPOMCPSolver::get_next_action,
           py::arg("observation"))
      .def("get_utility", &skdecide::PyPOMCPSolver::get_utility,
           py::arg("observation"))
      .def("is_solution_defined_for",
           &skdecide::PyPOMCPSolver::is_solution_defined_for,
           py::arg("observation"))
      .def("reset_belief", &skdecide::PyPOMCPSolver::reset_belief)
      .def("get_next_action_from_belief",
           &skdecide::PyPOMCPSolver::get_next_action_from_belief,
           py::arg("belief"))
      .def("get_utility_from_belief",
           &skdecide::PyPOMCPSolver::get_utility_from_belief, py::arg("belief"))
      .def("is_solution_defined_for_from_belief",
           &skdecide::PyPOMCPSolver::is_solution_defined_for_from_belief,
           py::arg("belief"))
      .def("get_nb_tree_nodes", &skdecide::PyPOMCPSolver::get_nb_tree_nodes)
      .def("get_solving_time", &skdecide::PyPOMCPSolver::get_solving_time)
      .def("get_last_trajectory",
           &skdecide::PyPOMCPSolver::get_last_trajectory);
}
