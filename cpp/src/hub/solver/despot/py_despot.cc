/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "py_despot.hh"

namespace py = pybind11;

void init_pydespot(py::module &m) {
  py::class_<skdecide::PyDespotSolver> py_despot_solver(m, "_DespotSolver_");
  py_despot_solver
      .def(py::init<py::object &, py::object &, std::size_t, std::size_t,
                    double, double, double, std::size_t, double, std::size_t,
                    std::size_t, double,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    bool,
                    const std::function<py::bool_(const py::object &,
                                                  const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("num_scenarios") = 500,
           py::arg("max_depth") = 90, py::arg("regularization_constant") = 0.0,
           py::arg("gap_reduction_rate") = 0.95, py::arg("target_gap") = 0.0,
           py::arg("time_budget") = 1000, py::arg("discount") = 0.95,
           py::arg("max_rollout_depth") = 90,
           py::arg("num_particles_belief_update") = 500,
           py::arg("ess_threshold_ratio") = 2.0,
           py::arg("default_policy") = nullptr,
           py::arg("upper_bound_heuristic") = nullptr,
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PyDespotSolver::close)
      .def("clear", &skdecide::PyDespotSolver::clear)
      .def("solve", &skdecide::PyDespotSolver::solve, py::arg("distribution"))
      .def("get_next_action", &skdecide::PyDespotSolver::get_next_action,
           py::arg("observation"))
      .def("get_utility", &skdecide::PyDespotSolver::get_utility,
           py::arg("observation"))
      .def("is_solution_defined_for",
           &skdecide::PyDespotSolver::is_solution_defined_for,
           py::arg("observation"))
      .def("reset_belief", &skdecide::PyDespotSolver::reset_belief)
      .def("get_next_action_from_belief",
           &skdecide::PyDespotSolver::get_next_action_from_belief,
           py::arg("belief"))
      .def("get_utility_from_belief",
           &skdecide::PyDespotSolver::get_utility_from_belief,
           py::arg("belief"))
      .def("is_solution_defined_for_from_belief",
           &skdecide::PyDespotSolver::is_solution_defined_for_from_belief,
           py::arg("belief"))
      .def("get_nb_tree_nodes", &skdecide::PyDespotSolver::get_nb_tree_nodes)
      .def("get_solving_time", &skdecide::PyDespotSolver::get_solving_time)
      .def("get_gap", &skdecide::PyDespotSolver::get_gap);
}
