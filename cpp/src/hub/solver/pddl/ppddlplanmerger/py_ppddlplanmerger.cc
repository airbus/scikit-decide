/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "py_ppddlplanmerger.hh"

void init_pyppddlplanmerger(py::module &m) {
  // PPDDLPlanMerger — pluggable inner solver
  py::class_<skdecide::PyPPDDLPlanMergerSolver>(m, "_PPDDLPlanMergerSolver_")
      .def(
          py::init<py::object &, const skdecide::pddl::Task &,
                   const std::string &, const std::string &, bool, double,
                   double, std::size_t, std::size_t, std::size_t, bool, double,
                   double, const std::function<py::bool_(const py::object &)> &,
                   bool, const py::dict &>(),
          py::arg("solver"), py::arg("task"),
          py::arg("inner_solver_name") = "FF",
          py::arg("determinization") = "most_probable_outcome",
          py::arg("parallel") = false, py::arg("dead_end_cost") = 1e9,
          py::arg("rho") = 0.1, py::arg("mc_samples") = 100,
          py::arg("max_iterations") = 50, py::arg("max_steps") = 10000,
          py::arg("optimize_policy_graph") = false, py::arg("discount") = 0.99,
          py::arg("epsilon") = 1e-3, py::arg("callback") = nullptr,
          py::arg("verbose") = false,
          py::arg("inner_solver_params") = py::dict(), py::keep_alive<1, 3>())
      .def("solve", &skdecide::PyPPDDLPlanMergerSolver::solve, py::arg("state"))
      .def("resolve", &skdecide::PyPPDDLPlanMergerSolver::resolve,
           py::arg("state"))
      .def("clear", &skdecide::PyPPDDLPlanMergerSolver::clear)
      .def("is_solution_defined_for",
           &skdecide::PyPPDDLPlanMergerSolver::is_solution_defined_for,
           py::arg("state"))
      .def("get_next_action",
           &skdecide::PyPPDDLPlanMergerSolver::get_next_action,
           py::arg("state"))
      .def("get_nb_iterations",
           &skdecide::PyPPDDLPlanMergerSolver::get_nb_iterations)
      .def("get_nb_plans", &skdecide::PyPPDDLPlanMergerSolver::get_nb_plans)
      .def("get_solving_time",
           &skdecide::PyPPDDLPlanMergerSolver::get_solving_time)
      .def("get_policy_size",
           &skdecide::PyPPDDLPlanMergerSolver::get_policy_size)
      .def("get_best_value", &skdecide::PyPPDDLPlanMergerSolver::get_best_value,
           py::arg("state"))
      .def("get_explored_states",
           &skdecide::PyPPDDLPlanMergerSolver::get_explored_states)
      .def("get_terminal_states",
           &skdecide::PyPPDDLPlanMergerSolver::get_terminal_states)
      .def("get_policy", &skdecide::PyPPDDLPlanMergerSolver::get_policy);

  // RFF — FF fixed as inner solver
  py::class_<skdecide::PyRFFSolver>(m, "_RFFSolver_")
      .def(py::init<py::object &, const skdecide::pddl::Task &,
                    const std::string &, bool, double, double, std::size_t,
                    std::size_t, std::size_t, bool, double, double,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("task"),
           py::arg("determinization") = "most_probable_outcome",
           py::arg("parallel") = false, py::arg("dead_end_cost") = 1e9,
           py::arg("rho") = 0.1, py::arg("mc_samples") = 100,
           py::arg("max_iterations") = 50, py::arg("max_steps") = 10000,
           py::arg("optimize_policy_graph") = false, py::arg("discount") = 0.99,
           py::arg("epsilon") = 1e-3, py::arg("callback") = nullptr,
           py::arg("verbose") = false, py::keep_alive<1, 3>())
      .def("solve", &skdecide::PyRFFSolver::solve, py::arg("state"))
      .def("resolve", &skdecide::PyRFFSolver::resolve, py::arg("state"))
      .def("clear", &skdecide::PyRFFSolver::clear)
      .def("is_solution_defined_for",
           &skdecide::PyRFFSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyRFFSolver::get_next_action,
           py::arg("state"))
      .def("get_nb_iterations", &skdecide::PyRFFSolver::get_nb_iterations)
      .def("get_nb_plans", &skdecide::PyRFFSolver::get_nb_plans)
      .def("get_solving_time", &skdecide::PyRFFSolver::get_solving_time)
      .def("get_policy_size", &skdecide::PyRFFSolver::get_policy_size)
      .def("get_best_value", &skdecide::PyRFFSolver::get_best_value,
           py::arg("state"))
      .def("get_explored_states", &skdecide::PyRFFSolver::get_explored_states)
      .def("get_terminal_states", &skdecide::PyRFFSolver::get_terminal_states)
      .def("get_policy", &skdecide::PyRFFSolver::get_policy);
}
