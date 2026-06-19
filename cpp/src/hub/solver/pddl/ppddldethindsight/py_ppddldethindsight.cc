/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "py_ppddldethindsight.hh"

void init_pyppddldethindsight(py::module &m) {
  // PPDDLDetHindsight — pluggable inner solver
  py::class_<skdecide::PyPPDDLDetHindsightSolver>(m,
                                                  "_PPDDLDetHindsightSolver_")
      .def(py::init<py::object &, const skdecide::pddl::Task &,
                    const std::string &, bool, std::size_t, double, std::size_t,
                    double, double,
                    const std::function<py::bool_(const py::object &)> &, bool,
                    const py::dict &>(),
           py::arg("solver"), py::arg("task"),
           py::arg("inner_solver_name") = "FF", py::arg("parallel") = false,
           py::arg("sample_width") = 30, py::arg("dead_end_cost") = 1e9,
           py::arg("max_steps") = 10000, py::arg("discount") = 0.99,
           py::arg("epsilon") = 1e-3, py::arg("callback") = nullptr,
           py::arg("verbose") = false,
           py::arg("inner_solver_params") = py::dict(), py::keep_alive<1, 3>())
      .def("solve", &skdecide::PyPPDDLDetHindsightSolver::solve,
           py::arg("state"))
      .def("clear", &skdecide::PyPPDDLDetHindsightSolver::clear)
      .def("is_solution_defined_for",
           &skdecide::PyPPDDLDetHindsightSolver::is_solution_defined_for,
           py::arg("state"))
      .def("get_next_action",
           &skdecide::PyPPDDLDetHindsightSolver::get_next_action,
           py::arg("state"))
      .def("get_best_value",
           &skdecide::PyPPDDLDetHindsightSolver::get_best_value,
           py::arg("state"))
      .def("get_nb_steps", &skdecide::PyPPDDLDetHindsightSolver::get_nb_steps)
      .def("get_solving_time",
           &skdecide::PyPPDDLDetHindsightSolver::get_solving_time)
      .def("get_explored_states",
           &skdecide::PyPPDDLDetHindsightSolver::get_explored_states)
      .def("get_terminal_states",
           &skdecide::PyPPDDLDetHindsightSolver::get_terminal_states);

  // FFDetHindsight — FF fixed as inner solver
  py::class_<skdecide::PyFFDetHindsightSolver>(m, "_FFDetHindsightSolver_")
      .def(py::init<py::object &, const skdecide::pddl::Task &, bool,
                    std::size_t, double, std::size_t, double, double,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("task"), py::arg("parallel") = false,
           py::arg("sample_width") = 30, py::arg("dead_end_cost") = 1e9,
           py::arg("max_steps") = 10000, py::arg("discount") = 0.99,
           py::arg("epsilon") = 1e-3, py::arg("callback") = nullptr,
           py::arg("verbose") = false, py::keep_alive<1, 3>())
      .def("solve", &skdecide::PyFFDetHindsightSolver::solve, py::arg("state"))
      .def("clear", &skdecide::PyFFDetHindsightSolver::clear)
      .def("is_solution_defined_for",
           &skdecide::PyFFDetHindsightSolver::is_solution_defined_for,
           py::arg("state"))
      .def("get_next_action",
           &skdecide::PyFFDetHindsightSolver::get_next_action, py::arg("state"))
      .def("get_best_value", &skdecide::PyFFDetHindsightSolver::get_best_value,
           py::arg("state"))
      .def("get_nb_steps", &skdecide::PyFFDetHindsightSolver::get_nb_steps)
      .def("get_solving_time",
           &skdecide::PyFFDetHindsightSolver::get_solving_time)
      .def("get_explored_states",
           &skdecide::PyFFDetHindsightSolver::get_explored_states)
      .def("get_terminal_states",
           &skdecide::PyFFDetHindsightSolver::get_terminal_states);
}
