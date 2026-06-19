/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "py_sspdethindsight.hh"

void init_pysspdethindsight(py::module &m) {
  py::class_<skdecide::PySSPDetHindsightSolver> py_solver(
      m, "_SSPDetHindsightSolver_");
  py_solver
      .def(py::init<py::object &, py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::string &, const py::dict &, std::size_t, double,
                    std::size_t, double, double, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"), py::arg("inner_solver") = "Astar",
           py::arg("inner_solver_params") = py::dict(),
           py::arg("sample_width") = 30, py::arg("dead_end_cost") = 1000.0,
           py::arg("max_steps") = 10000, py::arg("discount") = 0.99,
           py::arg("epsilon") = 1e-3, py::arg("parallel") = false,
           py::arg("callback") = nullptr, py::arg("verbose") = false)
      .def("close", &skdecide::PySSPDetHindsightSolver::close)
      .def("clear", &skdecide::PySSPDetHindsightSolver::clear)
      .def("solve", &skdecide::PySSPDetHindsightSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PySSPDetHindsightSolver::is_solution_defined_for,
           py::arg("state"))
      .def("get_next_action",
           &skdecide::PySSPDetHindsightSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PySSPDetHindsightSolver::get_utility,
           py::arg("state"))
      .def("get_nb_steps", &skdecide::PySSPDetHindsightSolver::get_nb_steps)
      .def("get_solving_time",
           &skdecide::PySSPDetHindsightSolver::get_solving_time)
      .def("get_explored_states",
           &skdecide::PySSPDetHindsightSolver::get_explored_states)
      .def("get_terminal_states",
           &skdecide::PySSPDetHindsightSolver::get_terminal_states);
}
