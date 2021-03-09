/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_astar.hh"

void init_pyastar(py::module& m) {
    py::class_<skdecide::PyAStarSolver> py_astar_solver(m, "_AStarSolver_");
        py_astar_solver
            .def(py::init<py::object&,
                          const std::function<py::object (const py::object&, const py::object&)>&,
                          const std::function<py::object (const py::object&, const py::object&)>&,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("goal_checker"),
                 py::arg("heuristic"),
                 py::arg("parallel")=false,
                 py::arg("debug_logs")=false)
            .def("clear", &skdecide::PyAStarSolver::clear)
            .def("solve", &skdecide::PyAStarSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &skdecide::PyAStarSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &skdecide::PyAStarSolver::get_next_action, py::arg("state"))
            .def("get_utility", &skdecide::PyAStarSolver::get_utility, py::arg("state"))
        ;
}
