/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_ilaostar.hh"

void init_pyilaostar(py::module& m) {
    py::class_<skdecide::PyILAOStarSolver> py_ilaostar_solver(m, "_ILAOStarSolver_");
        py_ilaostar_solver
            .def(py::init<py::object&,
                          const std::function<py::object (const py::object&, const py::object&)>&,
                          const std::function<py::object (const py::object&, const py::object&)>&,
                          double,
                          double,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("goal_checker"),
                 py::arg("heuristic"),
                 py::arg("discount")=1.0,
                 py::arg("epsilon")=0.001,
                 py::arg("parallel")=false,
                 py::arg("debug_logs")=false)
            .def("close", &skdecide::PyILAOStarSolver::close)
            .def("clear", &skdecide::PyILAOStarSolver::clear)
            .def("solve", &skdecide::PyILAOStarSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &skdecide::PyILAOStarSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &skdecide::PyILAOStarSolver::get_next_action, py::arg("state"))
            .def("get_utility", &skdecide::PyILAOStarSolver::get_utility, py::arg("state"))
            .def("get_nb_of_explored_states", &skdecide::PyILAOStarSolver::get_nb_of_explored_states)
            .def("best_solution_graph_size", &skdecide::PyILAOStarSolver::best_solution_graph_size)
            .def("get_policy", &skdecide::PyILAOStarSolver::get_policy)
        ;
}
