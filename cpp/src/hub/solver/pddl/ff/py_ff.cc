/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_ff.hh"

void init_pyff(py::module &m) {
  py::class_<skdecide::PyFFSolver>(m, "_PDDL_FFSolver_")
      .def(py::init<py::object &, const skdecide::pddl::Task &, bool, double,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("task"), py::arg("parallel") = false,
           py::arg("dead_end_cost") = 1e9, py::arg("callback") = nullptr,
           py::arg("verbose") = false, py::keep_alive<1, 3>())
      .def("solve", &skdecide::PyFFSolver::solve, py::arg("state"))
      .def("clear", &skdecide::PyFFSolver::clear)
      .def("is_solution_defined_for",
           &skdecide::PyFFSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyFFSolver::get_next_action,
           py::arg("state"))
      .def("get_plan", &skdecide::PyFFSolver::get_plan)
      .def("get_nb_explored_states",
           &skdecide::PyFFSolver::get_nb_explored_states)
      .def("get_explored_states", &skdecide::PyFFSolver::get_explored_states)
      .def("get_solving_time", &skdecide::PyFFSolver::get_solving_time);
}
