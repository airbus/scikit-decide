/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_ldfs.hh"

void init_pyldfs(py::module &m) {
  py::class_<skdecide::PyLDFSSolver> py_ldfs_solver(m, "_LDFSSolver_");
  py_ldfs_solver
      .def(py::init<py::object &, // Python solver
                    py::object &, // Python domain
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &)> &,
                    double, double, std::size_t, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"), py::arg("terminal_value") = nullptr,
           py::arg("discount") = 1.0, py::arg("epsilon") = 0.001,
           py::arg("max_depth") = 0, py::arg("parallel") = false,
           py::arg("callback") = nullptr, py::arg("verbose") = false)
      .def("close", &skdecide::PyLDFSSolver::close)
      .def("clear", &skdecide::PyLDFSSolver::clear)
      .def("solve", &skdecide::PyLDFSSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyLDFSSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyLDFSSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyLDFSSolver::get_utility,
           py::arg("state"))
      .def("get_nb_explored_states",
           &skdecide::PyLDFSSolver::get_nb_explored_states)
      .def("get_nb_tip_states", &skdecide::PyLDFSSolver::get_nb_tip_states)
      .def("get_solving_time", &skdecide::PyLDFSSolver::get_solving_time)
      .def("get_explored_states", &skdecide::PyLDFSSolver::get_explored_states)
      .def("get_solved_states", &skdecide::PyLDFSSolver::get_solved_states)
      .def("get_strongly_connected_components",
           &skdecide::PyLDFSSolver::get_strongly_connected_components)
      .def("get_policy", &skdecide::PyLDFSSolver::get_policy)
      .def("get_last_trajectory", &skdecide::PyLDFSSolver::get_last_trajectory);

  py::class_<skdecide::PyIDAstarSolver, skdecide::PyLDFSSolver>
      py_idastar_solver(m, "_IDAstarSolver_");
  py_idastar_solver
      .def(py::init<py::object &, py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    std::size_t, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"), py::arg("max_depth") = 0,
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("get_plan", &skdecide::PyIDAstarSolver::get_plan, py::arg("state"));
}
