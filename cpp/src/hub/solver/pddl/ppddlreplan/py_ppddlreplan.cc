/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "py_ppddlreplan.hh"

void init_pyppddlreplan(py::module &m) {
  py::class_<skdecide::PyFFReplanSolver>(m, "_FFReplanSolver_")
      .def(py::init<py::object &, const skdecide::pddl::Task &,
                    const std::string &, bool, double, std::size_t, std::size_t,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("task"),
           py::arg("determinization") = "most_probable_outcome",
           py::arg("parallel") = false, py::arg("dead_end_cost") = 1e9,
           py::arg("max_replans") = 1000, py::arg("max_steps") = 10000,
           py::arg("callback") = nullptr, py::arg("verbose") = false,
           py::keep_alive<1, 3>())
      .def("solve", &skdecide::PyFFReplanSolver::solve, py::arg("state"))
      .def("clear", &skdecide::PyFFReplanSolver::clear)
      .def("is_solution_defined_for",
           &skdecide::PyFFReplanSolver::is_solution_defined_for,
           py::arg("state"))
      .def("get_next_action", &skdecide::PyFFReplanSolver::get_next_action,
           py::arg("state"))
      .def("get_plan", &skdecide::PyFFReplanSolver::get_plan)
      .def("get_nb_replans", &skdecide::PyFFReplanSolver::get_nb_replans)
      .def("get_nb_steps", &skdecide::PyFFReplanSolver::get_nb_steps)
      .def("get_solving_time", &skdecide::PyFFReplanSolver::get_solving_time)
      .def("get_total_cost", &skdecide::PyFFReplanSolver::get_total_cost);

  py::class_<skdecide::PyPPDDLReplanSolver>(m, "_PPDDLReplanSolver_")
      .def(py::init<py::object &, const skdecide::pddl::Task &,
                    const std::string &, const std::string &, bool, double,
                    std::size_t, std::size_t,
                    const std::function<py::bool_(const py::object &)> &, bool,
                    const py::dict &>(),
           py::arg("solver"), py::arg("task"),
           py::arg("inner_solver_name") = "FF",
           py::arg("determinization") = "most_probable_outcome",
           py::arg("parallel") = false, py::arg("dead_end_cost") = 1e9,
           py::arg("max_replans") = 1000, py::arg("max_steps") = 10000,
           py::arg("callback") = nullptr, py::arg("verbose") = false,
           py::arg("inner_solver_params") = py::dict(), py::keep_alive<1, 3>())
      .def("solve", &skdecide::PyPPDDLReplanSolver::solve, py::arg("state"))
      .def("clear", &skdecide::PyPPDDLReplanSolver::clear)
      .def("is_solution_defined_for",
           &skdecide::PyPPDDLReplanSolver::is_solution_defined_for,
           py::arg("state"))
      .def("get_next_action", &skdecide::PyPPDDLReplanSolver::get_next_action,
           py::arg("state"))
      .def("get_plan", &skdecide::PyPPDDLReplanSolver::get_plan)
      .def("get_nb_replans", &skdecide::PyPPDDLReplanSolver::get_nb_replans)
      .def("get_nb_steps", &skdecide::PyPPDDLReplanSolver::get_nb_steps)
      .def("get_solving_time", &skdecide::PyPPDDLReplanSolver::get_solving_time)
      .def("get_total_cost", &skdecide::PyPPDDLReplanSolver::get_total_cost);
}
