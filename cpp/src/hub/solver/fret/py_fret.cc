/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_fret.hh"

void init_pyfret(py::module &m) {
  py::class_<skdecide::PyFRETSolver> py_fret_solver(m, "_FRETSolver_");
  py_fret_solver
      .def(py::init<py::object &, py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    double, double, double, const std::string &,
                    const py::dict &, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"), py::arg("discount") = 1.0,
           py::arg("epsilon") = 0.001, py::arg("dead_end_cost") = 10000.0,
           py::arg("inner_solver") = "LRTDP",
           py::arg("inner_solver_params") = py::dict(),
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PyFRETSolver::close)
      .def("clear", &skdecide::PyFRETSolver::clear)
      .def("solve", &skdecide::PyFRETSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyFRETSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyFRETSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyFRETSolver::get_utility,
           py::arg("state"))
      .def("get_nb_explored_states",
           &skdecide::PyFRETSolver::get_nb_explored_states)
      .def("get_nb_fret_iterations",
           &skdecide::PyFRETSolver::get_nb_fret_iterations)
      .def("get_nb_traps_eliminated",
           &skdecide::PyFRETSolver::get_nb_traps_eliminated)
      .def("get_solving_time", &skdecide::PyFRETSolver::get_solving_time)
      .def("get_explored_states", &skdecide::PyFRETSolver::get_explored_states)
      .def("get_dead_end_states", &skdecide::PyFRETSolver::get_dead_end_states)
      .def("get_trapped_sccs", &skdecide::PyFRETSolver::get_trapped_sccs)
      .def("get_policy", &skdecide::PyFRETSolver::get_policy);
}
