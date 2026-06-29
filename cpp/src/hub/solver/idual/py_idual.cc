/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "py_idual.hh"

void init_pyidual(py::module &m) {
  // --- IDualSolver (unconstrained SSP, deterministic policy) ---
  py::class_<skdecide::PyIDualSolver> py_idual_solver(m, "_IDualSolver_");
  py_idual_solver
      .def(py::init<py::object &, py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &)> &,
                    double, double, double, std::size_t, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"), py::arg("terminal_value"),
           py::arg("lp_infinity") = 1e20, py::arg("lp_tolerance") = 1e-15,
           py::arg("default_dead_end_cost") = 1000.0,
           py::arg("lp_callback_interval") = std::size_t(0),
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PyIDualSolver::close)
      .def("clear", &skdecide::PyIDualSolver::clear)
      .def("solve", &skdecide::PyIDualSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyIDualSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyIDualSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyIDualSolver::get_utility,
           py::arg("state"))
      .def("get_nb_explored_states",
           &skdecide::PyIDualSolver::get_nb_explored_states)
      .def("get_nb_lp_iterations",
           &skdecide::PyIDualSolver::get_nb_lp_iterations)
      .def("get_solving_time", &skdecide::PyIDualSolver::get_solving_time)
      .def("get_explored_states", &skdecide::PyIDualSolver::get_explored_states)
      .def("get_callback_event", &skdecide::PyIDualSolver::get_callback_event)
      .def("get_policy", &skdecide::PyIDualSolver::get_policy);

  // --- CIDualSolver (constrained SSP, stochastic policy) ---
  py::class_<skdecide::PyCIDualSolver> py_cidual_solver(m, "_CIDualSolver_");
  py_cidual_solver
      .def(py::init<py::object &, py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &, int)> &,
                    py::list, double, double, double, std::size_t, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"), py::arg("terminal_value"),
           py::arg("secondary_heuristic"), py::arg("dead_end_costs"),
           py::arg("lp_infinity") = 1e20, py::arg("lp_tolerance") = 1e-15,
           py::arg("default_dead_end_cost") = 1000.0,
           py::arg("lp_callback_interval") = std::size_t(0),
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PyCIDualSolver::close)
      .def("clear", &skdecide::PyCIDualSolver::clear)
      .def("solve", &skdecide::PyCIDualSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyCIDualSolver::is_solution_defined_for, py::arg("state"))
      .def("get_action_distribution",
           &skdecide::PyCIDualSolver::get_action_distribution, py::arg("state"))
      .def("get_utility", &skdecide::PyCIDualSolver::get_utility,
           py::arg("state"))
      .def("get_nb_explored_states",
           &skdecide::PyCIDualSolver::get_nb_explored_states)
      .def("get_nb_lp_iterations",
           &skdecide::PyCIDualSolver::get_nb_lp_iterations)
      .def("get_solving_time", &skdecide::PyCIDualSolver::get_solving_time)
      .def("get_explored_states",
           &skdecide::PyCIDualSolver::get_explored_states)
      .def("get_callback_event", &skdecide::PyCIDualSolver::get_callback_event)
      .def("get_policy", &skdecide::PyCIDualSolver::get_policy);
}
