/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_mdplp.hh"

void init_pymdplp(py::module &m) {
  py::class_<skdecide::PyMDPLPSolver> py_mdplp_solver(m, "_MDPLPSolver_");
  py_mdplp_solver
      .def(py::init<py::object &, py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &)> &,
                    const std::string &, double, double, double, std::size_t,
                    bool, const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("heuristic"),
           py::arg("terminal_value"), py::arg("variant") = "dual",
           py::arg("discount") = 0.99, py::arg("epsilon") = 0.001,
           py::arg("lp_infinity") = 1e20,
           py::arg("lp_callback_interval") = std::size_t(0),
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PyMDPLPSolver::close)
      .def("clear", &skdecide::PyMDPLPSolver::clear)
      .def("solve", &skdecide::PyMDPLPSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyMDPLPSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyMDPLPSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyMDPLPSolver::get_utility,
           py::arg("state"))
      .def("get_nb_states", &skdecide::PyMDPLPSolver::get_nb_states)
      .def("get_nb_lp_variables", &skdecide::PyMDPLPSolver::get_nb_lp_variables)
      .def("get_nb_lp_constraints",
           &skdecide::PyMDPLPSolver::get_nb_lp_constraints)
      .def("get_solving_time", &skdecide::PyMDPLPSolver::get_solving_time)
      .def("get_explored_states", &skdecide::PyMDPLPSolver::get_explored_states)
      .def("get_callback_event", &skdecide::PyMDPLPSolver::get_callback_event);

  // --- SSPLPSolver (undiscounted SSP with goals) ---
  py::class_<skdecide::PySSPLPSolver> py_ssplp_solver(m, "_SSPLPSolver_");
  py_ssplp_solver
      .def(py::init<py::object &, py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::string &, double, double, std::size_t, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"), py::arg("variant") = "dual",
           py::arg("epsilon") = 0.001, py::arg("lp_infinity") = 1e20,
           py::arg("lp_callback_interval") = std::size_t(0),
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PySSPLPSolver::close)
      .def("clear", &skdecide::PySSPLPSolver::clear)
      .def("solve", &skdecide::PySSPLPSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PySSPLPSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PySSPLPSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PySSPLPSolver::get_utility,
           py::arg("state"))
      .def("get_nb_states", &skdecide::PySSPLPSolver::get_nb_states)
      .def("get_nb_lp_variables", &skdecide::PySSPLPSolver::get_nb_lp_variables)
      .def("get_nb_lp_constraints",
           &skdecide::PySSPLPSolver::get_nb_lp_constraints)
      .def("get_solving_time", &skdecide::PySSPLPSolver::get_solving_time)
      .def("get_explored_states", &skdecide::PySSPLPSolver::get_explored_states)
      .def("get_callback_event", &skdecide::PySSPLPSolver::get_callback_event);
}
