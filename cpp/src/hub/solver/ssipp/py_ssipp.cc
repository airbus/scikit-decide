/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_ssipp.hh"

void init_pyssipp(py::module &m) {
  py::class_<skdecide::PySSiPPSolver> py_ssipp_solver(m, "_SSiPPSolver_");
  py_ssipp_solver
      .def(py::init<py::object &, py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    std::size_t, double, double, std::size_t,
                    const std::string &, const py::dict &, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"), py::arg("depth") = 3,
           py::arg("discount") = 1.0, py::arg("epsilon") = 0.001,
           py::arg("max_iterations") = 10000, py::arg("inner_solver") = "LRTDP",
           py::arg("inner_solver_params") = py::dict(),
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PySSiPPSolver::close)
      .def("clear", &skdecide::PySSiPPSolver::clear)
      .def("solve", &skdecide::PySSiPPSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PySSiPPSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PySSiPPSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PySSiPPSolver::get_utility,
           py::arg("state"))
      .def("get_nb_explored_states",
           &skdecide::PySSiPPSolver::get_nb_explored_states)
      .def("get_nb_sub_ssps", &skdecide::PySSiPPSolver::get_nb_sub_ssps)
      .def("get_solving_time", &skdecide::PySSiPPSolver::get_solving_time)
      .def("get_explored_states", &skdecide::PySSiPPSolver::get_explored_states)
      .def("get_current_subssp_states",
           &skdecide::PySSiPPSolver::get_current_subssp_states)
      .def("get_boundary_states", &skdecide::PySSiPPSolver::get_boundary_states)
      .def("get_policy", &skdecide::PySSiPPSolver::get_policy);
}
