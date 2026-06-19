/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_sspreplan.hh"

void init_pysspreplan(py::module &m) {
  py::class_<skdecide::PySSPReplanSolver> py_sspreplan_solver(
      m, "_SSPReplanSolver_");
  py_sspreplan_solver
      .def(py::init<py::object &, py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::string &, const std::string &, std::size_t,
                    std::size_t, const py::dict &, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"),
           py::arg("determinization") = "most_probable_outcome",
           py::arg("inner_solver") = "Astar", py::arg("max_replans") = 1000,
           py::arg("max_steps") = 10000,
           py::arg("inner_solver_params") = py::dict(),
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PySSPReplanSolver::close)
      .def("clear", &skdecide::PySSPReplanSolver::clear)
      .def("solve", &skdecide::PySSPReplanSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PySSPReplanSolver::is_solution_defined_for,
           py::arg("state"))
      .def("get_next_action", &skdecide::PySSPReplanSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PySSPReplanSolver::get_utility,
           py::arg("state"))
      .def("get_plan", &skdecide::PySSPReplanSolver::get_plan)
      .def("get_nb_replans", &skdecide::PySSPReplanSolver::get_nb_replans)
      .def("get_nb_steps", &skdecide::PySSPReplanSolver::get_nb_steps)
      .def("get_solving_time", &skdecide::PySSPReplanSolver::get_solving_time)
      .def("get_total_cost", &skdecide::PySSPReplanSolver::get_total_cost);
}
