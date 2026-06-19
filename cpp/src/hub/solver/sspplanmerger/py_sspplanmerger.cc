/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "py_sspplanmerger.hh"

void init_pysspplanmerger(py::module &m) {
  py::class_<skdecide::PySSPPlanMergerSolver>(m, "_SSPPlanMergerSolver_")
      .def(py::init<py::object &, py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::string &, const std::string &, const py::dict &,
                    double, std::size_t, std::size_t, std::size_t, double, bool,
                    double, double, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("heuristic"),
           py::arg("determinization") = "most_probable_outcome",
           py::arg("inner_solver") = "Astar",
           py::arg("inner_solver_params") = py::dict(), py::arg("rho") = 0.1,
           py::arg("mc_samples") = 100, py::arg("max_iterations") = 50,
           py::arg("max_steps") = 10000, py::arg("dead_end_cost") = 1e9,
           py::arg("optimize_policy_graph") = false, py::arg("discount") = 0.99,
           py::arg("epsilon") = 1e-3, py::arg("parallel") = false,
           py::arg("callback") = nullptr, py::arg("verbose") = false)
      .def("close", &skdecide::PySSPPlanMergerSolver::close)
      .def("clear", &skdecide::PySSPPlanMergerSolver::clear)
      .def("solve", &skdecide::PySSPPlanMergerSolver::solve, py::arg("state"))
      .def("resolve", &skdecide::PySSPPlanMergerSolver::resolve,
           py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PySSPPlanMergerSolver::is_solution_defined_for,
           py::arg("state"))
      .def("get_next_action", &skdecide::PySSPPlanMergerSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PySSPPlanMergerSolver::get_utility,
           py::arg("state"))
      .def("get_nb_iterations",
           &skdecide::PySSPPlanMergerSolver::get_nb_iterations)
      .def("get_nb_plans", &skdecide::PySSPPlanMergerSolver::get_nb_plans)
      .def("get_solving_time",
           &skdecide::PySSPPlanMergerSolver::get_solving_time)
      .def("get_policy_size", &skdecide::PySSPPlanMergerSolver::get_policy_size)
      .def("get_explored_states",
           &skdecide::PySSPPlanMergerSolver::get_explored_states)
      .def("get_terminal_states",
           &skdecide::PySSPPlanMergerSolver::get_terminal_states)
      .def("get_policy", &skdecide::PySSPPlanMergerSolver::get_policy);
}
