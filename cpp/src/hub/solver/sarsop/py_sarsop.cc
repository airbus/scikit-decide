/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_sarsop.hh"

namespace py = pybind11;

void init_pysarsop(py::module &m) {
  py::class_<skdecide::PySARSOPSolver> py_sarsop_solver(m, "_SARSOPSolver_");
  py_sarsop_solver
      .def(py::init<py::object &, py::object &, double, double, std::size_t,
                    std::size_t, double, std::size_t, double, std::size_t,
                    double, double, std::size_t, std::size_t,
                    const std::function<py::object(const py::object &)> &, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("epsilon") = 0.001,
           py::arg("discount") = 0.95, py::arg("time_budget") = 300000,
           py::arg("max_beliefs") = 100000, py::arg("pruning_delta") = 1e-6,
           py::arg("max_vi_iterations") = 1000,
           py::arg("vi_convergence_factor") = 0.01,
           py::arg("max_sample_depth") = 100, py::arg("prob_epsilon") = 1e-15,
           py::arg("ub_improvement_epsilon") = 1e-10,
           py::arg("pruning_interval") = 10, py::arg("logging_interval") = 50,
           py::arg("terminal_value") = nullptr, py::arg("parallel") = false,
           py::arg("callback") = nullptr, py::arg("verbose") = false)
      .def("close", &skdecide::PySARSOPSolver::close)
      .def("clear", &skdecide::PySARSOPSolver::clear)
      .def("solve", &skdecide::PySARSOPSolver::solve, py::arg("distribution"))
      .def("get_next_action", &skdecide::PySARSOPSolver::get_next_action,
           py::arg("observation"))
      .def("get_utility", &skdecide::PySARSOPSolver::get_utility,
           py::arg("observation"))
      .def("is_solution_defined_for",
           &skdecide::PySARSOPSolver::is_solution_defined_for,
           py::arg("observation"))
      .def("reset_belief", &skdecide::PySARSOPSolver::reset_belief)
      .def("get_next_action_from_belief",
           &skdecide::PySARSOPSolver::get_next_action_from_belief,
           py::arg("belief"))
      .def("get_utility_from_belief",
           &skdecide::PySARSOPSolver::get_utility_from_belief,
           py::arg("belief"))
      .def("is_solution_defined_for_from_belief",
           &skdecide::PySARSOPSolver::is_solution_defined_for_from_belief,
           py::arg("belief"))
      .def("get_nb_alpha_vectors",
           &skdecide::PySARSOPSolver::get_nb_alpha_vectors)
      .def("get_nb_explored_beliefs",
           &skdecide::PySARSOPSolver::get_nb_explored_beliefs)
      .def("get_solving_time", &skdecide::PySARSOPSolver::get_solving_time)
      .def("get_lower_bound", &skdecide::PySARSOPSolver::get_lower_bound)
      .def("get_upper_bound", &skdecide::PySARSOPSolver::get_upper_bound)
      .def("get_gap", &skdecide::PySARSOPSolver::get_gap)
      .def("get_alpha_vectors", &skdecide::PySARSOPSolver::get_alpha_vectors)
      .def("get_last_trajectory",
           &skdecide::PySARSOPSolver::get_last_trajectory);
}
