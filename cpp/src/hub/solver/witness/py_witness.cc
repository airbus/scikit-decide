/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_witness.hh"

void init_pywitness(py::module &m) {
  py::class_<skdecide::PyWitnessSolver> py_witness_solver(m, "_WitnessSolver_");
  py_witness_solver
      .def(py::init<py::object &, py::object &, double, double, std::size_t,
                    double, double, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("epsilon") = 0.001,
           py::arg("discount") = 0.95, py::arg("max_iterations") = 100,
           py::arg("lp_infinity") = 1e20, py::arg("lp_tolerance") = 1e-10,
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PyWitnessSolver::close)
      .def("clear", &skdecide::PyWitnessSolver::clear)
      .def("solve", &skdecide::PyWitnessSolver::solve, py::arg("distribution"))
      .def("get_next_action", &skdecide::PyWitnessSolver::get_next_action,
           py::arg("observation"))
      .def("get_utility", &skdecide::PyWitnessSolver::get_utility,
           py::arg("observation"))
      .def("is_solution_defined_for",
           &skdecide::PyWitnessSolver::is_solution_defined_for,
           py::arg("observation"))
      .def("reset_belief", &skdecide::PyWitnessSolver::reset_belief)
      .def("get_next_action_from_belief",
           &skdecide::PyWitnessSolver::get_next_action_from_belief,
           py::arg("belief"))
      .def("get_utility_from_belief",
           &skdecide::PyWitnessSolver::get_utility_from_belief,
           py::arg("belief"))
      .def("is_solution_defined_for_from_belief",
           &skdecide::PyWitnessSolver::is_solution_defined_for_from_belief,
           py::arg("belief"))
      .def("get_nb_alpha_vectors",
           &skdecide::PyWitnessSolver::get_nb_alpha_vectors)
      .def("get_nb_iterations", &skdecide::PyWitnessSolver::get_nb_iterations)
      .def("get_solving_time", &skdecide::PyWitnessSolver::get_solving_time)
      .def("get_callback_event",
           &skdecide::PyWitnessSolver::get_callback_event);
}
