/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_bfws.hh"

namespace py = pybind11;

void init_pybfws(py::module &m) {
  py::class_<skdecide::PyBFWSSolver> py_bfws_solver(m, "_BFWSSolver_");
  py_bfws_solver
      .def(py::init<py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    bool, bool, bool>(),
           py::arg("domain"), py::arg("state_features"), py::arg("heuristic"),
           py::arg("termination_checker"),
           py::arg("use_state_feature_hash") = false,
           py::arg("parallel") = false, py::arg("debug_logs") = false)
      .def("close", &skdecide::PyBFWSSolver::close)
      .def("clear", &skdecide::PyBFWSSolver::clear)
      .def("solve", &skdecide::PyBFWSSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyBFWSSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyBFWSSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyBFWSSolver::get_utility,
           py::arg("state"));
}
