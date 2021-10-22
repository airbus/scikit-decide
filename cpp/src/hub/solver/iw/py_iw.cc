/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_iw.hh"

namespace py = pybind11;

void init_pyiw(py::module &m) {
  py::class_<skdecide::PyIWSolver> py_iw_solver(m, "_IWSolver_");
  py_iw_solver
      .def(py::init<
               py::object &,
               const std::function<py::object(const py::object &,
                                              const py::object &)> &,
               bool,
               const std::function<bool(
                   const double &, const std::size_t &, const std::size_t &,
                   const double &, const std::size_t &, const std::size_t &)> &,
               std::size_t, bool, bool>(),
           py::arg("domain"), py::arg("state_features"),
           py::arg("use_state_feature_hash") = false,
           py::arg("node_ordering") = nullptr, py::arg("time_budget") = 0,
           py::arg("parallel") = false, py::arg("debug_logs") = false)
      .def("close", &skdecide::PyIWSolver::close)
      .def("clear", &skdecide::PyIWSolver::clear)
      .def("solve", &skdecide::PyIWSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyIWSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyIWSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyIWSolver::get_utility, py::arg("state"))
      .def("get_nb_of_explored_states",
           &skdecide::PyIWSolver::get_nb_of_explored_states)
      .def("get_nb_of_pruned_states",
           &skdecide::PyIWSolver::get_nb_of_pruned_states)
      .def("get_intermediate_scores",
           &skdecide::PyIWSolver::get_intermediate_scores);
}
