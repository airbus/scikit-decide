/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_pi.hh"

void init_pypi(py::module &m) {
  py::class_<skdecide::PyPISolver> py_pi_solver(m, "_PISolver_");
  py_pi_solver
      .def(py::init<py::object &, // Python solver
                    py::object &, // Python domain
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &)> &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    double, double, std::size_t, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("heuristic"),
           py::arg("terminal_value") = nullptr,
           py::arg("initial_policy") = nullptr, py::arg("discount") = 0.999,
           py::arg("epsilon") = 0.001, py::arg("max_eval_sweeps") = 0,
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PyPISolver::close)
      .def("clear", &skdecide::PyPISolver::clear)
      .def("solve", &skdecide::PyPISolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyPISolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyPISolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyPISolver::get_utility, py::arg("state"))
      .def("get_nb_explored_states",
           &skdecide::PyPISolver::get_nb_explored_states)
      .def("get_nb_iterations", &skdecide::PyPISolver::get_nb_iterations)
      .def("get_solving_time", &skdecide::PyPISolver::get_solving_time)
      .def("get_explored_states", &skdecide::PyPISolver::get_explored_states)
      .def("get_policy_changed_states",
           &skdecide::PyPISolver::get_policy_changed_states)
      .def("get_policy", &skdecide::PyPISolver::get_policy);
}
