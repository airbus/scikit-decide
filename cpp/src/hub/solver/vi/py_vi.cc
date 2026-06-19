/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_vi.hh"

void init_pyvi(py::module &m) {
  py::class_<skdecide::PyVISolver> py_vi_solver(m, "_VISolver_");
  py_vi_solver
      .def(py::init<py::object &, // Python solver
                    py::object &, // Python domain
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    const std::function<py::object(const py::object &)> &,
                    double, double, std::size_t, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("heuristic"),
           py::arg("terminal_value") = nullptr, py::arg("discount") = 0.999,
           py::arg("epsilon") = 0.001, py::arg("max_sweeps") = 0,
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PyVISolver::close)
      .def("clear", &skdecide::PyVISolver::clear)
      .def("solve", &skdecide::PyVISolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyVISolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyVISolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyVISolver::get_utility, py::arg("state"))
      .def("get_nb_explored_states",
           &skdecide::PyVISolver::get_nb_explored_states)
      .def("get_nb_iterations", &skdecide::PyVISolver::get_nb_iterations)
      .def("get_solving_time", &skdecide::PyVISolver::get_solving_time)
      .def("get_explored_states", &skdecide::PyVISolver::get_explored_states)
      .def("get_converged_states", &skdecide::PyVISolver::get_converged_states)
      .def("get_states_updated_in_last_sweep",
           &skdecide::PyVISolver::get_states_updated_in_last_sweep)
      .def("get_policy", &skdecide::PyVISolver::get_policy);
}
