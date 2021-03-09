/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_riw.hh"

void init_pyriw(py::module& m) {
    py::class_<skdecide::PyRIWSolver> py_riw_solver(m, "_RIWSolver_");
        py_riw_solver
            .def(py::init<py::object&,
                          const std::function<py::object (py::object&, const py::object&, const py::object&)>&,  // last arg used for optional thread_id
                          bool,
                          bool,
                          std::size_t,
                          std::size_t,
                          std::size_t,
                          double,
                          std::size_t,
                          double,
                          double,
                          bool,
                          bool,
                          bool,
                          const std::function<bool (const py::int_&, const py::int_&, const py::float_&, const py::float_&)>&>(),
                 py::arg("domain"),
                 py::arg("state_features"),
                 py::arg("use_state_feature_hash")=false,
                 py::arg("use_simulation_domain")=false,
                 py::arg("time_budget")=3600000,
                 py::arg("rollout_budget")=100000,
                 py::arg("max_depth")=1000,
                 py::arg("exploration")=0.25,
                 py::arg("epsilon_moving_average_window")=100,
                 py::arg("epsilon")=0.001,
                 py::arg("discount")=1.0,
                 py::arg("online_node_garbage")=false,
                 py::arg("parallel")=false,
                 py::arg("debug_logs")=false,
                 py::arg("watchdog")=nullptr)
            .def("clear", &skdecide::PyRIWSolver::clear)
            .def("solve", &skdecide::PyRIWSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &skdecide::PyRIWSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &skdecide::PyRIWSolver::get_next_action, py::arg("state"))
            .def("get_utility", &skdecide::PyRIWSolver::get_utility, py::arg("state"))
            .def("get_nb_of_explored_states", &skdecide::PyRIWSolver::get_nb_of_explored_states)
            .def("get_nb_of_pruned_states", &skdecide::PyRIWSolver::get_nb_of_pruned_states)
            .def("get_nb_rollouts", &skdecide::PyRIWSolver::get_nb_rollouts)
            .def("get_policy", &skdecide::PyRIWSolver::get_policy)
            .def("get_action_prefix", &skdecide::PyRIWSolver::get_action_prefix)
        ;
}
