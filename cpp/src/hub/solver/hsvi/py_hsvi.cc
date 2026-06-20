/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "py_hsvi.hh"

void init_pyhsvi(py::module &m) {
  py::class_<skdecide::PyHSVISolver> py_hsvi_solver(m, "_HSVISolver_");
  py_hsvi_solver
      .def(py::init<py::object &, py::object &, double, double, std::size_t,
                    std::size_t, bool, double, std::size_t, double, double,
                    double, bool,
                    const std::function<py::bool_(const py::object &)> &,
                    bool>(),
           py::arg("solver"), py::arg("domain"), py::arg("epsilon") = 0.001,
           py::arg("discount") = 0.95, py::arg("time_budget") = 300000,
           py::arg("max_sample_depth") = 100,
           py::arg("use_closed_list") = false, py::arg("depth_bound_eta") = 0.1,
           py::arg("max_vi_iterations") = 1000,
           py::arg("vi_convergence_factor") = 0.01,
           py::arg("prob_epsilon") = 1e-15,
           py::arg("belief_hash_resolution") = 1000.0,
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PyHSVISolver::close)
      .def("clear", &skdecide::PyHSVISolver::clear)
      .def("solve", &skdecide::PyHSVISolver::solve, py::arg("distribution"))
      .def("get_next_action", &skdecide::PyHSVISolver::get_next_action,
           py::arg("observation"))
      .def("get_utility", &skdecide::PyHSVISolver::get_utility,
           py::arg("observation"))
      .def("is_solution_defined_for",
           &skdecide::PyHSVISolver::is_solution_defined_for,
           py::arg("observation"))
      .def("reset_belief", &skdecide::PyHSVISolver::reset_belief)
      .def("get_next_action_from_belief",
           &skdecide::PyHSVISolver::get_next_action_from_belief,
           py::arg("belief"))
      .def("get_utility_from_belief",
           &skdecide::PyHSVISolver::get_utility_from_belief, py::arg("belief"))
      .def("is_solution_defined_for_from_belief",
           &skdecide::PyHSVISolver::is_solution_defined_for_from_belief,
           py::arg("belief"))
      .def("get_nb_alpha_vectors",
           &skdecide::PyHSVISolver::get_nb_alpha_vectors)
      .def("get_nb_bound_points", &skdecide::PyHSVISolver::get_nb_bound_points)
      .def("get_solving_time", &skdecide::PyHSVISolver::get_solving_time)
      .def("get_gap", &skdecide::PyHSVISolver::get_gap)
      .def("get_alpha_vectors", &skdecide::PyHSVISolver::get_alpha_vectors)
      .def("get_last_trajectory", &skdecide::PyHSVISolver::get_last_trajectory);

  py::class_<skdecide::PyGoalHSVISolver> py_goal_hsvi_solver(
      m, "_GoalHSVISolver_");
  py_goal_hsvi_solver
      .def(py::init<py::object &, py::object &,
                    const std::function<py::object(const py::object &,
                                                   const py::object &)> &,
                    double, double, std::size_t, std::size_t, bool, double,
                    std::size_t, double, double, double, bool,
                    const std::function<py::bool_(const py::object &)> &, bool,
                    std::optional<double>>(),
           py::arg("solver"), py::arg("domain"), py::arg("goal_checker"),
           py::arg("epsilon") = 0.001, py::arg("discount") = 1.0,
           py::arg("time_budget") = 300000, py::arg("max_sample_depth") = 100,
           py::arg("use_closed_list") = true, py::arg("depth_bound_eta") = 0.1,
           py::arg("max_vi_iterations") = 1000,
           py::arg("vi_convergence_factor") = 0.01,
           py::arg("prob_epsilon") = 1e-15,
           py::arg("belief_hash_resolution") = 1000.0,
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false, py::arg("dead_end_cost") = py::none())
      .def("close", &skdecide::PyGoalHSVISolver::close)
      .def("clear", &skdecide::PyGoalHSVISolver::clear)
      .def("solve", &skdecide::PyGoalHSVISolver::solve, py::arg("distribution"))
      .def("get_next_action", &skdecide::PyGoalHSVISolver::get_next_action,
           py::arg("observation"))
      .def("get_utility", &skdecide::PyGoalHSVISolver::get_utility,
           py::arg("observation"))
      .def("is_solution_defined_for",
           &skdecide::PyGoalHSVISolver::is_solution_defined_for,
           py::arg("observation"))
      .def("reset_belief", &skdecide::PyGoalHSVISolver::reset_belief)
      .def("get_next_action_from_belief",
           &skdecide::PyGoalHSVISolver::get_next_action_from_belief,
           py::arg("belief"))
      .def("get_utility_from_belief",
           &skdecide::PyGoalHSVISolver::get_utility_from_belief,
           py::arg("belief"))
      .def("is_solution_defined_for_from_belief",
           &skdecide::PyGoalHSVISolver::is_solution_defined_for_from_belief,
           py::arg("belief"))
      .def("get_nb_alpha_vectors",
           &skdecide::PyGoalHSVISolver::get_nb_alpha_vectors)
      .def("get_nb_bound_points",
           &skdecide::PyGoalHSVISolver::get_nb_bound_points)
      .def("get_solving_time", &skdecide::PyGoalHSVISolver::get_solving_time)
      .def("get_gap", &skdecide::PyGoalHSVISolver::get_gap)
      .def("get_alpha_vectors", &skdecide::PyGoalHSVISolver::get_alpha_vectors)
      .def("get_last_trajectory",
           &skdecide::PyGoalHSVISolver::get_last_trajectory);
}
