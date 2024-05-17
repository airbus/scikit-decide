/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "py_mcts.hh"

namespace skdecide {

PyMCTSSolver::PyMCTSSolver(
    py::object &solver, py::object &domain, std::size_t time_budget,
    std::size_t rollout_budget, std::size_t max_depth,
    std::size_t residual_moving_average_window, double epsilon, double discount,
    double ucb_constant, bool online_node_garbage,
    const CustomPolicyFunctor &custom_policy, const HeuristicFunctor &heuristic,
    double state_expansion_rate, double action_expansion_rate,
    PyMCTSOptions::TransitionMode transition_mode,
    PyMCTSOptions::TreePolicy tree_policy, PyMCTSOptions::Expander expander,
    PyMCTSOptions::ActionSelector action_selector_optimization,
    PyMCTSOptions::ActionSelector action_selector_execution,
    PyMCTSOptions::RolloutPolicy rollout_policy,
    PyMCTSOptions::BackPropagator back_propagator, bool parallel,
    const CallbackFunctor &callback, bool verbose)
    : _filtered_custom_policy(custom_policy) {

  TemplateInstantiator::select(
      ExecutionSelector(parallel),
      TransitionModeSelector(transition_mode, domain),
      TreePolicySelector(tree_policy), ExpanderSelector(expander),
      PartialSolverInstantiator(_implementation, action_selector_optimization,
                                action_selector_execution, rollout_policy,
                                back_propagator, _filtered_custom_policy,
                                heuristic))
      .instantiate(solver, domain, time_budget, rollout_budget, max_depth,
                   residual_moving_average_window, epsilon, discount,
                   ucb_constant, online_node_garbage, _filtered_custom_policy,
                   heuristic, state_expansion_rate, action_expansion_rate,
                   callback, verbose);
}

} // namespace skdecide

void init_pymcts(py::module &m) {
  py::class_<skdecide::PyMCTSOptions> py_mcts_options(m, "_MCTSOptions_");

  py::enum_<skdecide::PyMCTSOptions::TransitionMode>(py_mcts_options,
                                                     "TransitionMode")
      .value("Step", skdecide::PyMCTSOptions::TransitionMode::Step)
      .value("Sample", skdecide::PyMCTSOptions::TransitionMode::Sample)
      .value("Distribution",
             skdecide::PyMCTSOptions::TransitionMode::Distribution);

  py::enum_<skdecide::PyMCTSOptions::TreePolicy>(py_mcts_options, "TreePolicy")
      .value("Default", skdecide::PyMCTSOptions::TreePolicy::Default);

  py::enum_<skdecide::PyMCTSOptions::Expander>(py_mcts_options, "Expander")
      .value("Full", skdecide::PyMCTSOptions::Expander::Full)
      .value("Partial", skdecide::PyMCTSOptions::Expander::Partial);

  py::enum_<skdecide::PyMCTSOptions::ActionSelector>(py_mcts_options,
                                                     "ActionSelector")
      .value("UCB1", skdecide::PyMCTSOptions::ActionSelector::UCB1)
      .value("BestQValue", skdecide::PyMCTSOptions::ActionSelector::BestQValue);

  py::enum_<skdecide::PyMCTSOptions::RolloutPolicy>(py_mcts_options,
                                                    "RolloutPolicy")
      .value("Random", skdecide::PyMCTSOptions::RolloutPolicy::Random)
      .value("Custom", skdecide::PyMCTSOptions::RolloutPolicy::Custom)
      .value("Void", skdecide::PyMCTSOptions::RolloutPolicy::Void);

  py::enum_<skdecide::PyMCTSOptions::BackPropagator>(py_mcts_options,
                                                     "BackPropagator")
      .value("Graph", skdecide::PyMCTSOptions::BackPropagator::Graph);

  py::class_<skdecide::PyMCTSSolver> py_mcts_solver(m, "_MCTSSolver_");
  py_mcts_solver
      .def(py::init<
               py::object &, // Python solver
               py::object &, // Python domain
               std::size_t, std::size_t, std::size_t, std::size_t, double,
               double, double, bool,
               const std::function<py::object(
                   const py::object &, const py::object &,
                   const py::object & // last arg used for
                                      // optional thread_id
                   )> &,
               const std::function<py::object(
                   const py::object &, const py::object &,
                   const py::object & // last arg used for optional thread_id
                   )> &,
               double, double, skdecide::PyMCTSOptions::TransitionMode,
               skdecide::PyMCTSOptions::TreePolicy,
               skdecide::PyMCTSOptions::Expander,
               skdecide::PyMCTSOptions::ActionSelector,
               skdecide::PyMCTSOptions::ActionSelector,
               skdecide::PyMCTSOptions::RolloutPolicy,
               skdecide::PyMCTSOptions::BackPropagator, bool,
               const std::function<py::bool_(
                   const py::object &,
                   const py::object &)> // last arg used for optional thread_id
                   &,
               bool>(),
           py::arg("solver"), py::arg("domain"),
           py::arg("time_budget") = 3600000, py::arg("rollout_budget") = 100000,
           py::arg("max_depth") = 1000,
           py::arg("residual_moving_average_window") = 100,
           py::arg("epsilon") = 0.0, // not a stopping criterion by default
           py::arg("discount") = 1.0,
           py::arg("ucb_constant") = 1.0 / std::sqrt(2.0),
           py::arg("online_node_garbage") = false,
           py::arg("custom_policy") = nullptr, py::arg("heuristic") = nullptr,
           py::arg("state_expansion_rate") = 0.1,
           py::arg("action_expansion_rate") = 0.1,
           py::arg("transition_mode") =
               skdecide::PyMCTSOptions::TransitionMode::Distribution,
           py::arg("tree_policy") =
               skdecide::PyMCTSOptions::TreePolicy::Default,
           py::arg("expander") = skdecide::PyMCTSOptions::Expander::Full,
           py::arg("action_selector_optimization") =
               skdecide::PyMCTSOptions::ActionSelector::UCB1,
           py::arg("action_selector_execution") =
               skdecide::PyMCTSOptions::ActionSelector::BestQValue,
           py::arg("rollout_policy") =
               skdecide::PyMCTSOptions::RolloutPolicy::Random,
           py::arg("back_propagator") =
               skdecide::PyMCTSOptions::BackPropagator::Graph,
           py::arg("parallel") = false, py::arg("callback") = nullptr,
           py::arg("verbose") = false)
      .def("close", &skdecide::PyMCTSSolver::close)
      .def("clear", &skdecide::PyMCTSSolver::clear)
      .def("solve", &skdecide::PyMCTSSolver::solve, py::arg("state"))
      .def("is_solution_defined_for",
           &skdecide::PyMCTSSolver::is_solution_defined_for, py::arg("state"))
      .def("get_next_action", &skdecide::PyMCTSSolver::get_next_action,
           py::arg("state"))
      .def("get_utility", &skdecide::PyMCTSSolver::get_utility,
           py::arg("state"))
      .def("get_nb_explored_states",
           &skdecide::PyMCTSSolver::get_nb_explored_states)
      .def("get_nb_rollouts", &skdecide::PyMCTSSolver::get_nb_rollouts)
      .def("get_residual_moving_average",
           &skdecide::PyMCTSSolver::get_residual_moving_average)
      .def("get_solving_time", &skdecide::PyMCTSSolver::get_solving_time)
      .def("get_policy", &skdecide::PyMCTSSolver::get_policy)
      .def("get_action_prefix", &skdecide::PyMCTSSolver::get_action_prefix);
}
