/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "mcts.hh"
#include "core.hh"

#include "utils/python_gil_control.hh"
#include "utils/python_hash_eq.hh"
#include "utils/python_domain_adapter.hh"

namespace py = pybind11;


template <typename Texecution>
class PyMCTSDomain : public airlaps::PythonDomainAdapter<Texecution> {
public :

    PyMCTSDomain(const py::object& domain)
    : airlaps::PythonDomainAdapter<Texecution>(domain) {
        if (!py::hasattr(domain, "get_applicable_actions")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing get_applicable_actions()");
        }
        if (!py::hasattr(domain, "get_next_state_distribution")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing get_next_state_distribution()");
        }
        if (!py::hasattr(domain, "sample")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing sample()");
        }
        if (!py::hasattr(domain, "get_transition_value")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing get_transition_value()");
        }
        if (!py::hasattr(domain, "is_terminal")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing is_terminal()");
        }
    }

};


struct PyMCTSOptions {
    enum class TreePolicy {
        Default
    };

    enum class Expander {
        Full
    };

    enum class ActionSelector {
        UCB1,
        BestQValue
    };

    enum class DefaultPolicy {
        Random
    };

    enum class BackPropagator {
        Graph
    };
};


class PyMCTSSolver {
public :

    PyMCTSSolver(py::object& domain,
                 std::size_t time_budget = 3600000,
                 std::size_t rollout_budget = 100000,
                 std::size_t max_depth = 1000,
                 double discount = 1.0,
                 bool uct_mode = true,
                 double ucb_constant = 1.0 / std::sqrt(2.0),
                 PyMCTSOptions::TreePolicy tree_policy = PyMCTSOptions::TreePolicy::Default,
                 PyMCTSOptions::Expander expander = PyMCTSOptions::Expander::Full,
                 PyMCTSOptions::ActionSelector action_selector_optimization = PyMCTSOptions::ActionSelector::UCB1,
                 PyMCTSOptions::ActionSelector action_selector_execution = PyMCTSOptions::ActionSelector::BestQValue,
                 PyMCTSOptions::DefaultPolicy default_policy = PyMCTSOptions::DefaultPolicy::Random,
                 PyMCTSOptions::BackPropagator back_propagator = PyMCTSOptions::BackPropagator::Graph,
                 bool parallel = true,
                 bool debug_logs = false) {
        if (uct_mode) {
            initialize_tree_policy(domain,
                                   time_budget,
                                   rollout_budget,
                                   max_depth,
                                   discount,
                                   ucb_constant,
                                   PyMCTSOptions::TreePolicy::Default,
                                   expander,
                                   PyMCTSOptions::ActionSelector::UCB1,
                                   action_selector_execution,
                                   PyMCTSOptions::DefaultPolicy::Random,
                                   PyMCTSOptions::BackPropagator::Graph,
                                   parallel,
                                   debug_logs);
        } else {
            spdlog::error("MCTS only supports UCT mode at the moment.");
            throw std::runtime_error("MCTS only supports UCT mode at the moment.");
        }
    }

    void clear() {
        _implementation->clear();
    }

    void solve(const py::object& s) {
        _implementation->solve(s);
    }

    py::bool_ is_solution_defined_for(const py::object& s) {
        return _implementation->is_solution_defined_for(s);
    }

    py::object get_next_action(const py::object& s) {
        return _implementation->get_next_action(s);
    }

    py::float_ get_utility(const py::object& s) {
        return _implementation->get_utility(s);
    }

    virtual py::int_ get_nb_of_explored_states() {
        return _implementation->get_nb_of_explored_states();
    }

    virtual py::int_ get_nb_rollouts() {
        return _implementation->get_nb_rollouts();
    }

private :

    class BaseImplementation {
    public :
        virtual void clear() =0;
        virtual void solve(const py::object& s) =0;
        virtual py::bool_ is_solution_defined_for(const py::object& s) =0;
        virtual py::object get_next_action(const py::object& s) =0;
        virtual py::float_ get_utility(const py::object& s) =0;
        virtual py::int_ get_nb_of_explored_states() =0;
        virtual py::int_ get_nb_rollouts() =0;
    };

    template <typename Texecution,
              typename TtreePolicy,
              typename Texpander,
              typename TactionSelectorOptimization,
              typename TactionSelectorExecution,
              typename TdefaultPolicy,
              typename TbackPropagator>
    class Implementation : public BaseImplementation {
    public :
        Implementation(py::object& domain,
                       std::size_t time_budget = 3600000,
                       std::size_t rollout_budget = 100000,
                       std::size_t max_depth = 1000,
                       double discount = 1.0,
                       double ucb_constant = 1.0 / std::sqrt(2.0),
                       bool debug_logs = false) {

            _domain = std::make_unique<PyMCTSDomain<Texecution>>(domain);
            _solver = std::make_unique<airlaps::MCTSSolver<PyMCTSDomain<Texecution>,
                                                           Texecution,
                                                           TtreePolicy,
                                                           Texpander,
                                                           TactionSelectorOptimization,
                                                           TactionSelectorExecution,
                                                           TdefaultPolicy,
                                                           TbackPropagator>>(
                        *_domain,
                        time_budget,
                        rollout_budget,
                        max_depth,
                        discount,
                        debug_logs,
                        init_tree_policy(),
                        init_expander(),
                        init_action_selector((TactionSelectorOptimization*) nullptr, ucb_constant),
                        init_action_selector((TactionSelectorExecution*) nullptr, ucb_constant),
                        init_default_policy(),
                        init_back_propagator());
            _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(std::cout,
                                                                            py::module::import("sys").attr("stdout"));
            _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(std::cerr,
                                                                            py::module::import("sys").attr("stderr"));
        }

        TtreePolicy init_tree_policy() {
            return TtreePolicy();
        }

        Texpander init_expander() {
            return Texpander();
        }

        template <typename TactionSelector,
                  std::enable_if_t<std::is_same<TactionSelector, airlaps::UCB1ActionSelector>::value, int> = 0>
        TactionSelector init_action_selector(TactionSelector* dummy, double ucb_constant) {
            return airlaps::UCB1ActionSelector(ucb_constant);
        }

        template <typename TactionSelector,
                  std::enable_if_t<std::is_same<TactionSelector, airlaps::BestQValueActionSelector>::value, int> = 0>
        TactionSelector init_action_selector(TactionSelector* dummy, double ucb_constant) {
            return airlaps::BestQValueActionSelector();
        }

        TdefaultPolicy init_default_policy() {
            return TdefaultPolicy();
        }

        TbackPropagator init_back_propagator() {
            return TbackPropagator();
        }

        virtual void clear() {
            _solver->clear();
        }

        virtual void solve(const py::object& s) {
            typename airlaps::GilControl<Texecution>::Release release;
            _solver->solve(s);
        }

        virtual py::bool_ is_solution_defined_for(const py::object& s) {
            return _solver->is_solution_defined_for(s);
        }

        virtual py::object get_next_action(const py::object& s) {
            return _solver->get_best_action(s).get();
        }

        virtual py::float_ get_utility(const py::object& s) {
            return _solver->get_best_value(s);
        }

        virtual py::int_ get_nb_of_explored_states() {
            return _solver->nb_of_explored_states();
        }

        virtual py::int_ get_nb_rollouts() {
            return _solver->nb_rollouts();
        }

    private :
        std::unique_ptr<PyMCTSDomain<Texecution>> _domain;
        std::unique_ptr<airlaps::MCTSSolver<PyMCTSDomain<Texecution>,
                                            Texecution,
                                            TtreePolicy,
                                            Texpander,
                                            TactionSelectorOptimization,
                                            TactionSelectorExecution,
                                            TdefaultPolicy,
                                            TbackPropagator>> _solver;

        std::function<bool (const py::object&)> _goal_checker;
        std::function<double (const py::object&)> _heuristic;

        std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
        std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
    };

    std::unique_ptr<BaseImplementation> _implementation;

    void initialize_tree_policy(
                py::object& domain,
                std::size_t time_budget,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                PyMCTSOptions::TreePolicy tree_policy,
                PyMCTSOptions::Expander expander,
                PyMCTSOptions::ActionSelector action_selector_optimization,
                PyMCTSOptions::ActionSelector action_selector_execution,
                PyMCTSOptions::DefaultPolicy default_policy,
                PyMCTSOptions::BackPropagator back_propagator,
                bool parallel,
                bool debug_logs) {
        switch (tree_policy) {
            case PyMCTSOptions::TreePolicy::Default:
                initialize_expander<airlaps::DefaultTreePolicy>(
                                        domain,
                                        time_budget,
                                        rollout_budget,
                                        max_depth,
                                        discount,
                                        ucb_constant,
                                        expander,
                                        action_selector_optimization,
                                        action_selector_execution,
                                        default_policy,
                                        back_propagator,
                                        parallel,
                                        debug_logs);
                break;
            
            default:
                spdlog::error("Available tree policies: TreePolicy.Default");
                throw std::runtime_error("Available tree policies: TreePolicy.Default");
        }
    }

    template <typename TtreePolicy>
    void initialize_expander(
                py::object& domain,
                std::size_t time_budget,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                PyMCTSOptions::Expander expander,
                PyMCTSOptions::ActionSelector action_selector_optimization,
                PyMCTSOptions::ActionSelector action_selector_execution,
                PyMCTSOptions::DefaultPolicy default_policy,
                PyMCTSOptions::BackPropagator back_propagator,
                bool parallel,
                bool debug_logs) {
        switch (expander) {
            case PyMCTSOptions::Expander::Full:
                initialize_action_selector_optimization<TtreePolicy,
                                                        airlaps::FullExpand>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                action_selector_optimization,
                                action_selector_execution,
                                default_policy,
                                back_propagator,
                                parallel,
                                debug_logs);
                break;
            
            default:
                spdlog::error("Available expanders: Expander.Full");
                throw std::runtime_error("Available expanders: Expander.Full");
        }
    }

    template <typename TtreePolicy,
              typename Texpander>
    void initialize_action_selector_optimization(
                py::object& domain,
                std::size_t time_budget,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                PyMCTSOptions::ActionSelector action_selector_optimization,
                PyMCTSOptions::ActionSelector action_selector_execution,
                PyMCTSOptions::DefaultPolicy default_policy,
                PyMCTSOptions::BackPropagator back_propagator,
                bool parallel,
                bool debug_logs) {
        switch (action_selector_optimization) {
            case PyMCTSOptions::ActionSelector::UCB1:
                initialize_action_selector_execution<TtreePolicy,
                                                     Texpander,
                                                     airlaps::UCB1ActionSelector>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                action_selector_execution,
                                default_policy,
                                back_propagator,
                                parallel,
                                debug_logs);
                break;
            
            case PyMCTSOptions::ActionSelector::BestQValue:
                initialize_action_selector_execution<TtreePolicy,
                                                     Texpander,
                                                     airlaps::BestQValueActionSelector>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                action_selector_execution,
                                default_policy,
                                back_propagator,
                                parallel,
                                debug_logs);
                break;
            
            default:
                spdlog::error("Available action selector: ActionSelector.UCB1 , ActionSelector.BestQValue");
                throw std::runtime_error("Available action selector: ActionSelector.UCB1 , ActionSelector.BestQValue");
        }
    }

    template <typename TtreePolicy,
              typename Texpander,
              typename TactionSelectorOptimization>
    void initialize_action_selector_execution(
                py::object& domain,
                std::size_t time_budget,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                PyMCTSOptions::ActionSelector action_selector_execution,
                PyMCTSOptions::DefaultPolicy default_policy,
                PyMCTSOptions::BackPropagator back_propagator,
                bool parallel,
                bool debug_logs) {
        switch (action_selector_execution) {
            case PyMCTSOptions::ActionSelector::UCB1:
                initialize_default_policy<TtreePolicy,
                                          Texpander,
                                          TactionSelectorOptimization,
                                          airlaps::UCB1ActionSelector>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                default_policy,
                                back_propagator,
                                parallel,
                                debug_logs);
                break;
            
            case PyMCTSOptions::ActionSelector::BestQValue:
                initialize_default_policy<TtreePolicy,
                                          Texpander,
                                          TactionSelectorOptimization,
                                          airlaps::BestQValueActionSelector>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                default_policy,
                                back_propagator,
                                parallel,
                                debug_logs);
                break;
            
            default:
                spdlog::error("Available action selector: ActionSelector.UCB1 , ActionSelector.BestQValue");
                throw std::runtime_error("Available action selector: ActionSelector.UCB1 , ActionSelector.BestQValue");
        }
    }

    template <typename TtreePolicy,
              typename Texpander,
              typename TactionSelectorOptimization,
              typename TactionSelectorExecution>
    void initialize_default_policy(
                py::object& domain,
                std::size_t time_budget,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                PyMCTSOptions::DefaultPolicy default_policy,
                PyMCTSOptions::BackPropagator back_propagator,
                bool parallel,
                bool debug_logs) {
        switch (default_policy) {
            case PyMCTSOptions::DefaultPolicy::Random:
                initialize_back_propagator<TtreePolicy,
                                           Texpander,
                                           TactionSelectorOptimization,
                                           TactionSelectorExecution,
                                           airlaps::RandomDefaultPolicy>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                back_propagator,
                                parallel,
                                debug_logs);
                break;
            
            default:
                spdlog::error("Available default policies: DefaultPolicy.Random");
                throw std::runtime_error("Available default policies: DefaultPolicy.Random");
        }
    }

    template <typename TtreePolicy,
              typename Texpander,
              typename TactionSelectorOptimization,
              typename TactionSelectorExecution,
              typename TdefaultPolicy>
    void initialize_back_propagator(
                py::object& domain,
                std::size_t time_budget,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                PyMCTSOptions::BackPropagator back_propagator,
                bool parallel,
                bool debug_logs) {
        switch (back_propagator) {
            case PyMCTSOptions::BackPropagator::Graph:
                initialize_execution<TtreePolicy,
                                     Texpander,
                                     TactionSelectorOptimization,
                                     TactionSelectorExecution,
                                     TdefaultPolicy,
                                     airlaps::GraphBackup>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                parallel,
                                debug_logs);
                break;
            
            default:
                spdlog::error("Available back propagators: BackPropagator.Graph");
                throw std::runtime_error("Available back propagators: BackPropagator.Graph");
        }
    }

    template <typename TtreePolicy,
              typename Texpander,
              typename TactionSelectorOptimization,
              typename TactionSelectorExecution,
              typename TdefaultPolicy,
              typename TbackPropagator>
    void initialize_execution(
                py::object& domain,
                std::size_t time_budget ,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                bool parallel ,
                bool debug_logs) {
        if (parallel) {
            initialize<airlaps::ParallelExecution,
                       TtreePolicy,
                       Texpander,
                       TactionSelectorOptimization,
                       TactionSelectorExecution,
                       TdefaultPolicy,
                       TbackPropagator>(domain,
                                        time_budget,
                                        rollout_budget,
                                        max_depth,
                                        discount,
                                        ucb_constant,
                                        debug_logs);
        } else {
            initialize<airlaps::SequentialExecution,
                       TtreePolicy,
                       Texpander,
                       TactionSelectorOptimization,
                       TactionSelectorExecution,
                       TdefaultPolicy,
                       TbackPropagator>(domain,
                                        time_budget,
                                        rollout_budget,
                                        max_depth,
                                        discount,
                                        ucb_constant,
                                                debug_logs);
        }
    }

    template <typename Texecution,
              typename TtreePolicy,
              typename Texpander,
              typename TactionSelectorOptimization,
              typename TactionSelectorExecution,
              typename TdefaultPolicy,
              typename TbackPropagator>
    void initialize(
                py::object& domain,
                std::size_t time_budget = 3600000,
                std::size_t rollout_budget = 100000,
                std::size_t max_depth = 1000,
                double discount = 1.0,
                double ucb_constant = 1.0 / std::sqrt(2.0),
                bool parallel = true,
                bool debug_logs = false) {
        _implementation = std::make_unique<Implementation<Texecution,
                                                          TtreePolicy,
                                                          Texpander,
                                                          TactionSelectorOptimization,
                                                          TactionSelectorExecution,
                                                          TdefaultPolicy,
                                                          TbackPropagator>>(
                                    domain,
                                    time_budget,
                                    rollout_budget,
                                    max_depth,
                                    discount,
                                    ucb_constant,
                                    debug_logs);
    }
};


void init_pymcts(py::module& m) {
    py::class_<PyMCTSOptions> py_mcts_options(m, "_MCTSOptions_");

    py::enum_<PyMCTSOptions::TreePolicy>(py_mcts_options, "TreePolicy")
        .value("Default", PyMCTSOptions::TreePolicy::Default);
    
    py::enum_<PyMCTSOptions::Expander>(py_mcts_options, "Expander")
        .value("Full", PyMCTSOptions::Expander::Full);
    
    py::enum_<PyMCTSOptions::ActionSelector>(py_mcts_options, "ActionSelector")
        .value("UCB1", PyMCTSOptions::ActionSelector::UCB1)
        .value("BestQValue", PyMCTSOptions::ActionSelector::BestQValue);
    
    py::enum_<PyMCTSOptions::DefaultPolicy>(py_mcts_options, "DefaultPolicy")
        .value("Random", PyMCTSOptions::DefaultPolicy::Random);
    
    py::enum_<PyMCTSOptions::BackPropagator>(py_mcts_options, "BackPropagator")
        .value("Graph", PyMCTSOptions::BackPropagator::Graph);
    
    py::class_<PyMCTSSolver> py_mcts_solver(m, "_MCTSSolver_");
        py_mcts_solver
            .def(py::init<py::object&,
                          std::size_t,
                          std::size_t,
                          std::size_t,
                          double,
                          bool,
                          double,
                          PyMCTSOptions::TreePolicy,
                          PyMCTSOptions::Expander,
                          PyMCTSOptions::ActionSelector,
                          PyMCTSOptions::ActionSelector,
                          PyMCTSOptions::DefaultPolicy,
                          PyMCTSOptions::BackPropagator,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("time_budget")=3600000,
                 py::arg("rollout_budget")=100000,
                 py::arg("max_depth")=1000,
                 py::arg("discount")=1.0,
                 py::arg("uct_mode")=true,
                 py::arg("ucb_constant")=1.0/std::sqrt(2.0),
                 py::arg("tree_policy")=PyMCTSOptions::TreePolicy::Default,
                 py::arg("expander")=PyMCTSOptions::Expander::Full,
                 py::arg("action_selector_optimization")=PyMCTSOptions::ActionSelector::UCB1,
                 py::arg("action_selector_execution")=PyMCTSOptions::ActionSelector::BestQValue,
                 py::arg("default_policy")=PyMCTSOptions::DefaultPolicy::Random,
                 py::arg("back_propagator")=PyMCTSOptions::BackPropagator::Graph,
                 py::arg("parallel")=true,
                 py::arg("debug_logs")=false)
            .def("clear", &PyMCTSSolver::clear)
            .def("solve", &PyMCTSSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &PyMCTSSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &PyMCTSSolver::get_next_action, py::arg("state"))
            .def("get_utility", &PyMCTSSolver::get_utility, py::arg("state"))
            .def("get_nb_of_explored_states", &PyMCTSSolver::get_nb_of_explored_states)
            .def("get_nb_rollouts", &PyMCTSSolver::get_nb_rollouts)
        ;
}
