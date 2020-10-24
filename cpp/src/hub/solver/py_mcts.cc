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


struct PyMCTSOptions {
    enum class TransitionMode {
        Step,
        Sample,
        Distribution
    };

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

    enum class RolloutPolicy {
        Random,
        Custom
    };

    enum class BackPropagator {
        Graph
    };
};


template <typename Texecution>
class PyMCTSDomain : public skdecide::PythonDomainAdapter<Texecution> {
public :

    template <typename Tsolver, typename TtransitionMode,
              std::enable_if_t<std::is_same<TtransitionMode, skdecide::StepTransitionMode<Tsolver>>::value, int> = 0>
    PyMCTSDomain(const py::object& domain, [[maybe_unused]] Tsolver* dummy_solver, [[maybe_unused]] TtransitionMode* dummy_transition_mode)
    : skdecide::PythonDomainAdapter<Texecution>(domain) {
        if (!py::hasattr(domain, "get_applicable_actions")) {
            throw std::invalid_argument("SKDECIDE exception: MCTS algorithm needs python domain for implementing get_applicable_actions()");
        }
        if (!py::hasattr(domain, "step")) {
            throw std::invalid_argument("SKDECIDE exception: MCTS algorithm with step transition mode needs python domain for implementing step()");
        }
    }

    template <typename Tsolver, typename TtransitionMode,
              std::enable_if_t<std::is_same<TtransitionMode, skdecide::SampleTransitionMode<Tsolver>>::value, int> = 0>
    PyMCTSDomain(const py::object& domain, [[maybe_unused]] Tsolver* dummy_solver, [[maybe_unused]] TtransitionMode* dummy_transition_mode)
    : skdecide::PythonDomainAdapter<Texecution>(domain) {
        if (!py::hasattr(domain, "get_applicable_actions")) {
            throw std::invalid_argument("SKDECIDE exception: MCTS algorithm needs python domain for implementing get_applicable_actions()");
        }
        if (!py::hasattr(domain, "sample")) {
            throw std::invalid_argument("SKDECIDE exception: MCTS algorithm with sample or distribution transition mode needs python domain for implementing sample()");
        }
    }

    template <typename Tsolver, typename TtransitionMode,
              std::enable_if_t<std::is_same<TtransitionMode, skdecide::DistributionTransitionMode<Tsolver>>::value, int> = 0>
    PyMCTSDomain(const py::object& domain, [[maybe_unused]] Tsolver* dummy_solver, [[maybe_unused]] TtransitionMode* dummy_transition_mode)
    : skdecide::PythonDomainAdapter<Texecution>(domain) {
        if (!py::hasattr(domain, "get_applicable_actions")) {
            throw std::invalid_argument("SKDECIDE exception: MCTS algorithm needs python domain for implementing get_applicable_actions()");
        }
        if (!py::hasattr(domain, "sample")) {
            throw std::invalid_argument("SKDECIDE exception: MCTS algorithm with sample or distribution transition mode needs python domain for implementing sample()");
        }
        if (!py::hasattr(domain, "get_next_state_distribution")) {
            throw std::invalid_argument("SKDECIDE exception: MCTS algorithm with distribution transition mode needs python domain for implementing get_next_state_distribution()");
        }
        if (!py::hasattr(domain, "get_transition_value")) {
            throw std::invalid_argument("SKDECIDE exception: MCTS algorithm with distribution transition mode needs python domain for implementing get_transition_value()");
        }
        if (!py::hasattr(domain, "is_terminal")) {
            throw std::invalid_argument("SKDECIDE exception: MCTS algorithm with distribution transition mode needs python domain for implementing is_terminal()");
        }
    }

};


class PyMCTSSolver {
public :

    typedef std::function<py::object (py::object&, const py::object&, const py::object&)> CustomPolicyFunctor;
    typedef std::function<py::object (py::object&, const py::object&, const py::object&)> HeuristicFunctor;

    PyMCTSSolver(py::object& domain,
                 std::size_t time_budget = 3600000,
                 std::size_t rollout_budget = 100000,
                 std::size_t max_depth = 1000,
                 double discount = 1.0,
                 bool uct_mode = true,
                 double ucb_constant = 1.0 / std::sqrt(2.0),
                 bool online_node_garbage = false,
                 const CustomPolicyFunctor& custom_policy = nullptr,
                 const HeuristicFunctor& heuristic = nullptr,
                 PyMCTSOptions::TransitionMode transition_mode = PyMCTSOptions::TransitionMode::Distribution,
                 PyMCTSOptions::TreePolicy tree_policy = PyMCTSOptions::TreePolicy::Default,
                 PyMCTSOptions::Expander expander = PyMCTSOptions::Expander::Full,
                 PyMCTSOptions::ActionSelector action_selector_optimization = PyMCTSOptions::ActionSelector::UCB1,
                 PyMCTSOptions::ActionSelector action_selector_execution = PyMCTSOptions::ActionSelector::BestQValue,
                 PyMCTSOptions::RolloutPolicy rollout_policy = PyMCTSOptions::RolloutPolicy::Random,
                 PyMCTSOptions::BackPropagator back_propagator = PyMCTSOptions::BackPropagator::Graph,
                 bool parallel = false,
                 bool debug_logs = false) {
        if (uct_mode) {
            initialize_transition_mode(domain,
                                       time_budget,
                                       rollout_budget,
                                       max_depth,
                                       discount,
                                       ucb_constant,
                                       online_node_garbage,
                                       custom_policy,
                                       heuristic,
                                       transition_mode,
                                       PyMCTSOptions::TreePolicy::Default,
                                       expander,
                                       PyMCTSOptions::ActionSelector::UCB1,
                                       action_selector_execution,
                                       PyMCTSOptions::RolloutPolicy::Random,
                                       PyMCTSOptions::BackPropagator::Graph,
                                       parallel,
                                       debug_logs);
        } else {
            initialize_transition_mode(domain,
                                       time_budget,
                                       rollout_budget,
                                       max_depth,
                                       discount,
                                       ucb_constant,
                                       online_node_garbage,
                                       custom_policy,
                                       heuristic,
                                       transition_mode,
                                       tree_policy,
                                       expander,
                                       action_selector_optimization,
                                       action_selector_execution,
                                       rollout_policy,
                                       back_propagator,
                                       parallel,
                                       debug_logs);
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

    py::int_ get_nb_of_explored_states() {
        return _implementation->get_nb_of_explored_states();
    }

    py::int_ get_nb_rollouts() {
        return _implementation->get_nb_rollouts();
    }

    py::dict get_policy() {
        return _implementation->get_policy();
    }

    py::list get_action_prefix() {
        return _implementation->get_action_prefix();
    }

private :

    class BaseImplementation {
    public :
        virtual ~BaseImplementation() {}
        virtual void clear() =0;
        virtual void solve(const py::object& s) =0;
        virtual py::bool_ is_solution_defined_for(const py::object& s) =0;
        virtual py::object get_next_action(const py::object& s) =0;
        virtual py::float_ get_utility(const py::object& s) =0;
        virtual py::int_ get_nb_of_explored_states() =0;
        virtual py::int_ get_nb_rollouts() =0;
        virtual py::dict get_policy() =0;
        virtual py::list get_action_prefix() =0;
    };

    template <typename Texecution,
              template <typename Tsolver> class TtransitionMode,
              template <typename Tsolver> class TtreePolicy,
              template <typename Tsolver> class Texpander,
              template <typename Tsolver> class TactionSelectorOptimization,
              template <typename Tsolver> class TactionSelectorExecution,
              template <typename Tsolver> class TrolloutPolicy,
              template <typename Tsolver> class TbackPropagator>
    class Implementation : public BaseImplementation {
    public :

        typedef skdecide::MCTSSolver<PyMCTSDomain<Texecution>, Texecution,
                                     TtransitionMode, TtreePolicy, Texpander,
                                     TactionSelectorOptimization, TactionSelectorExecution,
                                     TrolloutPolicy, TbackPropagator> PyMCTSSolver;
        
        Implementation(py::object& domain,
                       std::size_t time_budget,
                       std::size_t rollout_budget,
                       std::size_t max_depth,
                       double discount,
                       double ucb_constant,
                       bool online_node_garbage,
                       const CustomPolicyFunctor& custom_policy,
                       const HeuristicFunctor& heuristic,
                       bool debug_logs)
            : _custom_policy(custom_policy),
              _heuristic(heuristic) {

            _domain = std::make_unique<PyMCTSDomain<Texecution>>(domain, (PyMCTSSolver*) nullptr, (TtransitionMode<PyMCTSSolver>*) nullptr);
            _solver = std::make_unique<PyMCTSSolver>(
                        *_domain,
                        time_budget,
                        rollout_budget,
                        max_depth,
                        discount,
                        online_node_garbage,
                        debug_logs,
                        init_tree_policy(),
                        init_expander(_heuristic),
                        init_action_selector((TactionSelectorOptimization<PyMCTSSolver>*) nullptr, ucb_constant),
                        init_action_selector((TactionSelectorExecution<PyMCTSSolver>*) nullptr, ucb_constant),
                        init_rollout_policy(_custom_policy),
                        init_back_propagator());
            _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(std::cout,
                                                                             py::module::import("sys").attr("stdout"));
            _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(std::cerr,
                                                                             py::module::import("sys").attr("stderr"));
        }

        virtual ~Implementation() {}

        std::unique_ptr<TtreePolicy<PyMCTSSolver>> init_tree_policy() {
            return std::make_unique<TtreePolicy<PyMCTSSolver>>();
        }

        std::unique_ptr<Texpander<PyMCTSSolver>> init_expander(const HeuristicFunctor& heuristic) {
            if (!heuristic) { // initialize new nodes with 0 value and 0 visits count
                return std::make_unique<Texpander<PyMCTSSolver>>();
            } else {
                return std::make_unique<Texpander<PyMCTSSolver>>([&heuristic](
                                                    PyMCTSDomain<Texecution>& d,
                                                    const typename PyMCTSDomain<Texecution>::State& s,
                                                    const std::size_t* thread_id) -> std::pair<double, std::size_t> {
                    try {
                        std::unique_ptr<py::object> r = d.call(thread_id, heuristic, s.pyobj());
                        typename skdecide::GilControl<Texecution>::Acquire acquire;
                        std::pair<double, std::size_t> rr = r->template cast<std::pair<double, std::size_t>>();
                        r.reset();
                        return  rr;
                    } catch (const std::exception& e) {
                        spdlog::error(std::string("SKDECIDE exception when calling the custom heuristic: ") + e.what());
                        throw;
                    }
                });
            }
        }

        template <typename TactionSelector,
                  std::enable_if_t<std::is_same<TactionSelector, skdecide::UCB1ActionSelector<PyMCTSSolver>>::value, int> = 0>
        std::unique_ptr<TactionSelector> init_action_selector(TactionSelector* dummy, double ucb_constant) {
            return std::make_unique<skdecide::UCB1ActionSelector<PyMCTSSolver>>(ucb_constant);
        }

        template <typename TactionSelector,
                  std::enable_if_t<std::is_same<TactionSelector, skdecide::BestQValueActionSelector<PyMCTSSolver>>::value, int> = 0>
        std::unique_ptr<TactionSelector> init_action_selector(TactionSelector* dummy, double ucb_constant) {
            return std::make_unique<skdecide::BestQValueActionSelector<PyMCTSSolver>>();
        }

        std::unique_ptr<TrolloutPolicy<PyMCTSSolver>> init_rollout_policy(const CustomPolicyFunctor& custom_policy) {
            if (!custom_policy) { // use random rollout policy
                return std::make_unique<TrolloutPolicy<PyMCTSSolver>>();
            } else {
                return std::make_unique<TrolloutPolicy<PyMCTSSolver>>([&custom_policy](
                                                    PyMCTSDomain<Texecution>& d,
                                                    const typename PyMCTSDomain<Texecution>::State& s,
                                                    const std::size_t* thread_id) -> typename PyMCTSDomain<Texecution>::Action {
                    try {
                        return typename PyMCTSDomain<Texecution>::Action(d.call(thread_id, custom_policy, s.pyobj()));
                    } catch (const std::exception& e) {
                        spdlog::error(std::string("SKDECIDE exception when calling the custom rollout policy: ") + e.what());
                        throw;
                    }
                });
            }
        }

        std::unique_ptr<TbackPropagator<PyMCTSSolver>> init_back_propagator() {
            return std::make_unique<TbackPropagator<PyMCTSSolver>>();
        }

        virtual void clear() {
            _solver->clear();
        }

        virtual void solve(const py::object& s) {
            typename skdecide::GilControl<Texecution>::Release release;
            _solver->solve(s);
        }

        virtual py::bool_ is_solution_defined_for(const py::object& s) {
            return _solver->is_solution_defined_for(s);
        }

        virtual py::object get_next_action(const py::object& s) {
            return _solver->get_best_action(s).pyobj();
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

        virtual py::dict get_policy() {
            py::dict d;
            auto&& p = _solver->policy();
            for (auto& e : p) {
                d[e.first.pyobj()] = py::make_tuple(e.second.first.pyobj(), e.second.second);
            }
            return d;
        }

        virtual py::list get_action_prefix() {
            py::list l;
            const auto& ll = _solver->action_prefix();
            for (const auto& e : ll) {
                l.append(e);
            }
            return l;
        }

    private :
        std::unique_ptr<PyMCTSDomain<Texecution>> _domain;
        std::unique_ptr<PyMCTSSolver> _solver;

        CustomPolicyFunctor _custom_policy;
        HeuristicFunctor _heuristic;

        std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
        std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
    };

    std::unique_ptr<BaseImplementation> _implementation;

    void initialize_transition_mode(
                py::object& domain,
                std::size_t time_budget,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                bool online_node_garbage,
                const CustomPolicyFunctor& custom_policy,
                const HeuristicFunctor& heuristic,
                PyMCTSOptions::TransitionMode transition_mode,
                PyMCTSOptions::TreePolicy tree_policy,
                PyMCTSOptions::Expander expander,
                PyMCTSOptions::ActionSelector action_selector_optimization,
                PyMCTSOptions::ActionSelector action_selector_execution,
                PyMCTSOptions::RolloutPolicy rollout_policy,
                PyMCTSOptions::BackPropagator back_propagator,
                bool parallel,
                bool debug_logs) {
        switch (transition_mode) {
            case PyMCTSOptions::TransitionMode::Step:
                initialize_tree_policy<skdecide::StepTransitionMode>(
                                            domain,
                                            time_budget,
                                            rollout_budget,
                                            max_depth,
                                            discount,
                                            ucb_constant,
                                            online_node_garbage,
                                            custom_policy,
                                            heuristic,
                                            tree_policy,
                                            expander,
                                            action_selector_optimization,
                                            action_selector_execution,
                                            rollout_policy,
                                            back_propagator,
                                            parallel,
                                            debug_logs);
                break;
            
            case PyMCTSOptions::TransitionMode::Sample:
                initialize_tree_policy<skdecide::SampleTransitionMode>(
                                            domain,
                                            time_budget,
                                            rollout_budget,
                                            max_depth,
                                            discount,
                                            ucb_constant,
                                            online_node_garbage,
                                            custom_policy,
                                            heuristic,
                                            tree_policy,
                                            expander,
                                            action_selector_optimization,
                                            action_selector_execution,
                                            rollout_policy,
                                            back_propagator,
                                            parallel,
                                            debug_logs);
                break;
            
            case PyMCTSOptions::TransitionMode::Distribution:
                initialize_tree_policy<skdecide::DistributionTransitionMode>(
                                            domain,
                                            time_budget,
                                            rollout_budget,
                                            max_depth,
                                            discount,
                                            ucb_constant,
                                            online_node_garbage,
                                            custom_policy,
                                            heuristic,
                                            tree_policy,
                                            expander,
                                            action_selector_optimization,
                                            action_selector_execution,
                                            rollout_policy,
                                            back_propagator,
                                            parallel,
                                            debug_logs);
                break;
            
            default:
                spdlog::error("Available transition modes: TransitionMode.Step , TransitionMode.Sample , TransitionMode.Distribution");
                throw std::runtime_error("Available transition modes: TransitionMode.Step , TransitionMode.Sample , TransitionMode.Distribution");
        }
    }

    template <template <typename Tsolver> class TtransitionMode>
    void initialize_tree_policy(
                py::object& domain,
                std::size_t time_budget,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                bool online_node_garbage,
                const CustomPolicyFunctor& custom_policy,
                const HeuristicFunctor& heuristic,
                PyMCTSOptions::TreePolicy tree_policy,
                PyMCTSOptions::Expander expander,
                PyMCTSOptions::ActionSelector action_selector_optimization,
                PyMCTSOptions::ActionSelector action_selector_execution,
                PyMCTSOptions::RolloutPolicy rollout_policy,
                PyMCTSOptions::BackPropagator back_propagator,
                bool parallel,
                bool debug_logs) {
        switch (tree_policy) {
            case PyMCTSOptions::TreePolicy::Default:
                initialize_expander<TtransitionMode,
                                    skdecide::DefaultTreePolicy>(
                                        domain,
                                        time_budget,
                                        rollout_budget,
                                        max_depth,
                                        discount,
                                        ucb_constant,
                                        online_node_garbage,
                                        custom_policy,
                                        heuristic,
                                        expander,
                                        action_selector_optimization,
                                        action_selector_execution,
                                        rollout_policy,
                                        back_propagator,
                                        parallel,
                                        debug_logs);
                break;
            
            default:
                spdlog::error("Available tree policies: TreePolicy.Default");
                throw std::runtime_error("Available tree policies: TreePolicy.Default");
        }
    }

    template <template <typename Tsolver> class TtransitionMode,
              template <typename Tsolver> class TtreePolicy>
    void initialize_expander(
                py::object& domain,
                std::size_t time_budget,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                bool online_node_garbage,
                const CustomPolicyFunctor& custom_policy,
                const HeuristicFunctor& heuristic,
                PyMCTSOptions::Expander expander,
                PyMCTSOptions::ActionSelector action_selector_optimization,
                PyMCTSOptions::ActionSelector action_selector_execution,
                PyMCTSOptions::RolloutPolicy rollout_policy,
                PyMCTSOptions::BackPropagator back_propagator,
                bool parallel,
                bool debug_logs) {
        switch (expander) {
            case PyMCTSOptions::Expander::Full:
                initialize_action_selector_optimization<TtransitionMode,
                                                        TtreePolicy,
                                                        skdecide::FullExpand>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                online_node_garbage,
                                custom_policy,
                                heuristic,
                                action_selector_optimization,
                                action_selector_execution,
                                rollout_policy,
                                back_propagator,
                                parallel,
                                debug_logs);
                break;
            
            default:
                spdlog::error("Available expanders: Expander.Full");
                throw std::runtime_error("Available expanders: Expander.Full");
        }
    }

    template <template <typename Tsolver> class TtransitionMode,
              template <typename Tsolver> class TtreePolicy,
              template <typename Tsolver> class Texpander>
    void initialize_action_selector_optimization(
                py::object& domain,
                std::size_t time_budget,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                bool online_node_garbage,
                const CustomPolicyFunctor& custom_policy,
                const HeuristicFunctor& heuristic,
                PyMCTSOptions::ActionSelector action_selector_optimization,
                PyMCTSOptions::ActionSelector action_selector_execution,
                PyMCTSOptions::RolloutPolicy rollout_policy,
                PyMCTSOptions::BackPropagator back_propagator,
                bool parallel,
                bool debug_logs) {
        switch (action_selector_optimization) {
            case PyMCTSOptions::ActionSelector::UCB1:
                initialize_action_selector_execution<TtransitionMode,
                                                     TtreePolicy,
                                                     Texpander,
                                                     skdecide::UCB1ActionSelector>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                online_node_garbage,
                                custom_policy,
                                heuristic,
                                action_selector_execution,
                                rollout_policy,
                                back_propagator,
                                parallel,
                                debug_logs);
                break;
            
            case PyMCTSOptions::ActionSelector::BestQValue:
                initialize_action_selector_execution<TtransitionMode,
                                                     TtreePolicy,
                                                     Texpander,
                                                     skdecide::BestQValueActionSelector>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                online_node_garbage,
                                custom_policy,
                                heuristic,
                                action_selector_execution,
                                rollout_policy,
                                back_propagator,
                                parallel,
                                debug_logs);
                break;
            
            default:
                spdlog::error("Available action selector: ActionSelector.UCB1 , ActionSelector.BestQValue");
                throw std::runtime_error("Available action selector: ActionSelector.UCB1 , ActionSelector.BestQValue");
        }
    }

    template <template <typename Tsolver> class TtransitionMode,
              template <typename Tsolver> class TtreePolicy,
              template <typename Tsolver> class Texpander,
              template <typename Tsolver> class TactionSelectorOptimization>
    void initialize_action_selector_execution(
                py::object& domain,
                std::size_t time_budget,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                bool online_node_garbage,
                const CustomPolicyFunctor& custom_policy,
                const HeuristicFunctor& heuristic,
                PyMCTSOptions::ActionSelector action_selector_execution,
                PyMCTSOptions::RolloutPolicy rollout_policy,
                PyMCTSOptions::BackPropagator back_propagator,
                bool parallel,
                bool debug_logs) {
        switch (action_selector_execution) {
            case PyMCTSOptions::ActionSelector::UCB1:
                initialize_rollout_policy<TtransitionMode,
                                          TtreePolicy,
                                          Texpander,
                                          TactionSelectorOptimization,
                                          skdecide::UCB1ActionSelector>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                online_node_garbage,
                                custom_policy,
                                heuristic,
                                rollout_policy,
                                back_propagator,
                                parallel,
                                debug_logs);
                break;
            
            case PyMCTSOptions::ActionSelector::BestQValue:
                initialize_rollout_policy<TtransitionMode,
                                          TtreePolicy,
                                          Texpander,
                                          TactionSelectorOptimization,
                                          skdecide::BestQValueActionSelector>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                online_node_garbage,
                                custom_policy,
                                heuristic,
                                rollout_policy,
                                back_propagator,
                                parallel,
                                debug_logs);
                break;
            
            default:
                spdlog::error("Available action selector: ActionSelector.UCB1 , ActionSelector.BestQValue");
                throw std::runtime_error("Available action selector: ActionSelector.UCB1 , ActionSelector.BestQValue");
        }
    }

    template <template <typename Tsolver> class TtransitionMode,
              template <typename Tsolver> class TtreePolicy,
              template <typename Tsolver> class Texpander,
              template <typename Tsolver> class TactionSelectorOptimization,
              template <typename Tsolver> class TactionSelectorExecution>
    void initialize_rollout_policy(
                py::object& domain,
                std::size_t time_budget,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                bool online_node_garbage,
                const CustomPolicyFunctor& custom_policy,
                const HeuristicFunctor& heuristic,
                PyMCTSOptions::RolloutPolicy rollout_policy,
                PyMCTSOptions::BackPropagator back_propagator,
                bool parallel,
                bool debug_logs) {
        switch (rollout_policy) {
            case PyMCTSOptions::RolloutPolicy::Random:
                initialize_back_propagator<TtransitionMode,
                                           TtreePolicy,
                                           Texpander,
                                           TactionSelectorOptimization,
                                           TactionSelectorExecution,
                                           skdecide::DefaultRolloutPolicy>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                online_node_garbage,
                                nullptr,
                                heuristic,
                                back_propagator,
                                parallel,
                                debug_logs);
                break;
            
            case PyMCTSOptions::RolloutPolicy::Custom:
                initialize_back_propagator<TtransitionMode,
                                           TtreePolicy,
                                           Texpander,
                                           TactionSelectorOptimization,
                                           TactionSelectorExecution,
                                           skdecide::DefaultRolloutPolicy>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                online_node_garbage,
                                custom_policy,
                                heuristic,
                                back_propagator,
                                parallel,
                                debug_logs);
                break;
            
            default:
                spdlog::error("Available default policies: RolloutPolicy.Random");
                throw std::runtime_error("Available default policies: RolloutPolicy.Random");
        }
    }

    template <template <typename Tsolver> class TtransitionMode,
              template <typename Tsolver> class TtreePolicy,
              template <typename Tsolver> class Texpander,
              template <typename Tsolver> class TactionSelectorOptimization,
              template <typename Tsolver> class TactionSelectorExecution,
              template <typename Tsolver> class TrolloutPolicy>
    void initialize_back_propagator(
                py::object& domain,
                std::size_t time_budget,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                bool online_node_garbage,
                const CustomPolicyFunctor& custom_policy,
                const HeuristicFunctor& heuristic,
                PyMCTSOptions::BackPropagator back_propagator,
                bool parallel,
                bool debug_logs) {
        switch (back_propagator) {
            case PyMCTSOptions::BackPropagator::Graph:
                initialize_execution<TtransitionMode,
                                     TtreePolicy,
                                     Texpander,
                                     TactionSelectorOptimization,
                                     TactionSelectorExecution,
                                     TrolloutPolicy,
                                     skdecide::GraphBackup>(
                                domain,
                                time_budget,
                                rollout_budget,
                                max_depth,
                                discount,
                                ucb_constant,
                                online_node_garbage,
                                custom_policy,
                                heuristic,
                                parallel,
                                debug_logs);
                break;
            
            default:
                spdlog::error("Available back propagators: BackPropagator.Graph");
                throw std::runtime_error("Available back propagators: BackPropagator.Graph");
        }
    }

    template <template <typename Tsolver> class TtransitionMode,
              template <typename Tsolver> class TtreePolicy,
              template <typename Tsolver> class Texpander,
              template <typename Tsolver> class TactionSelectorOptimization,
              template <typename Tsolver> class TactionSelectorExecution,
              template <typename Tsolver> class TrolloutPolicy,
              template <typename Tsolver> class TbackPropagator>
    void initialize_execution(
                py::object& domain,
                std::size_t time_budget ,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                bool online_node_garbage,
                const CustomPolicyFunctor& custom_policy,
                const HeuristicFunctor& heuristic,
                bool parallel ,
                bool debug_logs) {
        if (parallel) {
            initialize<skdecide::ParallelExecution,
                       TtransitionMode,
                       TtreePolicy,
                       Texpander,
                       TactionSelectorOptimization,
                       TactionSelectorExecution,
                       TrolloutPolicy,
                       TbackPropagator>(domain,
                                        time_budget,
                                        rollout_budget,
                                        max_depth,
                                        discount,
                                        ucb_constant,
                                        online_node_garbage,
                                        custom_policy,
                                        heuristic,
                                        debug_logs);
        } else {
            initialize<skdecide::SequentialExecution,
                       TtransitionMode,
                       TtreePolicy,
                       Texpander,
                       TactionSelectorOptimization,
                       TactionSelectorExecution,
                       TrolloutPolicy,
                       TbackPropagator>(domain,
                                        time_budget,
                                        rollout_budget,
                                        max_depth,
                                        discount,
                                        ucb_constant,
                                        online_node_garbage,
                                        custom_policy,
                                        heuristic,
                                        debug_logs);
        }
    }

    template <typename Texecution,
              template <typename Tsolver> class TtransitionMode,
              template <typename Tsolver> class TtreePolicy,
              template <typename Tsolver> class Texpander,
              template <typename Tsolver> class TactionSelectorOptimization,
              template <typename Tsolver> class TactionSelectorExecution,
              template <typename Tsolver> class TrolloutPolicy,
              template <typename Tsolver> class TbackPropagator>
    void initialize(
                py::object& domain,
                std::size_t time_budget,
                std::size_t rollout_budget,
                std::size_t max_depth,
                double discount,
                double ucb_constant,
                bool online_node_garbage,
                const CustomPolicyFunctor& custom_policy,
                const HeuristicFunctor& heuristic,
                bool debug_logs = false) {
        _implementation = std::make_unique<Implementation<Texecution,
                                                          TtransitionMode,
                                                          TtreePolicy,
                                                          Texpander,
                                                          TactionSelectorOptimization,
                                                          TactionSelectorExecution,
                                                          TrolloutPolicy,
                                                          TbackPropagator>>(
                                    domain,
                                    time_budget,
                                    rollout_budget,
                                    max_depth,
                                    discount,
                                    ucb_constant,
                                    online_node_garbage,
                                    custom_policy,
                                    heuristic,
                                    debug_logs);
    }
};


void init_pymcts(py::module& m) {
    py::class_<PyMCTSOptions> py_mcts_options(m, "_MCTSOptions_");

    py::enum_<PyMCTSOptions::TransitionMode>(py_mcts_options, "TransitionMode")
        .value("Step", PyMCTSOptions::TransitionMode::Step)
        .value("Sample", PyMCTSOptions::TransitionMode::Sample)
        .value("Distribution", PyMCTSOptions::TransitionMode::Distribution);

    py::enum_<PyMCTSOptions::TreePolicy>(py_mcts_options, "TreePolicy")
        .value("Default", PyMCTSOptions::TreePolicy::Default);
    
    py::enum_<PyMCTSOptions::Expander>(py_mcts_options, "Expander")
        .value("Full", PyMCTSOptions::Expander::Full);
    
    py::enum_<PyMCTSOptions::ActionSelector>(py_mcts_options, "ActionSelector")
        .value("UCB1", PyMCTSOptions::ActionSelector::UCB1)
        .value("BestQValue", PyMCTSOptions::ActionSelector::BestQValue);
    
    py::enum_<PyMCTSOptions::RolloutPolicy>(py_mcts_options, "RolloutPolicy")
        .value("Random", PyMCTSOptions::RolloutPolicy::Random)
        .value("Custom", PyMCTSOptions::RolloutPolicy::Custom);
    
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
                          bool,
                          const std::function<py::object (const py::object&, const py::object&, const py::object&)>&,
                          const std::function<py::object (const py::object&, const py::object&, const py::object&)>&,
                          PyMCTSOptions::TransitionMode,
                          PyMCTSOptions::TreePolicy,
                          PyMCTSOptions::Expander,
                          PyMCTSOptions::ActionSelector,
                          PyMCTSOptions::ActionSelector,
                          PyMCTSOptions::RolloutPolicy,
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
                 py::arg("online_node_garbage")=false,
                 py::arg("custom_policy")=nullptr,
                 py::arg("heuristic")=nullptr,
                 py::arg("transition_mode")=PyMCTSOptions::TransitionMode::Distribution,
                 py::arg("tree_policy")=PyMCTSOptions::TreePolicy::Default,
                 py::arg("expander")=PyMCTSOptions::Expander::Full,
                 py::arg("action_selector_optimization")=PyMCTSOptions::ActionSelector::UCB1,
                 py::arg("action_selector_execution")=PyMCTSOptions::ActionSelector::BestQValue,
                 py::arg("rollout_policy")=PyMCTSOptions::RolloutPolicy::Random,
                 py::arg("back_propagator")=PyMCTSOptions::BackPropagator::Graph,
                 py::arg("parallel")=false,
                 py::arg("debug_logs")=false)
            .def("clear", &PyMCTSSolver::clear)
            .def("solve", &PyMCTSSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &PyMCTSSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &PyMCTSSolver::get_next_action, py::arg("state"))
            .def("get_utility", &PyMCTSSolver::get_utility, py::arg("state"))
            .def("get_nb_of_explored_states", &PyMCTSSolver::get_nb_of_explored_states)
            .def("get_nb_rollouts", &PyMCTSSolver::get_nb_rollouts)
            .def("get_policy", &PyMCTSSolver::get_policy)
            .def("get_action_prefix", &PyMCTSSolver::get_action_prefix)
        ;
}
