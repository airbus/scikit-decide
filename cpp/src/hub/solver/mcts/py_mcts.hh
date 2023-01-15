/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_MCTS_HH
#define SKDECIDE_PY_MCTS_HH

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "utils/execution.hh"
#include "utils/python_gil_control.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "mcts.hh"

namespace py = pybind11;

namespace skdecide {

struct PyMCTSOptions {
  enum class TransitionMode { Step, Sample, Distribution };

  enum class TreePolicy { Default };

  enum class Expander { Full, Partial };

  enum class ActionSelector { UCB1, BestQValue };

  enum class RolloutPolicy { Random, Custom, Void };

  enum class BackPropagator { Graph };
};

template <typename Texecution>
using PyMCTSDomain = PythonDomainProxy<Texecution>;

#define MCTS_SOLVER_DECL_ARGS                                                  \
  py::object &domain, std::size_t time_budget, std::size_t rollout_budget,     \
      std::size_t max_depth, std::size_t epsilon_moving_average_window,        \
      double epsilon, double discount, double ucb_constant,                    \
      bool online_node_garbage, const CustomPolicyFunctor &custom_policy,      \
      const HeuristicFunctor &heuristic, double state_expansion_rate,          \
      double action_expansion_rate, bool debug_logs,                           \
      const WatchdogFunctor &watchdog

#define MCTS_SOLVER_ARGS                                                       \
  domain, time_budget, rollout_budget, max_depth,                              \
      epsilon_moving_average_window, epsilon, discount, ucb_constant,          \
      online_node_garbage, custom_policy, heuristic, state_expansion_rate,     \
      action_expansion_rate, debug_logs, watchdog

class PyMCTSSolver {
public:
  typedef std::function<py::object(py::object &, const py::object &,
                                   const py::object &)>
      CustomPolicyFunctor;
  typedef std::function<py::object(py::object &, const py::object &,
                                   const py::object &)>
      HeuristicFunctor;
  typedef std::function<bool(const std::size_t &, const std::size_t &,
                             const double &, const double &)>
      WatchdogFunctor;

private:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void close() = 0;
    virtual void clear() = 0;
    virtual void solve(const py::object &s) = 0;
    virtual py::bool_ is_solution_defined_for(const py::object &s) = 0;
    virtual py::object get_next_action(const py::object &s) = 0;
    virtual py::float_ get_utility(const py::object &s) = 0;
    virtual py::int_ get_nb_of_explored_states() = 0;
    virtual py::int_ get_nb_rollouts() = 0;
    virtual py::dict get_policy() = 0;
    virtual py::list get_action_prefix() = 0;
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
  public:
    typedef skdecide::MCTSSolver<
        PyMCTSDomain<Texecution>, Texecution, TtransitionMode, TtreePolicy,
        Texpander, TactionSelectorOptimization, TactionSelectorExecution,
        TrolloutPolicy, TbackPropagator>
        PyMCTSSolver;

    Implementation(MCTS_SOLVER_DECL_ARGS)
        : _custom_policy(custom_policy), _heuristic(heuristic),
          _watchdog(watchdog) {

      _domain = std::make_unique<PyMCTSDomain<Texecution>>(domain);
      _solver = std::make_unique<PyMCTSSolver>(
          *_domain, time_budget, rollout_budget, max_depth,
          epsilon_moving_average_window, epsilon, discount, online_node_garbage,
          debug_logs, init_watchdog(), init_tree_policy(),
          init_expander(_heuristic, state_expansion_rate,
                        action_expansion_rate),
          init_action_selector<TactionSelectorOptimization<PyMCTSSolver>>(
              ucb_constant),
          init_action_selector<TactionSelectorExecution<PyMCTSSolver>>(
              ucb_constant),
          init_rollout_policy(_custom_policy), init_back_propagator());
      _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(
          std::cout, py::module::import("sys").attr("stdout"));
      _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(
          std::cerr, py::module::import("sys").attr("stderr"));
    }

    virtual ~Implementation() {}

    WatchdogFunctor init_watchdog() {
      return [this](const std::size_t &elapsed_time,
                    const std::size_t &nb_rollouts, const double &best_value,
                    const double &epsilon_moving_average) -> bool {
        if (_watchdog) {
          typename skdecide::GilControl<Texecution>::Acquire acquire;
          return _watchdog(elapsed_time, nb_rollouts, best_value,
                           epsilon_moving_average);
        } else {
          return true;
        }
      };
    }

    std::unique_ptr<TtreePolicy<PyMCTSSolver>> init_tree_policy() {
      return std::make_unique<TtreePolicy<PyMCTSSolver>>();
    }

    std::function<std::pair<typename PyMCTSDomain<Texecution>::Value,
                            std::size_t>(
        PyMCTSDomain<Texecution> &,
        const typename PyMCTSDomain<Texecution>::State &, const std::size_t *)>
    construct_heuristic(const HeuristicFunctor &heuristic) {
      return [&heuristic](PyMCTSDomain<Texecution> &d,
                          const typename PyMCTSDomain<Texecution>::State &s,
                          const std::size_t *thread_id)
                 -> std::pair<typename PyMCTSDomain<Texecution>::Value,
                              std::size_t> {
        try {
          std::unique_ptr<py::object> r =
              d.call(thread_id, heuristic, s.pyobj());
          typename skdecide::GilControl<Texecution>::Acquire acquire;
          py::tuple t = py::cast<py::tuple>(*r);
          std::pair<typename PyMCTSDomain<Texecution>::Value, std::size_t> rr =
              std::make_pair(typename PyMCTSDomain<Texecution>::Value(t[0]),
                             t[1].template cast<std::size_t>());
          r.reset();
          return rr;
        } catch (const std::exception &e) {
          Logger::error(
              std::string(
                  "SKDECIDE exception when calling the custom heuristic: ") +
              e.what());
          throw;
        }
      };
    }

    template <
        typename TExpander = Texpander<PyMCTSSolver>,
        std::enable_if_t<
            std::is_same<TExpander, skdecide::FullExpand<PyMCTSSolver>>::value,
            int> = 0>
    std::unique_ptr<Texpander<PyMCTSSolver>>
    init_expander(const HeuristicFunctor &heuristic,
                  double state_expansion_rate, double action_expansion_rate) {
      if (!heuristic) { // use (0.0, 0) heuristic
        return std::make_unique<skdecide::FullExpand<PyMCTSSolver>>();
      } else {
        return std::make_unique<skdecide::FullExpand<PyMCTSSolver>>(
            construct_heuristic(heuristic));
      }
    }

    template <typename TExpander = Texpander<PyMCTSSolver>,
              std::enable_if_t<
                  std::is_same<TExpander,
                               skdecide::PartialExpand<PyMCTSSolver>>::value,
                  int> = 0>
    std::unique_ptr<Texpander<PyMCTSSolver>>
    init_expander(const HeuristicFunctor &heuristic,
                  double state_expansion_rate, double action_expansion_rate) {
      if (!heuristic) { // use (0.0, 0) heuristic
        return std::make_unique<skdecide::PartialExpand<PyMCTSSolver>>(
            state_expansion_rate, action_expansion_rate);
      } else {
        return std::make_unique<skdecide::PartialExpand<PyMCTSSolver>>(
            state_expansion_rate, action_expansion_rate,
            construct_heuristic(heuristic));
      }
    }

    template <typename TactionSelector,
              std::enable_if_t<
                  std::is_same<TactionSelector, skdecide::UCB1ActionSelector<
                                                    PyMCTSSolver>>::value,
                  int> = 0>
    std::unique_ptr<TactionSelector> init_action_selector(double ucb_constant) {
      return std::make_unique<skdecide::UCB1ActionSelector<PyMCTSSolver>>(
          ucb_constant);
    }

    template <typename TactionSelector,
              std::enable_if_t<std::is_same<TactionSelector,
                                            skdecide::BestQValueActionSelector<
                                                PyMCTSSolver>>::value,
                               int> = 0>
    std::unique_ptr<TactionSelector> init_action_selector(double ucb_constant) {
      return std::make_unique<
          skdecide::BestQValueActionSelector<PyMCTSSolver>>();
    }

    template <typename TRolloutPolicy = TrolloutPolicy<PyMCTSSolver>,
              std::enable_if_t<
                  std::is_same<TRolloutPolicy, skdecide::DefaultRolloutPolicy<
                                                   PyMCTSSolver>>::value,
                  int> = 0>
    std::unique_ptr<TrolloutPolicy<PyMCTSSolver>>
    init_rollout_policy(const CustomPolicyFunctor &custom_policy) {
      if (!custom_policy) { // use random rollout policy
        return std::make_unique<TrolloutPolicy<PyMCTSSolver>>();
      } else {
        return std::make_unique<TrolloutPolicy<PyMCTSSolver>>(
            [&custom_policy](PyMCTSDomain<Texecution> &d,
                             const typename PyMCTSDomain<Texecution>::State &s,
                             const std::size_t *thread_id) ->
            typename PyMCTSDomain<Texecution>::Action {
              try {
                return typename PyMCTSDomain<Texecution>::Action(
                    d.call(thread_id, custom_policy, s.pyobj()));
              } catch (const std::exception &e) {
                Logger::error(std::string("SKDECIDE exception when calling the "
                                          "custom rollout policy: ") +
                              e.what());
                throw;
              }
            });
      }
    }

    template <typename TRolloutPolicy = TrolloutPolicy<PyMCTSSolver>,
              std::enable_if_t<
                  std::is_same<TRolloutPolicy, skdecide::VoidRolloutPolicy<
                                                   PyMCTSSolver>>::value,
                  int> = 0>
    std::unique_ptr<TrolloutPolicy<PyMCTSSolver>>
    init_rollout_policy(const CustomPolicyFunctor &custom_policy) {
      return std::make_unique<TrolloutPolicy<PyMCTSSolver>>();
    }

    std::unique_ptr<TbackPropagator<PyMCTSSolver>> init_back_propagator() {
      return std::make_unique<TbackPropagator<PyMCTSSolver>>();
    }

    virtual void close() { _domain->close(); }

    virtual void clear() { _solver->clear(); }

    virtual void solve(const py::object &s) {
      typename skdecide::GilControl<Texecution>::Release release;
      _solver->solve(s);
    }

    virtual py::bool_ is_solution_defined_for(const py::object &s) {
      return _solver->is_solution_defined_for(s);
    }

    virtual py::object get_next_action(const py::object &s) {
      return _solver->get_best_action(s).pyobj();
    }

    virtual py::float_ get_utility(const py::object &s) {
      return _solver->get_best_value(s);
    }

    virtual py::int_ get_nb_of_explored_states() {
      return _solver->nb_of_explored_states();
    }

    virtual py::int_ get_nb_rollouts() { return _solver->nb_rollouts(); }

    virtual py::dict get_policy() {
      py::dict d;
      auto &&p = _solver->policy();
      for (auto &e : p) {
        d[e.first.pyobj()] =
            py::make_tuple(e.second.first.pyobj(), e.second.second);
      }
      return d;
    }

    virtual py::list get_action_prefix() {
      py::list l;
      const auto &ll = _solver->action_prefix();
      for (const auto &e : ll) {
        l.append(e);
      }
      return l;
    }

  private:
    std::unique_ptr<PyMCTSDomain<Texecution>> _domain;
    std::unique_ptr<PyMCTSSolver> _solver;

    CustomPolicyFunctor _custom_policy;
    HeuristicFunctor _heuristic;
    WatchdogFunctor _watchdog;

    std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
    std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
  };

  struct ExecutionSelector {
    bool _parallel;

    ExecutionSelector(bool parallel) : _parallel(parallel) {}

    template <typename Propagator> struct Select {
      template <typename... Args>
      Select(ExecutionSelector &This, Args... args) {
        if (This._parallel) {
          Propagator::template PushType<ParallelExecution>::Forward(args...);
        } else {
          Propagator::template PushType<SequentialExecution>::Forward(args...);
        }
      }
    };
  };

  struct TransitionModeSelector {
    PyMCTSOptions::TransitionMode _transition_mode;
    py::object &_domain; // for domain check

    TransitionModeSelector(PyMCTSOptions::TransitionMode transition_mode,
                           py::object &domain)
        : _transition_mode(transition_mode), _domain(domain) {}

    template <typename Propagator> struct Select {
      template <typename... Args>
      Select(TransitionModeSelector &This, Args... args) {
        switch (This._transition_mode) {
        case PyMCTSOptions::TransitionMode::Step:
          This.check_domain<PyMCTSOptions::TransitionMode::Step>();
          Propagator::template PushTemplate<StepTransitionMode>::Forward(
              args...);
          break;
        case PyMCTSOptions::TransitionMode::Sample:
          This.check_domain<PyMCTSOptions::TransitionMode::Sample>();
          Propagator::template PushTemplate<SampleTransitionMode>::Forward(
              args...);
          break;
        case PyMCTSOptions::TransitionMode::Distribution:
          This.check_domain<PyMCTSOptions::TransitionMode::Distribution>();
          Propagator::template PushTemplate<
              DistributionTransitionMode>::Forward(args...);
          break;
        default:
          Logger::error("Available transition modes: TransitionMode.Step , "
                        "TransitionMode.Sample , TransitionMode.Distribution");
          throw std::runtime_error(
              "Available transition modes: TransitionMode.Step , "
              "TransitionMode.Sample , TransitionMode.Distribution");
        }
      }
    };

    template <
        PyMCTSOptions::TransitionMode transition_mode,
        std::enable_if_t<transition_mode == PyMCTSOptions::TransitionMode::Step,
                         int> = 0>
    void check_domain() {
      if (!py::hasattr(_domain, "get_applicable_actions")) {
        throw std::invalid_argument(
            "SKDECIDE exception: MCTS algorithm needs python domain for "
            "implementing get_applicable_actions()");
      }
      if (!py::hasattr(_domain, "step")) {
        throw std::invalid_argument(
            "SKDECIDE exception: MCTS algorithm with step transition mode "
            "needs python domain for implementing step()");
      }
    }

    template <
        PyMCTSOptions::TransitionMode transition_mode,
        std::enable_if_t<
            transition_mode == PyMCTSOptions::TransitionMode::Sample, int> = 0>
    void check_domain() {
      if (!py::hasattr(_domain, "get_applicable_actions")) {
        throw std::invalid_argument(
            "SKDECIDE exception: MCTS algorithm needs python domain for "
            "implementing get_applicable_actions()");
      }
      if (!py::hasattr(_domain, "sample")) {
        throw std::invalid_argument(
            "SKDECIDE exception: MCTS algorithm with sample or distribution "
            "transition mode needs python domain for implementing sample()");
      }
    }

    template <PyMCTSOptions::TransitionMode transition_mode,
              std::enable_if_t<transition_mode ==
                                   PyMCTSOptions::TransitionMode::Distribution,
                               int> = 0>
    void check_domain() {
      if (!py::hasattr(_domain, "get_applicable_actions")) {
        throw std::invalid_argument(
            "SKDECIDE exception: MCTS algorithm needs python domain for "
            "implementing get_applicable_actions()");
      }
      if (!py::hasattr(_domain, "sample")) {
        throw std::invalid_argument(
            "SKDECIDE exception: MCTS algorithm with sample or distribution "
            "transition mode needs python domain for implementing sample()");
      }
      if (!py::hasattr(_domain, "get_next_state_distribution")) {
        throw std::invalid_argument(
            "SKDECIDE exception: MCTS algorithm with distribution transition "
            "mode needs python domain for implementing "
            "get_next_state_distribution()");
      }
      if (!py::hasattr(_domain, "get_transition_value")) {
        throw std::invalid_argument(
            "SKDECIDE exception: MCTS algorithm with distribution transition "
            "mode needs python domain for implementing get_transition_value()");
      }
      if (!py::hasattr(_domain, "is_terminal")) {
        throw std::invalid_argument(
            "SKDECIDE exception: MCTS algorithm with distribution transition "
            "mode needs python domain for implementing is_terminal()");
      }
    }
  };

  struct TreePolicySelector {
    PyMCTSOptions::TreePolicy _tree_policy;

    TreePolicySelector(PyMCTSOptions::TreePolicy tree_policy)
        : _tree_policy(tree_policy) {}

    template <typename Propagator> struct Select {
      template <typename... Args>
      Select(TreePolicySelector &This, Args... args) {
        switch (This._tree_policy) {
        case PyMCTSOptions::TreePolicy::Default:
          Propagator::template PushTemplate<DefaultTreePolicy>::Forward(
              args...);
          break;
        default:
          Logger::error("Available tree policies: TreePolicy.Default");
          throw std::runtime_error(
              "Available tree policies: TreePolicy.Default");
        }
      }
    };
  };

  struct ExpanderSelector {
    PyMCTSOptions::Expander _expander;

    ExpanderSelector(PyMCTSOptions::Expander expander) : _expander(expander) {}

    template <typename Propagator> struct Select {
      template <typename... Args> Select(ExpanderSelector &This, Args... args) {
        switch (This._expander) {
        case PyMCTSOptions::Expander::Full:
          Propagator::template PushTemplate<FullExpand>::Forward(args...);
          break;
        case PyMCTSOptions::Expander::Partial:
          Propagator::template PushTemplate<PartialExpand>::Forward(args...);
          break;
        default:
          Logger::error("Available expanders: Expander.Full, Expander.Partial");
          throw std::runtime_error(
              "Available expanders: Expander.Full, Expander.Partial");
        }
      }
    };
  };

  struct ActionSelector {
    PyMCTSOptions::ActionSelector _action_selector;

    ActionSelector(PyMCTSOptions::ActionSelector action_selector)
        : _action_selector(action_selector) {}

    template <typename Propagator> struct Select {
      template <typename... Args> Select(ActionSelector &This, Args... args) {
        switch (This._action_selector) {
        case PyMCTSOptions::ActionSelector::UCB1:
          Propagator::template PushTemplate<UCB1ActionSelector>::Forward(
              args...);
          break;
        case PyMCTSOptions::ActionSelector::BestQValue:
          Propagator::template PushTemplate<BestQValueActionSelector>::Forward(
              args...);
          break;
        default:
          Logger::error("Available action selector: ActionSelector.UCB1 , "
                        "ActionSelector.BestQValue");
          throw std::runtime_error(
              "Available action selector: ActionSelector.UCB1 , "
              "ActionSelector.BestQValue");
        }
      }
    };
  };

  struct RolloutPolicySelector {
    PyMCTSOptions::RolloutPolicy _rollout_policy;
    CustomPolicyFunctor &_custom_policy_functor;
    const HeuristicFunctor &_heuristic_functor;

    RolloutPolicySelector(PyMCTSOptions::RolloutPolicy rollout_policy,
                          CustomPolicyFunctor &custom_policy_functor,
                          const HeuristicFunctor &heuristic_functor)
        : _rollout_policy(rollout_policy),
          _custom_policy_functor(custom_policy_functor),
          _heuristic_functor(heuristic_functor) {}

    template <typename Propagator> struct Select {
      template <typename... Args>
      Select(RolloutPolicySelector &This, Args... args) {
        switch (This._rollout_policy) {
        case PyMCTSOptions::RolloutPolicy::Random:
          Propagator::template PushTemplate<DefaultRolloutPolicy>::Forward(
              args...);
          if (!This._custom_policy_functor) {
            Logger::warn("Requesting MCTS random rollout policy but providing "
                         "custom policy functor (will be ignored)");
          }
          This._custom_policy_functor = nullptr;
          break;
        case PyMCTSOptions::RolloutPolicy::Custom:
          Propagator::template PushTemplate<DefaultRolloutPolicy>::Forward(
              args...);
          if (!This._custom_policy_functor) {
            Logger::error("Requesting MCTS custom rollout policy but giving "
                          "null rollout policy functor");
            throw std::runtime_error(
                "Requesting MCTS custom rollout policy but providing null "
                "rollout policy functor");
          }
          break;
        case PyMCTSOptions::RolloutPolicy::Void:
          Propagator::template PushTemplate<VoidRolloutPolicy>::Forward(
              args...);
          if (!This._heuristic_functor) {
            Logger::warn("Requesting MCTS void rollout policy but giving "
                         "null heuristic functor: leaf node values will be "
                         "initialized and back-propagated using only "
                         "default values (e.g. 0)");
          }
          break;
        default:
          Logger::error("Available default policies: RolloutPolicy.Random, "
                        "RolloutPolicy.Custom, RolloutPolicy.Void");
          throw std::runtime_error(
              "Available default policies: RolloutPolicy.Random, "
              "RolloutPolicy.Custom");
        }
      }
    };
  };

  struct BackPropagatorSelector {
    PyMCTSOptions::BackPropagator _back_propagator;

    BackPropagatorSelector(PyMCTSOptions::BackPropagator back_propagator)
        : _back_propagator(back_propagator) {}

    template <typename Propagator> struct Select {
      template <typename... Args>
      Select(BackPropagatorSelector &This, Args... args) {
        switch (This._back_propagator) {
        case PyMCTSOptions::BackPropagator::Graph:
          Propagator::template PushTemplate<GraphBackup>::Forward(args...);
          break;
        default:
          Logger::error("Available back propagators: BackPropagator.Graph");
          throw std::runtime_error(
              "Available back propagators: BackPropagator.Graph");
        }
      }
    };
  };

  // Separate template instantiations in two parts in order to reduce
  // compilation effort. Otherwise we would test in one shot all the possible
  // combinations of all the template instantiations, which consumes too much
  // memory.
  struct PartialSolverInstantiator {
    std::unique_ptr<BaseImplementation> &_implementation;
    PyMCTSOptions::ActionSelector _action_selector_optimization;
    PyMCTSOptions::ActionSelector _action_selector_execution;
    PyMCTSOptions::RolloutPolicy _rollout_policy;
    PyMCTSOptions::BackPropagator _back_propagator;
    CustomPolicyFunctor &_custom_policy_functor;
    const HeuristicFunctor &_heuristic_functor;

    PartialSolverInstantiator(
        std::unique_ptr<BaseImplementation> &implementation,
        PyMCTSOptions::ActionSelector action_selector_optimization,
        PyMCTSOptions::ActionSelector action_selector_execution,
        PyMCTSOptions::RolloutPolicy rollout_policy,
        PyMCTSOptions::BackPropagator back_propagator,
        CustomPolicyFunctor &custom_policy_functor,
        const HeuristicFunctor &heuristic_functor)
        : _implementation(implementation),
          _action_selector_optimization(action_selector_optimization),
          _action_selector_execution(action_selector_execution),
          _rollout_policy(rollout_policy), _back_propagator(back_propagator),
          _custom_policy_functor(custom_policy_functor),
          _heuristic_functor(heuristic_functor) {}

    template <typename... TypeInstantiations> struct TypeList {
      template <template <typename...> class... TemplateInstantiations>
      struct TemplateList {
        struct Instantiate {
          Instantiate(PartialSolverInstantiator &This, MCTS_SOLVER_DECL_ARGS);
        };
      };
    };
  };

  // Separate template instantiations in two parts in order to reduce
  // compilation effort. Otherwise we would test in one shot all the possible
  // combinations of all the template instantiations, which consumes too much
  // memory.
  struct FullSolverInstantiator {
    template <typename... PartialTypeInstantiations> struct TypeList {
      template <template <typename...> class... PartialTemplateInstantiations>
      struct TemplateList {
        std::unique_ptr<BaseImplementation> &_implementation;

        TemplateList(std::unique_ptr<BaseImplementation> &implementation)
            : _implementation(implementation) {}

        template <template <typename...> class... TemplateInstantiations>
        struct Instantiate {
          Instantiate(TemplateList &This, MCTS_SOLVER_DECL_ARGS);
        };
      };
    };
  };

  std::unique_ptr<BaseImplementation> _implementation;
  CustomPolicyFunctor _filtered_custom_policy;

public:
  PyMCTSSolver(py::object &domain, std::size_t time_budget = 3600000,
               std::size_t rollout_budget = 100000,
               std::size_t max_depth = 1000,
               std::size_t epsilon_moving_average_window = 100,
               double epsilon = 0.0, // not a stopping criterion by default
               double discount = 1.0, bool uct_mode = true,
               double ucb_constant = 1.0 / std::sqrt(2.0),
               bool online_node_garbage = false,
               const CustomPolicyFunctor &custom_policy = nullptr,
               const HeuristicFunctor &heuristic = nullptr,
               double state_expansion_rate = 0.1,
               double action_expansion_rate = 0.1,
               PyMCTSOptions::TransitionMode transition_mode =
                   PyMCTSOptions::TransitionMode::Distribution,
               PyMCTSOptions::TreePolicy tree_policy =
                   PyMCTSOptions::TreePolicy::Default,
               PyMCTSOptions::Expander expander = PyMCTSOptions::Expander::Full,
               PyMCTSOptions::ActionSelector action_selector_optimization =
                   PyMCTSOptions::ActionSelector::UCB1,
               PyMCTSOptions::ActionSelector action_selector_execution =
                   PyMCTSOptions::ActionSelector::BestQValue,
               PyMCTSOptions::RolloutPolicy rollout_policy =
                   PyMCTSOptions::RolloutPolicy::Random,
               PyMCTSOptions::BackPropagator back_propagator =
                   PyMCTSOptions::BackPropagator::Graph,
               bool parallel = false, bool debug_logs = false,
               const WatchdogFunctor &watchdog = nullptr);

  void close() { _implementation->close(); }

  void clear() { _implementation->clear(); }

  void solve(const py::object &s) { _implementation->solve(s); }

  py::bool_ is_solution_defined_for(const py::object &s) {
    return _implementation->is_solution_defined_for(s);
  }

  py::object get_next_action(const py::object &s) {
    return _implementation->get_next_action(s);
  }

  py::float_ get_utility(const py::object &s) {
    return _implementation->get_utility(s);
  }

  py::int_ get_nb_of_explored_states() {
    return _implementation->get_nb_of_explored_states();
  }

  py::int_ get_nb_rollouts() { return _implementation->get_nb_rollouts(); }

  py::dict get_policy() { return _implementation->get_policy(); }

  py::list get_action_prefix() { return _implementation->get_action_prefix(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_MCTS_HH
