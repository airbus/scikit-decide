/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_RIW_HH
#define SKDECIDE_PY_RIW_HH

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "utils/execution.hh"
#include "utils/python_gil_control.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/python_container_proxy.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "riw.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PyRIWDomain = PythonDomainProxy<Texecution>;

template <typename Texecution>
using PyRIWFeatureVector = PythonContainerProxy<Texecution>;

class PyRIWSolver {
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
    virtual py::int_ get_nb_of_pruned_states() = 0;
    virtual py::int_ get_nb_rollouts() = 0;
    virtual py::dict get_policy() = 0;
    virtual py::list get_action_prefix() = 0;
  };

  template <typename Texecution, template <typename...> class Thashing_policy,
            template <typename...> class Trollout_policy>
  class Implementation : public BaseImplementation {
  public:
    Implementation(
        py::object &domain,
        const std::function<py::object(py::object &, const py::object &,
                                       const py::object &)>
            &state_features, // last arg used for optional thread_id
        std::size_t time_budget = 3600000, std::size_t rollout_budget = 100000,
        std::size_t max_depth = 1000, double exploration = 0.25,
        std::size_t epsilon_moving_average_window = 100, double epsilon = 0.001,
        double discount = 1.0, bool online_node_garbage = false,
        bool debug_logs = false,
        const std::function<bool(const std::size_t &, const std::size_t &,
                                 const double &, const double &)> &watchdog =
            nullptr)
        : _state_features(state_features), _watchdog(watchdog) {

      check_domain(domain);
      _domain = std::make_unique<PyRIWDomain<Texecution>>(domain);
      _solver = std::make_unique<
          RIWSolver<PyRIWDomain<Texecution>, PyRIWFeatureVector<Texecution>,
                    Thashing_policy, Trollout_policy, Texecution>>(
          *_domain,
          [this](PyRIWDomain<Texecution> &d,
                 const typename PyRIWDomain<Texecution>::State &s,
                 const std::size_t *thread_id)
              -> std::unique_ptr<PyRIWFeatureVector<Texecution>> {
            try {
              std::unique_ptr<py::object> r =
                  d.call(thread_id, _state_features, s.pyobj());
              typename GilControl<Texecution>::Acquire acquire;
              std::unique_ptr<PyRIWFeatureVector<Texecution>> rr =
                  std::make_unique<PyRIWFeatureVector<Texecution>>(*r);
              r.reset();
              return rr;
            } catch (const std::exception &e) {
              Logger::error(
                  std::string(
                      "SKDECIDE exception when calling state features: ") +
                  e.what());
              throw;
            }
          },
          time_budget, rollout_budget, max_depth, exploration,
          epsilon_moving_average_window, epsilon, discount, online_node_garbage,
          debug_logs,
          [this](const std::size_t &elapsed_time,
                 const std::size_t &nb_rollouts, const double &best_value,
                 const double &epsilon_moving_average) -> bool {
            if (_watchdog) {
              typename skdecide::GilControl<Texecution>::Acquire acquire;
              return _watchdog(elapsed_time, nb_rollouts, best_value,
                               epsilon_moving_average);
            } else {
              return true;
            }
          });
      _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(
          std::cout, py::module::import("sys").attr("stdout"));
      _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(
          std::cerr, py::module::import("sys").attr("stderr"));
    }

    virtual ~Implementation() {}

    template <
        typename TTrollout_policy = Trollout_policy<PyRIWDomain<Texecution>>,
        std::enable_if_t<
            std::is_same<TTrollout_policy,
                         SimulationRollout<PyRIWDomain<Texecution>>>::value,
            int> = 0>
    void check_domain(py::object &domain) {
      if (!py::hasattr(domain, "get_applicable_actions")) {
        throw std::invalid_argument(
            "SKDECIDE exception: RIW algorithm needs python domain for "
            "implementing get_applicable_actions()");
      }
      if (!py::hasattr(domain, "sample")) {
        throw std::invalid_argument(
            "SKDECIDE exception: RIW algorithm needs python domain for "
            "implementing sample() in simulation mode");
      }
    }

    template <
        typename TTrollout_policy = Trollout_policy<PyRIWDomain<Texecution>>,
        std::enable_if_t<
            std::is_same<TTrollout_policy,
                         EnvironmentRollout<PyRIWDomain<Texecution>>>::value,
            int> = 0>
    void check_domain(py::object &domain) {
      if (!py::hasattr(domain, "get_applicable_actions")) {
        throw std::invalid_argument(
            "SKDECIDE exception: RIW algorithm needs python domain for "
            "implementing get_applicable_actions()");
      }
      if (!py::hasattr(domain, "reset")) {
        throw std::invalid_argument(
            "SKDECIDE exception: RIW algorithm needs python domain for "
            "implementing reset() in environment mode");
      }
      if (!py::hasattr(domain, "step")) {
        throw std::invalid_argument(
            "SKDECIDE exception: RIW algorithm needs python domain for "
            "implementing step() in environment mode");
      }
    }

    virtual void close() { _domain->close(); }

    virtual void clear() { _solver->clear(); }

    virtual void solve(const py::object &s) {
      typename GilControl<Texecution>::Release release;
      _solver->solve(s);
    }

    virtual py::bool_ is_solution_defined_for(const py::object &s) {
      return _solver->is_solution_defined_for(s);
    }

    virtual py::object get_next_action(const py::object &s) {
      try {
        return _solver->get_best_action(s).pyobj();
      } catch (const std::runtime_error &) {
        return py::none();
      }
    }

    virtual py::float_ get_utility(const py::object &s) {
      try {
        return _solver->get_best_value(s);
      } catch (const std::runtime_error &) {
        return py::none();
      }
    }

    virtual py::int_ get_nb_of_explored_states() {
      return _solver->get_nb_of_explored_states();
    }

    virtual py::int_ get_nb_of_pruned_states() {
      return _solver->get_nb_of_pruned_states();
    }

    virtual py::int_ get_nb_rollouts() { return _solver->get_nb_rollouts(); }

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
    std::unique_ptr<PyRIWDomain<Texecution>> _domain;
    std::unique_ptr<
        RIWSolver<PyRIWDomain<Texecution>, PyRIWFeatureVector<Texecution>,
                  Thashing_policy, Trollout_policy, Texecution>>
        _solver;

    std::function<py::object(py::object &, const py::object &,
                             const py::object &)>
        _state_features; // last arg used for optional thread_id
    std::function<bool(const std::size_t &, const std::size_t &, const double &,
                       const double &)>
        _watchdog;

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

  struct HashingPolicySelector {
    bool _use_state_feature_hash;

    HashingPolicySelector(bool use_state_feature_hash)
        : _use_state_feature_hash(use_state_feature_hash) {}

    template <typename Propagator> struct Select {
      template <typename... Args>
      Select(HashingPolicySelector &This, Args... args) {
        if (This._use_state_feature_hash) {
          Propagator::template PushTemplate<StateFeatureHash>::Forward(args...);
        } else {
          Propagator::template PushTemplate<DomainStateHash>::Forward(args...);
        }
      }
    };
  };

  struct RolloutPolicySelector {
    bool _use_simulation_domain;

    RolloutPolicySelector(bool use_simulation_domain)
        : _use_simulation_domain(use_simulation_domain) {}

    template <typename Propagator> struct Select {
      template <typename... Args>
      Select(RolloutPolicySelector &This, Args... args) {
        if (This._use_simulation_domain) {
          Propagator::template PushTemplate<SimulationRollout>::Forward(
              args...);
        } else {
          Propagator::template PushTemplate<EnvironmentRollout>::Forward(
              args...);
        }
      }
    };
  };

  struct SolverInstantiator {
    std::unique_ptr<BaseImplementation> &_implementation;

    SolverInstantiator(std::unique_ptr<BaseImplementation> &implementation)
        : _implementation(implementation) {}

    template <typename... TypeInstantiations> struct TypeList {
      template <template <typename...> class... TemplateInstantiations>
      struct TemplateList {
        struct Instantiate {
          template <typename... Args>
          Instantiate(SolverInstantiator &This, Args... args) {
            This._implementation = std::make_unique<Implementation<
                TypeInstantiations..., TemplateInstantiations...>>(args...);
          }
        };
      };
    };
  };

  std::unique_ptr<BaseImplementation> _implementation;

public:
  PyRIWSolver(
      py::object &domain,
      const std::function<py::object(py::object &, const py::object &,
                                     const py::object &)>
          &state_features, // last arg used for optional thread_id
      bool use_state_feature_hash = false, bool use_simulation_domain = false,
      std::size_t time_budget = 3600000, std::size_t rollout_budget = 100000,
      std::size_t max_depth = 1000, double exploration = 0.25,
      std::size_t epsilon_moving_average_window = 100, double epsilon = 0.001,
      double discount = 1.0, bool online_node_garbage = false,
      bool parallel = false, bool debug_logs = false,
      const std::function<bool(const std::size_t &, const std::size_t &,
                               const double &, const double &)> &watchdog =
          nullptr) {

    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 HashingPolicySelector(use_state_feature_hash),
                                 RolloutPolicySelector(use_simulation_domain),
                                 SolverInstantiator(_implementation))
        .instantiate(domain, state_features, time_budget, rollout_budget,
                     max_depth, exploration, epsilon_moving_average_window,
                     epsilon, discount, online_node_garbage, debug_logs,
                     watchdog);
  }

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

  py::int_ get_nb_of_pruned_states() {
    return _implementation->get_nb_of_pruned_states();
  }

  py::int_ get_nb_rollouts() { return _implementation->get_nb_rollouts(); }

  py::dict get_policy() { return _implementation->get_policy(); }

  py::list get_action_prefix() { return _implementation->get_action_prefix(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_RIW_HH
