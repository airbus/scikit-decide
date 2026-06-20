/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_RTDP_BEL_HH
#define SKDECIDE_PY_RTDP_BEL_HH

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "utils/execution.hh"
#include "utils/python_gil_control.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "rtdp_bel.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PyRTDPBelDomain =
    PythonDomainProxy<Texecution, SingleAgent, PartiallyObservable>;

class PyRTDPBelSolver {
private:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void close() = 0;
    virtual void clear() = 0;
    virtual void solve(const py::object &s) = 0;
    // Observation-based interface
    virtual py::object get_next_action(const py::object &o) = 0;
    virtual py::object get_utility(const py::object &o) = 0;
    virtual py::bool_ is_solution_defined_for(const py::object &o) = 0;
    virtual py::tuple get_policy(const py::object &o) = 0;
    virtual void reset_belief() = 0;
    // Statistics
    virtual py::int_ get_nb_explored_beliefs() = 0;
    virtual py::list get_explored_beliefs() = 0;
    virtual py::int_ get_nb_rollouts() = 0;
    virtual py::int_ get_solving_time() = 0;
    virtual py::list get_last_trajectory() = 0;
    // Belief-state policy accessor
    virtual py::dict get_belief_policy() = 0;
    // Belief-state query interface
    virtual py::object get_next_action_from_belief(const py::object &d) = 0;
    virtual py::object get_utility_from_belief(const py::object &d) = 0;
    virtual py::bool_
    is_solution_defined_for_from_belief(const py::object &d) = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    Implementation(
        py::object &solver, py::object &domain,
        const std::function<py::object(const py::object &, const py::object &)>
            &goal_checker,
        const std::function<py::object(const py::object &, const py::object &)>
            &heuristic,
        std::size_t discretization = 10, std::size_t time_budget = 3600000,
        std::size_t rollout_budget = 100000, std::size_t max_depth = 1000,
        double epsilon = 0.001, double discount = 1.0,
        const std::function<py::bool_(const py::object &, const py::object &)>
            &callback = nullptr,
        bool verbose = false)
        : _goal_checker(goal_checker), _heuristic(heuristic),
          _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      _domain = std::make_unique<PyRTDPBelDomain<Texecution>>(domain);
      _solver = std::make_unique<
          RTDPBelSolver<PyRTDPBelDomain<Texecution>, Texecution>>(
          *_domain,
          [this](PyRTDPBelDomain<Texecution> &d,
                 const typename PyRTDPBelDomain<Texecution>::State &s,
                 const std::size_t *thread_id) ->
          typename PyRTDPBelDomain<Texecution>::Predicate {
            try {
              auto fgc = [this](const py::object &dd, const py::object &ss,
                                [[maybe_unused]] const py::object &ii) {
                return _goal_checker(dd, ss);
              };
              std::unique_ptr<py::object> r = d.call(thread_id, fgc, s.pyobj());
              typename GilControl<Texecution>::Acquire acquire;
              bool rr = r->template cast<bool>();
              r.reset();
              return rr;
            } catch (const std::exception &e) {
              Logger::error(std::string("SKDECIDE exception when calling goal "
                                        "checker: ") +
                            e.what());
              throw;
            }
          },
          [this](PyRTDPBelDomain<Texecution> &d,
                 const typename PyRTDPBelDomain<Texecution>::State &s,
                 const std::size_t *thread_id) ->
          typename PyRTDPBelDomain<Texecution>::Value {
            try {
              auto fh = [this](const py::object &dd, const py::object &ss,
                               [[maybe_unused]] const py::object &ii) {
                return _heuristic(dd, ss);
              };
              return typename PyRTDPBelDomain<Texecution>::Value(
                  d.call(thread_id, fh, s.pyobj()));
            } catch (const std::exception &e) {
              Logger::error(
                  std::string("SKDECIDE exception when calling heuristic: ") +
                  e.what());
              throw;
            }
          },
          discretization, time_budget, rollout_budget, max_depth, epsilon,
          discount,
          [this](
              const RTDPBelSolver<PyRTDPBelDomain<Texecution>, Texecution> &s,
              PyRTDPBelDomain<Texecution> &d,
              const std::size_t *thread_id) -> bool {
            if (_callback) {
              try {
                std::unique_ptr<py::bool_> r;
                typename GilControl<Texecution>::Acquire acquire;
                if (thread_id) {
                  r = std::make_unique<py::bool_>(
                      _callback(*_pysolver, py::int_(*thread_id)));
                } else {
                  r = std::make_unique<py::bool_>(
                      _callback(*_pysolver, py::none()));
                }
                bool rr = r->template cast<bool>();
                r.reset();
                return rr;
              } catch (const std::exception &e) {
                Logger::error(std::string("SKDECIDE exception when calling "
                                          "callback: ") +
                              e.what());
                throw;
              }
            }
            return false;
          },
          verbose);
      _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(
          std::cout, py::module::import("sys").attr("stdout"));
      _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(
          std::cerr, py::module::import("sys").attr("stderr"));
    }

    virtual ~Implementation() {}

    virtual void close() { _domain->close(); }
    virtual void clear() { _solver->clear(); }

    virtual void solve(const py::object &distribution) {
      std::vector<
          std::pair<typename PyRTDPBelDomain<Texecution>::State, double>>
          dist;
      py::list values = distribution.attr("get_values")();
      for (auto item : values) {
        py::tuple t = item.cast<py::tuple>();
        dist.emplace_back(typename PyRTDPBelDomain<Texecution>::State(t[0]),
                          t[1].cast<double>());
      }
      typename GilControl<Texecution>::Release release;
      _solver->solve(dist);
    }

    virtual py::object get_next_action(const py::object &o) {
      try {
        typename PyRTDPBelDomain<Texecution>::Observation obs(o);
        const typename PyRTDPBelDomain<Texecution>::Action *action_ptr;
        {
          typename GilControl<Texecution>::Release release;
          action_ptr = &_solver->get_best_action(obs);
        }
        return action_ptr->pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[RTDPBel.get_next_action] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility(const py::object &o) {
      try {
        typename PyRTDPBelDomain<Texecution>::Observation obs(o);
        typename PyRTDPBelDomain<Texecution>::Value val;
        {
          typename GilControl<Texecution>::Release release;
          val = _solver->get_best_value(obs);
        }
        return val.pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[RTDPBel.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::bool_ is_solution_defined_for(const py::object &o) {
      typename PyRTDPBelDomain<Texecution>::Observation obs(o);
      bool result;
      {
        typename GilControl<Texecution>::Release release;
        result = _solver->is_solution_defined_for(obs);
      }
      return result;
    }

    virtual py::tuple get_policy(const py::object &o) {
      try {
        typename PyRTDPBelDomain<Texecution>::Observation obs(o);
        std::pair<typename PyRTDPBelDomain<Texecution>::Action, double> p;
        {
          typename GilControl<Texecution>::Release release;
          p = _solver->get_policy(obs);
        }
        return py::make_tuple(p.first.pyobj(), p.second);
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[RTDPBel.get_policy] ") + e.what() +
                     " - returning None");
        return py::make_tuple(py::none(), 0.0);
      }
    }

    virtual void reset_belief() { _solver->reset_belief(); }

    virtual py::int_ get_nb_explored_beliefs() {
      return _solver->get_nb_explored_beliefs();
    }

    virtual py::list get_explored_beliefs() {
      py::list result;
      const auto &graph = _solver->get_belief_graph();
      const auto &idx_map = _solver->get_index_to_state();
      for (const auto &entry : graph) {
        py::dict belief_dict;
        for (const auto &bp : entry.second->belief) {
          auto it = idx_map.find(bp.first);
          if (it != idx_map.end()) {
            belief_dict[it->second.pyobj()] = bp.second;
          }
        }
        result.append(belief_dict);
      }
      return result;
    }

    virtual py::int_ get_nb_rollouts() { return _solver->get_nb_rollouts(); }

    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }

    virtual py::list get_last_trajectory() {
      py::list result;
      auto &&trajectory = _solver->get_last_trajectory();
      const auto &idx_map = _solver->get_index_to_state();
      for (const auto &belief : trajectory) {
        py::dict belief_dict;
        for (const auto &bp : belief) {
          auto it = idx_map.find(bp.first);
          if (it != idx_map.end()) {
            belief_dict[it->second.pyobj()] = bp.second;
          }
        }
        result.append(belief_dict);
      }
      return result;
    }

    virtual py::dict get_belief_policy() {
      py::dict result;
      auto &&p = _solver->get_belief_policy();
      const auto &idx_map = _solver->get_index_to_state();
      const auto &graph = _solver->get_belief_graph();
      for (const auto &entry : p) {
        const auto &db = entry.first;
        // Build frozenset key from discretized belief: {(state, prob), ...}
        // Look up the continuous belief from the graph for probabilities
        auto git = graph.find(db);
        if (git == graph.end())
          continue;
        py::set belief_set;
        for (const auto &bp : git->second->belief) {
          auto it = idx_map.find(bp.first);
          if (it != idx_map.end()) {
            belief_set.add(py::make_tuple(it->second.pyobj(), bp.second));
          }
        }
        py::frozenset key(belief_set);
        result[key] = py::make_tuple(entry.second.first.pyobj(),
                                     entry.second.second.pyobj());
      }
      return result;
    }

    virtual py::object get_next_action_from_belief(const py::object &d) {
      try {
        auto belief = distribution_to_belief(d);
        const typename PyRTDPBelDomain<Texecution>::Action *action_ptr;
        {
          typename GilControl<Texecution>::Release release;
          action_ptr = &_solver->get_best_action_from_belief(belief);
        }
        return action_ptr->pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[RTDPBel.get_next_action_from_belief] ") +
                     e.what() + " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility_from_belief(const py::object &d) {
      try {
        auto belief = distribution_to_belief(d);
        typename PyRTDPBelDomain<Texecution>::Value val;
        {
          typename GilControl<Texecution>::Release release;
          val = _solver->get_best_value_from_belief(belief);
        }
        return val.pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[RTDPBel.get_utility_from_belief] ") +
                     e.what() + " - returning None");
        return py::none();
      }
    }

    virtual py::bool_ is_solution_defined_for_from_belief(const py::object &d) {
      auto belief = distribution_to_belief(d);
      bool result;
      {
        typename GilControl<Texecution>::Release release;
        result = _solver->is_solution_defined_for_from_belief(belief);
      }
      return result;
    }

  private:
    typedef RTDPBelSolver<PyRTDPBelDomain<Texecution>, Texecution> SolverType;

    typename SolverType::Belief distribution_to_belief(const py::object &d) {
      typename SolverType::Belief b;
      py::list values = d.attr("get_values")();
      for (auto item : values) {
        py::tuple t = item.cast<py::tuple>();
        typename PyRTDPBelDomain<Texecution>::State state(t[0]);
        std::size_t idx = _solver->get_state_index(state);
        b[idx] = t[1].cast<double>();
      }
      return b;
    }

    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<PyRTDPBelDomain<Texecution>> _domain;
    std::unique_ptr<RTDPBelSolver<PyRTDPBelDomain<Texecution>, Texecution>>
        _solver;

    std::function<py::object(const py::object &, const py::object &)>
        _goal_checker;
    std::function<py::object(const py::object &, const py::object &)>
        _heuristic;
    std::function<py::bool_(const py::object &, const py::object &)> _callback;

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

  struct SolverInstantiator {
    std::unique_ptr<BaseImplementation> &_implementation;
    SolverInstantiator(std::unique_ptr<BaseImplementation> &implementation)
        : _implementation(implementation) {}
    template <typename... TypeInstantiations> struct Instantiate {
      template <typename... Args>
      Instantiate(SolverInstantiator &This, Args... args) {
        This._implementation =
            std::make_unique<Implementation<TypeInstantiations...>>(args...);
      }
    };
  };

  std::unique_ptr<BaseImplementation> _implementation;

public:
  PyRTDPBelSolver(
      py::object &solver, py::object &domain,
      const std::function<py::object(const py::object &, const py::object &)>
          &goal_checker,
      const std::function<py::object(const py::object &, const py::object &)>
          &heuristic,
      std::size_t discretization = 10, std::size_t time_budget = 3600000,
      std::size_t rollout_budget = 100000, std::size_t max_depth = 1000,
      double epsilon = 0.001, double discount = 1.0, bool parallel = false,
      const std::function<py::bool_(const py::object &, const py::object &)>
          &callback = nullptr,
      bool verbose = false) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, goal_checker, heuristic, discretization,
                     time_budget, rollout_budget, max_depth, epsilon, discount,
                     callback, verbose);
  }

  void close() { _implementation->close(); }
  void clear() { _implementation->clear(); }
  void solve(const py::object &s) { _implementation->solve(s); }

  py::object get_next_action(const py::object &o) {
    return _implementation->get_next_action(o);
  }

  py::object get_utility(const py::object &o) {
    return _implementation->get_utility(o);
  }

  py::bool_ is_solution_defined_for(const py::object &o) {
    return _implementation->is_solution_defined_for(o);
  }

  py::tuple get_policy(const py::object &o) {
    return _implementation->get_policy(o);
  }

  void reset_belief() { _implementation->reset_belief(); }

  py::int_ get_nb_explored_beliefs() {
    return _implementation->get_nb_explored_beliefs();
  }

  py::list get_explored_beliefs() {
    return _implementation->get_explored_beliefs();
  }

  py::int_ get_nb_rollouts() { return _implementation->get_nb_rollouts(); }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
  py::list get_last_trajectory() {
    return _implementation->get_last_trajectory();
  }
  py::dict get_belief_policy() { return _implementation->get_belief_policy(); }

  py::object get_next_action_from_belief(const py::object &d) {
    return _implementation->get_next_action_from_belief(d);
  }

  py::object get_utility_from_belief(const py::object &d) {
    return _implementation->get_utility_from_belief(d);
  }

  py::bool_ is_solution_defined_for_from_belief(const py::object &d) {
    return _implementation->is_solution_defined_for_from_belief(d);
  }
};

} // namespace skdecide

#endif // SKDECIDE_PY_RTDP_BEL_HH
