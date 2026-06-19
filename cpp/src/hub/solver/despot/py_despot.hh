/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_DESPOT_HH
#define SKDECIDE_PY_DESPOT_HH

#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

#include "utils/execution.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/python_gil_control.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "despot.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PyDespotDomain =
    PythonDomainProxy<Texecution, SingleAgent, PartiallyObservable>;

class PyDespotSolver {
private:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void close() = 0;
    virtual void clear() = 0;
    virtual void solve(const py::object &distribution) = 0;
    virtual py::object get_next_action(const py::object &o) = 0;
    virtual py::object get_utility(const py::object &o) = 0;
    virtual py::bool_ is_solution_defined_for(const py::object &o) = 0;
    virtual void reset_belief() = 0;
    virtual py::object get_next_action_from_belief(const py::object &d) = 0;
    virtual py::object get_utility_from_belief(const py::object &d) = 0;
    virtual py::bool_
    is_solution_defined_for_from_belief(const py::object &d) = 0;
    virtual py::int_ get_nb_tree_nodes() = 0;
    virtual py::int_ get_solving_time() = 0;
    virtual py::float_ get_gap() = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    Implementation(
        py::object &solver, py::object &domain, std::size_t num_scenarios = 500,
        std::size_t max_depth = 90, double regularization_constant = 0.0,
        double gap_reduction_rate = 0.95, double target_gap = 0.0,
        std::size_t time_budget = 1000, double discount = 0.95,
        std::size_t max_rollout_depth = 90,
        std::size_t num_particles_belief_update = 500,
        const std::function<py::object(const py::object &, const py::object &)>
            &default_policy = nullptr,
        const std::function<py::object(const py::object &, const py::object &)>
            &upper_bound_heuristic = nullptr,
        const std::function<py::bool_(const py::object &, const py::object &)>
            &callback = nullptr,
        bool verbose = false)
        : _default_policy_py(default_policy),
          _upper_bound_heuristic_py(upper_bound_heuristic),
          _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      _domain = std::make_unique<PyDespotDomain<Texecution>>(domain);

      // Wrap Python default policy functor
      typename DespotSolver<PyDespotDomain<Texecution>,
                            Texecution>::DefaultPolicyFunctor
          default_policy_cpp = nullptr;
      if (_default_policy_py) {
        default_policy_cpp =
            [this](PyDespotDomain<Texecution> &d,
                   const typename PyDespotDomain<Texecution>::State &s,
                   const std::size_t *thread_id) ->
            typename PyDespotDomain<Texecution>::Value {
              auto fh = [this](const py::object &dd, const py::object &ss,
                               [[maybe_unused]] const py::object &ii) {
                return _default_policy_py(dd, ss);
              };
              return typename PyDespotDomain<Texecution>::Value(
                  d.call(thread_id, fh, s.pyobj()));
            };
      }

      // Wrap Python upper bound heuristic functor
      typename DespotSolver<PyDespotDomain<Texecution>,
                            Texecution>::UpperBoundFunctor upper_bound_cpp =
          nullptr;
      if (_upper_bound_heuristic_py) {
        upper_bound_cpp =
            [this](PyDespotDomain<Texecution> &d,
                   const typename PyDespotDomain<Texecution>::State &s,
                   const std::size_t *thread_id) ->
            typename PyDespotDomain<Texecution>::Value {
              auto fh = [this](const py::object &dd, const py::object &ss,
                               [[maybe_unused]] const py::object &ii) {
                return _upper_bound_heuristic_py(dd, ss);
              };
              return typename PyDespotDomain<Texecution>::Value(
                  d.call(thread_id, fh, s.pyobj()));
            };
      }

      _solver = std::make_unique<
          DespotSolver<PyDespotDomain<Texecution>, Texecution>>(
          *_domain, num_scenarios, max_depth, regularization_constant,
          gap_reduction_rate, target_gap, time_budget, discount,
          max_rollout_depth, num_particles_belief_update, default_policy_cpp,
          upper_bound_cpp,
          [this](const DespotSolver<PyDespotDomain<Texecution>, Texecution> &s,
                 PyDespotDomain<Texecution> &d,
                 const std::size_t *thread_id) -> bool {
            if (_callback) {
              std::unique_ptr<py::bool_> r;
              typename skdecide::GilControl<Texecution>::Acquire acquire;
              try {
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
              } catch (const py::error_already_set *e) {
                Logger::error(std::string("SKDECIDE exception when calling "
                                          "callback function: ") +
                              e->what());
                std::runtime_error err(e->what());
                r.reset();
                delete e;
                throw err;
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
      std::vector<std::pair<typename PyDespotDomain<Texecution>::State, double>>
          dist;
      py::list values = distribution.attr("get_values")();
      for (auto item : values) {
        py::tuple t = item.cast<py::tuple>();
        dist.emplace_back(typename PyDespotDomain<Texecution>::State(t[0]),
                          t[1].cast<double>());
      }
      typename GilControl<Texecution>::Release release;
      _solver->solve(dist);
    }

    virtual py::object get_next_action(const py::object &o) {
      try {
        typename PyDespotDomain<Texecution>::Observation obs(o);
        const typename PyDespotDomain<Texecution>::Action *action_ptr;
        {
          typename GilControl<Texecution>::Release release;
          action_ptr = &_solver->get_best_action(obs);
        }
        return action_ptr->pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[DESPOT.get_next_action] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility(const py::object &o) {
      try {
        typename PyDespotDomain<Texecution>::Observation obs(o);
        typename PyDespotDomain<Texecution>::Value val;
        {
          typename GilControl<Texecution>::Release release;
          val = _solver->get_best_value(obs);
        }
        return val.pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[DESPOT.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::bool_ is_solution_defined_for(const py::object &o) {
      return _solver->is_solution_defined_for(
          typename PyDespotDomain<Texecution>::Observation(o));
    }

    virtual void reset_belief() { _solver->reset_belief(); }

    virtual py::object get_next_action_from_belief(const py::object &d) {
      try {
        auto belief = distribution_to_belief(d);
        const typename PyDespotDomain<Texecution>::Action *action_ptr;
        {
          typename GilControl<Texecution>::Release release;
          action_ptr = &_solver->get_best_action_from_belief(belief);
        }
        return action_ptr->pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[DESPOT.get_next_action_from_belief] ") +
                     e.what() + " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility_from_belief(const py::object &d) {
      try {
        auto belief = distribution_to_belief(d);
        typename PyDespotDomain<Texecution>::Value val;
        {
          typename GilControl<Texecution>::Release release;
          val = _solver->get_best_value_from_belief(belief);
        }
        return val.pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[DESPOT.get_utility_from_belief] ") +
                     e.what() + " - returning None");
        return py::none();
      }
    }

    virtual py::bool_ is_solution_defined_for_from_belief(const py::object &d) {
      auto belief = distribution_to_belief(d);
      return _solver->is_solution_defined_for_from_belief(belief);
    }

    virtual py::int_ get_nb_tree_nodes() {
      return _solver->get_nb_tree_nodes();
    }

    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }

    virtual py::float_ get_gap() { return _solver->get_gap(); }

  private:
    typedef DespotSolver<PyDespotDomain<Texecution>, Texecution> SolverType;

    typename SolverType::Belief distribution_to_belief(const py::object &d) {
      typename SolverType::Belief b;
      py::list values = d.attr("get_values")();
      for (auto item : values) {
        py::tuple t = item.cast<py::tuple>();
        typename PyDespotDomain<Texecution>::State state(t[0]);
        std::size_t idx = _solver->get_state_index(state);
        b[idx] = t[1].cast<double>();
      }
      return b;
    }

    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<PyDespotDomain<Texecution>> _domain;
    std::unique_ptr<DespotSolver<PyDespotDomain<Texecution>, Texecution>>
        _solver;

    std::function<py::object(const py::object &, const py::object &)>
        _default_policy_py;
    std::function<py::object(const py::object &, const py::object &)>
        _upper_bound_heuristic_py;
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
  PyDespotSolver(
      py::object &solver, py::object &domain, std::size_t num_scenarios = 500,
      std::size_t max_depth = 90, double regularization_constant = 0.0,
      double gap_reduction_rate = 0.95, double target_gap = 0.0,
      std::size_t time_budget = 1000, double discount = 0.95,
      std::size_t max_rollout_depth = 90,
      std::size_t num_particles_belief_update = 500,
      const std::function<py::object(const py::object &, const py::object &)>
          &default_policy = nullptr,
      const std::function<py::object(const py::object &, const py::object &)>
          &upper_bound_heuristic = nullptr,
      bool parallel = false,
      const std::function<py::bool_(const py::object &, const py::object &)>
          &callback = nullptr,
      bool verbose = false) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, num_scenarios, max_depth,
                     regularization_constant, gap_reduction_rate, target_gap,
                     time_budget, discount, max_rollout_depth,
                     num_particles_belief_update, default_policy,
                     upper_bound_heuristic, callback, verbose);
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

  void reset_belief() { _implementation->reset_belief(); }

  py::object get_next_action_from_belief(const py::object &d) {
    return _implementation->get_next_action_from_belief(d);
  }

  py::object get_utility_from_belief(const py::object &d) {
    return _implementation->get_utility_from_belief(d);
  }

  py::bool_ is_solution_defined_for_from_belief(const py::object &d) {
    return _implementation->is_solution_defined_for_from_belief(d);
  }

  py::int_ get_nb_tree_nodes() { return _implementation->get_nb_tree_nodes(); }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
  py::float_ get_gap() { return _implementation->get_gap(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_DESPOT_HH
