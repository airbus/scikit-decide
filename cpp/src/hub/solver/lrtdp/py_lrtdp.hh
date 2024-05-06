/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_LRTDP_HH
#define SKDECIDE_PY_LRTDP_HH

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "utils/execution.hh"
#include "utils/python_gil_control.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "lrtdp.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PyLRTDPDomain = PythonDomainProxy<Texecution>;

class PyLRTDPSolver {
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
    virtual py::float_ get_residual_moving_average() = 0;
    virtual py::int_ get_solving_time() = 0;
    virtual py::dict get_policy() = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    Implementation(
        py::object &solver, // Python solver wrapper
        py::object &domain,
        const std::function<py::object(py::object &, const py::object &,
                                       const py::object &)>
            &goal_checker, // last arg used for optional thread_id
        const std::function<py::object(py::object &, const py::object &,
                                       const py::object &)>
            &heuristic, // last arg used for optional thread_id
        bool use_labels = true, std::size_t time_budget = 3600000,
        std::size_t rollout_budget = 100000, std::size_t max_depth = 1000,
        std::size_t residual_moving_average_window = 100,
        double epsilon = 0.001, double discount = 1.0,
        bool online_node_garbage = false, bool debug_logs = false,
        const std::function<py::bool_(py::object &, const py::object &,
                                      const py::object &)> &callback = nullptr)
        : _goal_checker(goal_checker), _heuristic(heuristic),
          _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      check_domain(domain);
      _domain = std::make_unique<PyLRTDPDomain<Texecution>>(domain);
      _solver = std::make_unique<
          skdecide::LRTDPSolver<PyLRTDPDomain<Texecution>, Texecution>>(
          *_domain,
          [this](PyLRTDPDomain<Texecution> &d,
                 const typename PyLRTDPDomain<Texecution>::State &s,
                 const std::size_t *thread_id) ->
          typename PyLRTDPDomain<Texecution>::Predicate {
            try {
              std::unique_ptr<py::object> r =
                  d.call(thread_id, _goal_checker, s.pyobj());
              typename skdecide::GilControl<Texecution>::Acquire acquire;
              bool rr = r->template cast<bool>();
              r.reset();
              return rr;
            } catch (const std::exception &e) {
              Logger::error(
                  std::string(
                      "SKDECIDE exception when calling goal checker: ") +
                  e.what());
              throw;
            }
          },
          [this](PyLRTDPDomain<Texecution> &d,
                 const typename PyLRTDPDomain<Texecution>::State &s,
                 const std::size_t *thread_id) ->
          typename PyLRTDPDomain<Texecution>::Value {
            try {
              return typename PyLRTDPDomain<Texecution>::Value(
                  d.call(thread_id, _heuristic, s.pyobj()));
            } catch (const std::exception &e) {
              Logger::error(
                  std::string("SKDECIDE exception when calling heuristic: ") +
                  e.what());
              throw;
            }
          },
          use_labels, time_budget, rollout_budget, max_depth,
          residual_moving_average_window, epsilon, discount,
          online_node_garbage, debug_logs,
          [this](const skdecide::LRTDPSolver<PyLRTDPDomain<Texecution>,
                                             Texecution> &s,
                 PyLRTDPDomain<Texecution> &d,
                 const std::size_t *thread_id) -> bool {
            // we don't make use of the C++ solver object 's' from Python
            // but we rather use its Python wrapper 'solver'
            try {
              if (_callback) {
                std::unique_ptr<py::object> r =
                    d.call(thread_id, _callback, *_pysolver);
                typename skdecide::GilControl<Texecution>::Acquire acquire;
                bool rr = r->template cast<bool>();
                r.reset();
                return rr;
              } else {
                return false;
              }
            } catch (const std::exception &e) {
              Logger::error(
                  std::string(
                      "SKDECIDE exception when calling callback function: ") +
                  e.what());
              throw;
            }
          });
      _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(
          std::cout, py::module::import("sys").attr("stdout"));
      _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(
          std::cerr, py::module::import("sys").attr("stderr"));
    }

    virtual ~Implementation() {}

    void check_domain(py::object &domain) {
      if (!py::hasattr(domain, "get_applicable_actions")) {
        throw std::invalid_argument(
            "SKDECIDE exception: LRTDP algorithm needs python domain for "
            "implementing get_applicable_actions()");
      }
      if (!py::hasattr(domain, "get_next_state_distribution")) {
        throw std::invalid_argument(
            "SKDECIDE exception: LRTDP algorithm needs python domain for "
            "implementing get_next_state_distribution()");
      }
      if (!py::hasattr(domain, "get_transition_value")) {
        throw std::invalid_argument(
            "SKDECIDE exception: LRTDP algorithm needs python domain for "
            "implementing get_transition_value()");
      }
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

    virtual py::int_ get_nb_rollouts() { return _solver->get_nb_rollouts(); }

    virtual py::float_ get_residual_moving_average() {
      return _solver->get_residual_moving_average();
    }

    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }

    virtual py::dict get_policy() {
      py::dict d;
      auto &&p = _solver->get_policy();
      for (auto &e : p) {
        d[e.first.pyobj()] =
            py::make_tuple(e.second.first.pyobj(), e.second.second);
      }
      return d;
    }

  private:
    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<PyLRTDPDomain<Texecution>> _domain;
    std::unique_ptr<
        skdecide::LRTDPSolver<PyLRTDPDomain<Texecution>, Texecution>>
        _solver;

    std::function<py::object(py::object &, const py::object &,
                             const py::object &)>
        _goal_checker; // last arg used for optional thread_id
    std::function<py::object(py::object &, const py::object &,
                             const py::object &)>
        _heuristic; // last arg used for optional thread_id
    std::function<py::bool_(py::object &, const py::object &,
                            const py::object &)>
        _callback; // last arg used for optional thread_id

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
  PyLRTDPSolver(
      py::object &solver, // Python solver wrapper
      py::object &domain,
      const std::function<py::object(py::object &, const py::object &,
                                     const py::object & // last arg used for
                                                        // optional thread_id
                                     )> &goal_checker,
      const std::function<py::object(py::object &, const py::object &,
                                     const py::object & // last arg used for
                                                        // optional thread_id
                                     )> &heuristic,
      bool use_labels = true, std::size_t time_budget = 3600000,
      std::size_t rollout_budget = 100000, std::size_t max_depth = 1000,
      std::size_t residual_moving_average_window = 100, double epsilon = 0.001,
      double discount = 1.0, bool online_node_garbage = false,
      bool parallel = false, bool debug_logs = false,
      const std::function<py::bool_(py::object &, const py::object &,
                                    const py::object & // last arg used for
                                                       // optional thread_id
                                    )> &callback = nullptr) {

    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, goal_checker, heuristic, use_labels,
                     time_budget, rollout_budget, max_depth,
                     residual_moving_average_window, epsilon, discount,
                     online_node_garbage, debug_logs, callback);
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

  py::int_ get_nb_rollouts() { return _implementation->get_nb_rollouts(); }

  py::float_ get_residual_moving_average() {
    return _implementation->get_residual_moving_average();
  }

  py::int_ get_solving_time() { return _implementation->get_solving_time(); }

  py::dict get_policy() { return _implementation->get_policy(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_LRTDP_HH
