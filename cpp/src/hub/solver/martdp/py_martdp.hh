/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_MARTDP_HH
#define SKDECIDE_PY_MARTDP_HH

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "utils/execution.hh"
#include "utils/python_gil_control.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "martdp.hh"

namespace py = pybind11;

namespace skdecide {

// we pretend the domain can be sequential or parallel even if the current
// implementation of MARTDP is sequential, because it will ease the python
// wrapper implementation if MARTDP is made parallel in the future
template <typename Texecution>
using PyMARTDPDomain = PythonDomainProxy<Texecution, skdecide::MultiAgent>;

class PyMARTDPSolver {
private:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void close() = 0;
    virtual void clear() = 0;
    virtual void solve(const py::object &s) = 0;
    virtual py::bool_ is_solution_defined_for(const py::object &s) = 0;
    virtual py::object get_next_action(const py::object &s) = 0;
    virtual py::object get_utility(const py::object &s) = 0;
    virtual py::int_ get_nb_explored_states() = 0;
    virtual py::int_ get_state_nb_actions(const py::object &s) = 0;
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
        const std::function<py::object(const py::object &, const py::object &)>
            &goal_checker,
        const std::function<py::object(const py::object &, const py::object &)>
            &heuristic,
        std::size_t time_budget = 3600000, std::size_t rollout_budget = 100000,
        std::size_t max_depth = 1000, std::size_t max_feasibility_trials = 0,
        double graph_expansion_rate = 0.1,
        std::size_t residual_moving_average_window = 100,
        double epsilon = 0.0, // not a stopping criterion by default
        double discount = 1.0, double action_choice_noise = 0.1,
        double dead_end_cost = 10e4, bool online_node_garbage = false,
        const std::function<py::bool_(const py::object &)> &callback = nullptr,
        bool verbose = false)
        : _goal_checker(goal_checker), _heuristic(heuristic),
          _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      check_domain(domain);
      _domain = std::make_unique<PyMARTDPDomain<Texecution>>(domain);
      _solver =
          std::make_unique<skdecide::MARTDPSolver<PyMARTDPDomain<Texecution>>>(
              *_domain,
              [this](PyMARTDPDomain<Texecution> &d,
                     const typename PyMARTDPDomain<Texecution>::State &s) ->
              typename PyMARTDPDomain<Texecution>::Predicate {
                try {
                  auto fgc = [this](const py::object &dd, const py::object &ss,
                                    [[maybe_unused]] const py::object &ii) {
                    return _goal_checker(dd, ss);
                  };
                  return typename PyMARTDPDomain<Texecution>::Predicate(
                      d.call(nullptr, fgc, s.pyobj()));
                } catch (const std::exception &e) {
                  Logger::error(
                      std::string(
                          "SKDECIDE exception when calling goal checker: ") +
                      e.what());
                  throw;
                }
              },
              [this](PyMARTDPDomain<Texecution> &d,
                     const typename PyMARTDPDomain<Texecution>::State &s)
                  -> std::pair<typename PyMARTDPDomain<Texecution>::Value,
                               typename PyMARTDPDomain<Texecution>::Action> {
                try {
                  auto fh = [this](const py::object &dd, const py::object &ss,
                                   [[maybe_unused]] const py::object &ii) {
                    return _heuristic(dd, ss);
                  };
                  std::unique_ptr<py::object> r =
                      d.call(nullptr, fh, s.pyobj());
                  typename skdecide::GilControl<Texecution>::Acquire acquire;
                  py::tuple t = py::cast<py::tuple>(*r);
                  auto rr = std::make_pair(
                      typename PyMARTDPDomain<Texecution>::Value(t[0]),
                      typename PyMARTDPDomain<Texecution>::Action(t[1]));
                  r.reset();
                  return rr;
                } catch (const std::exception &e) {
                  Logger::error(
                      std::string(
                          "SKDECIDE exception when calling heuristic: ") +
                      e.what());
                  throw;
                }
              },
              time_budget, rollout_budget, max_depth, max_feasibility_trials,
              graph_expansion_rate, residual_moving_average_window, epsilon,
              discount, action_choice_noise, dead_end_cost, online_node_garbage,
              [this](
                  const skdecide::MARTDPSolver<PyMARTDPDomain<Texecution>> &s,
                  PyMARTDPDomain<Texecution> &d) -> bool {
                // we don't make use of the C++ solver object 's' from Python
                // but we rather use its Python wrapper 'solver'
                if (_callback) {
                  std::unique_ptr<py::bool_> r;
                  try {
                    r = std::make_unique<py::bool_>(_callback(*_pysolver));
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
                } else {
                  return false;
                }
              },
              verbose);
      _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(
          std::cout, py::module::import("sys").attr("stdout"));
      _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(
          std::cerr, py::module::import("sys").attr("stderr"));
    }

    virtual ~Implementation() {}

    void check_domain(py::object &domain) {
      if (!py::hasattr(domain, "get_agent_applicable_actions")) {
        throw std::invalid_argument(
            "SKDECIDE exception: MA-RTDP algorithm needs python domain for "
            "implementing get_agent_applicable_actions()");
      }
      if (!py::hasattr(domain, "sample")) {
        throw std::invalid_argument(
            "SKDECIDE exception: MA-RTDP algorithm needs python domain for "
            "implementing sample()");
      }
      if (!py::hasattr(domain, "get_agent_applicable_actions")) {
        throw std::invalid_argument(
            "SKDECIDE exception: MA-RTDP algorithm needs python domain for "
            "implementing get_agent_applicable_actions()");
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
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[MARTDP.get_next_action] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility(const py::object &s) {
      try {
        return _solver->get_best_value(s).pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[MARTDP.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::int_ get_nb_explored_states() {
      return _solver->get_nb_explored_states();
    }

    virtual py::int_ get_state_nb_actions(const py::object &s) {
      return _solver->get_state_nb_actions(s);
    }

    virtual py::int_ get_nb_rollouts() { return _solver->get_nb_rollouts(); }

    virtual py::float_ get_residual_moving_average() {
      return _solver->get_residual_moving_average();
    }

    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }

    virtual py::dict get_policy() {
      py::dict d;
      auto &&p = _solver->policy();
      for (auto &e : p) {
        d[e.first.pyobj()] =
            py::make_tuple(e.second.first.pyobj(), e.second.second.pyobj());
      }
      return d;
    }

  private:
    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<PyMARTDPDomain<Texecution>> _domain;
    std::unique_ptr<skdecide::MARTDPSolver<PyMARTDPDomain<Texecution>>> _solver;

    std::function<py::object(const py::object &, const py::object &)>
        _goal_checker;
    std::function<py::object(const py::object &, const py::object &)>
        _heuristic;
    std::function<py::bool_(const py::object &)> _callback;

    std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
    std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
  };

  struct ExecutionSelector {
    bool _parallel;

    ExecutionSelector(bool parallel) : _parallel(parallel) {}

    template <typename Propagator> struct Select {
      template <typename... Args>
      Select(ExecutionSelector &This, Args... args) {
        // The following test will have to be uncommented if a parallel version
        // of MARTP is implemented in the future
        // if (This._parallel) {
        //   Propagator::template PushType<ParallelExecution>::Forward(args...);
        // } else {
        Propagator::template PushType<SequentialExecution>::Forward(args...);
        // }
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
  PyMARTDPSolver(
      py::object &solver, // Python solver wrapper
      py::object &domain,
      const std::function<py::object(const py::object &, const py::object &)>
          &goal_checker,
      const std::function<py::object(const py::object &, const py::object &)>
          &heuristic,
      std::size_t time_budget = 3600000, std::size_t rollout_budget = 100000,
      std::size_t max_depth = 1000, std::size_t max_feasibility_trials = 0,
      double graph_expansion_rate = 0.1,
      std::size_t residual_moving_average_window = 100,
      double epsilon = 0.0, // not a stopping criterion by default
      double discount = 1.0, double action_choice_noise = 0.1,
      double dead_end_cost = 10e4, bool online_node_garbage = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {

    // we keep the execution selector logics in case we would implement a
    // parallel version of MARTDP in the future
    TemplateInstantiator::select(ExecutionSelector(false),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, goal_checker, heuristic, time_budget,
                     rollout_budget, max_depth, max_feasibility_trials,
                     graph_expansion_rate, residual_moving_average_window,
                     epsilon, discount, action_choice_noise, dead_end_cost,
                     online_node_garbage, callback, verbose);
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

  py::object get_utility(const py::object &s) {
    return _implementation->get_utility(s);
  }

  py::int_ get_nb_explored_states() {
    return _implementation->get_nb_explored_states();
  }

  py::int_ get_state_nb_actions(const py::object &s) {
    return _implementation->get_state_nb_actions(s);
  }

  py::int_ get_nb_rollouts() { return _implementation->get_nb_rollouts(); }

  py::float_ get_residual_moving_average() {
    return _implementation->get_residual_moving_average();
  }

  py::int_ get_solving_time() { return _implementation->get_solving_time(); }

  py::dict get_policy() { return _implementation->get_policy(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_MARTDP_HH
