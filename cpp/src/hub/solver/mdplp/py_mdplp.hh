/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_MDPLP_HH
#define SKDECIDE_PY_MDPLP_HH

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "utils/execution.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/python_gil_control.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "mdplp.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PyMDPLPDomain = PythonDomainProxy<Texecution>;

class PyMDPLPSolver {
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
    virtual py::int_ get_nb_states() = 0;
    virtual py::int_ get_nb_lp_variables() = 0;
    virtual py::int_ get_nb_lp_constraints() = 0;
    virtual py::int_ get_solving_time() = 0;
    virtual py::set get_explored_states() = 0;
    virtual py::str get_callback_event() = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    typedef PyMDPLPDomain<Texecution> Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Action Action;
    typedef typename Domain::Value Value;

    Implementation(
        py::object &solver, py::object &domain,
        const std::function<py::object(const py::object &, const py::object &)>
            &heuristic,
        const std::function<py::object(const py::object &)> &terminal_value,
        const std::string &variant_str, double discount, double epsilon,
        double lp_infinity, std::size_t lp_callback_interval,
        const std::function<py::bool_(const py::object &)> &callback,
        bool verbose)
        : _heuristic(heuristic), _terminal_value(terminal_value),
          _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      _domain = std::make_unique<Domain>(domain);

      _solver = std::make_unique<MDPLPSolver<Domain, Texecution>>(
          *_domain,
          [this](Domain &d, const State &s) -> Value {
            try {
              auto fh = [this](const py::object &dd, const py::object &ss,
                               [[maybe_unused]] const py::object &ii) {
                return _heuristic(dd, ss);
              };
              return Value(d.call(nullptr, fh, s.pyobj()));
            } catch (const std::exception &e) {
              Logger::error(
                  std::string("SKDECIDE exception when calling heuristic: ") +
                  e.what());
              throw;
            }
          },
          [this](const State &s) -> Value {
            try {
              typename GilControl<Texecution>::Acquire acquire;
              return Value(_terminal_value(s.pyobj()));
            } catch (const std::exception &e) {
              Logger::error(
                  std::string(
                      "SKDECIDE exception when calling terminal_value: ") +
                  e.what());
              throw;
            }
          },
          lp_variant_from_string(variant_str), discount, epsilon, lp_infinity,
          lp_callback_interval,
          [this](const MDPLPSolver<Domain, Texecution> &, Domain &) -> bool {
            if (_callback) {
              try {
                return _callback(*_pysolver);
              } catch (const std::exception &e) {
                Logger::error(
                    std::string("SKDECIDE exception when calling callback: ") +
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
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[MDPLP.get_next_action] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility(const py::object &s) {
      try {
        return _solver->get_best_value(s).pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[MDPLP.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::int_ get_nb_states() { return _solver->get_nb_states(); }
    virtual py::int_ get_nb_lp_variables() {
      return _solver->get_nb_lp_variables();
    }
    virtual py::int_ get_nb_lp_constraints() {
      return _solver->get_nb_lp_constraints();
    }
    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }
    virtual py::set get_explored_states() {
      py::set s;
      auto &&es = _solver->get_explored_states();
      for (auto &e : es) {
        s.add(e.pyobj());
      }
      return s;
    }
    virtual py::str get_callback_event() {
      using LPCallbackEvent =
          typename MDPLPSolver<Domain, Texecution>::LPCallbackEvent;
      switch (_solver->get_callback_event()) {
      case LPCallbackEvent::SolverIteration:
        return py::str("SolverIteration");
      case LPCallbackEvent::LPProgress:
        return py::str("LPProgress");
      default:
        return py::str("Unknown");
      }
    }

  private:
    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<Domain> _domain;
    std::unique_ptr<MDPLPSolver<Domain, Texecution>> _solver;

    std::function<py::object(const py::object &, const py::object &)>
        _heuristic;
    std::function<py::object(const py::object &)> _terminal_value;
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
  PyMDPLPSolver(
      py::object &solver, py::object &domain,
      const std::function<py::object(const py::object &, const py::object &)>
          &heuristic,
      const std::function<py::object(const py::object &)> &terminal_value,
      const std::string &variant = "dual", double discount = 0.99,
      double epsilon = 0.001, double lp_infinity = 1e20,
      std::size_t lp_callback_interval = 0, bool parallel = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, heuristic, terminal_value, variant,
                     discount, epsilon, lp_infinity, lp_callback_interval,
                     callback, verbose);
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

  py::int_ get_nb_states() { return _implementation->get_nb_states(); }
  py::int_ get_nb_lp_variables() {
    return _implementation->get_nb_lp_variables();
  }
  py::int_ get_nb_lp_constraints() {
    return _implementation->get_nb_lp_constraints();
  }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
  py::set get_explored_states() {
    return _implementation->get_explored_states();
  }
  py::str get_callback_event() { return _implementation->get_callback_event(); }
};

// =========================================================================
// PySSPLPSolver — pybind wrapper for SSPLPSolver (undiscounted SSP)
// =========================================================================

class PySSPLPSolver {
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
    virtual py::int_ get_nb_states() = 0;
    virtual py::int_ get_nb_lp_variables() = 0;
    virtual py::int_ get_nb_lp_constraints() = 0;
    virtual py::int_ get_solving_time() = 0;
    virtual py::set get_explored_states() = 0;
    virtual py::str get_callback_event() = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    typedef PyMDPLPDomain<Texecution> Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Action Action;
    typedef typename Domain::Value Value;
    typedef typename Domain::Predicate Predicate;

    Implementation(
        py::object &solver, py::object &domain,
        const std::function<py::object(const py::object &, const py::object &)>
            &goal_checker,
        const std::function<py::object(const py::object &, const py::object &)>
            &heuristic,
        const std::string &variant_str, double epsilon, double lp_infinity,
        std::size_t lp_callback_interval,
        const std::function<py::bool_(const py::object &)> &callback,
        bool verbose)
        : _goal_checker(goal_checker), _heuristic(heuristic),
          _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      _domain = std::make_unique<Domain>(domain);

      _solver = std::make_unique<SSPLPSolver<Domain, Texecution>>(
          *_domain,
          [this](Domain &d, const State &s) -> Predicate {
            try {
              auto fgc = [this](const py::object &dd, const py::object &ss,
                                [[maybe_unused]] const py::object &ii) {
                return _goal_checker(dd, ss);
              };
              std::unique_ptr<py::object> r = d.call(nullptr, fgc, s.pyobj());
              typename GilControl<Texecution>::Acquire acquire;
              return r->template cast<bool>();
            } catch (const std::exception &e) {
              Logger::error(
                  std::string(
                      "SKDECIDE exception when calling goal_checker: ") +
                  e.what());
              throw;
            }
          },
          [this](Domain &d, const State &s) -> Value {
            try {
              auto fh = [this](const py::object &dd, const py::object &ss,
                               [[maybe_unused]] const py::object &ii) {
                return _heuristic(dd, ss);
              };
              return Value(d.call(nullptr, fh, s.pyobj()));
            } catch (const std::exception &e) {
              Logger::error(
                  std::string("SKDECIDE exception when calling heuristic: ") +
                  e.what());
              throw;
            }
          },
          lp_variant_from_string(variant_str), epsilon, lp_infinity,
          lp_callback_interval,
          [this](const SSPLPSolver<Domain, Texecution> &, Domain &) -> bool {
            if (_callback) {
              try {
                return _callback(*_pysolver);
              } catch (const std::exception &e) {
                Logger::error(
                    std::string("SKDECIDE exception when calling callback: ") +
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
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[SSPLP.get_next_action] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility(const py::object &s) {
      try {
        return _solver->get_best_value(s).pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[SSPLP.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::int_ get_nb_states() { return _solver->get_nb_states(); }
    virtual py::int_ get_nb_lp_variables() {
      return _solver->get_nb_lp_variables();
    }
    virtual py::int_ get_nb_lp_constraints() {
      return _solver->get_nb_lp_constraints();
    }
    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }
    virtual py::set get_explored_states() {
      py::set s;
      auto &&es = _solver->get_explored_states();
      for (auto &e : es) {
        s.add(e.pyobj());
      }
      return s;
    }
    virtual py::str get_callback_event() {
      using LPCallbackEvent =
          typename SSPLPSolver<Domain, Texecution>::LPCallbackEvent;
      switch (_solver->get_callback_event()) {
      case LPCallbackEvent::SolverIteration:
        return py::str("SolverIteration");
      case LPCallbackEvent::LPProgress:
        return py::str("LPProgress");
      default:
        return py::str("Unknown");
      }
    }

  private:
    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<Domain> _domain;
    std::unique_ptr<SSPLPSolver<Domain, Texecution>> _solver;

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
  PySSPLPSolver(
      py::object &solver, py::object &domain,
      const std::function<py::object(const py::object &, const py::object &)>
          &goal_checker,
      const std::function<py::object(const py::object &, const py::object &)>
          &heuristic,
      const std::string &variant = "dual", double epsilon = 0.001,
      double lp_infinity = 1e20, std::size_t lp_callback_interval = 0,
      bool parallel = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, goal_checker, heuristic, variant, epsilon,
                     lp_infinity, lp_callback_interval, callback, verbose);
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

  py::int_ get_nb_states() { return _implementation->get_nb_states(); }
  py::int_ get_nb_lp_variables() {
    return _implementation->get_nb_lp_variables();
  }
  py::int_ get_nb_lp_constraints() {
    return _implementation->get_nb_lp_constraints();
  }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
  py::set get_explored_states() {
    return _implementation->get_explored_states();
  }
  py::str get_callback_event() { return _implementation->get_callback_event(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_MDPLP_HH
