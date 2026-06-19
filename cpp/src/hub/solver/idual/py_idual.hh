/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_IDUAL_HH
#define SKDECIDE_PY_IDUAL_HH

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>

#include "utils/execution.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/python_gil_control.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "idual.hh"

namespace py = pybind11;

namespace skdecide {

// Unconstrained domain alias (no get_constraints)
template <typename Texecution>
using PyIDualDomain = PythonDomainProxy<Texecution>;

// Constrained domain: extends PythonDomainProxy with get_constraints()
template <typename Texecution>
class PyConstrainedIDualDomain : public PythonDomainProxy<Texecution> {
public:
  using PythonDomainProxy<Texecution>::PythonDomainProxy;
  using State = typename PythonDomainProxy<Texecution>::State;
  using Action = typename PythonDomainProxy<Texecution>::Action;

  struct ConstraintProxy {
    double bound;
    std::function<double(const State &, const Action &)> evaluate;
  };

  void init_constraints(const py::object &domain) {
    py::list py_constraints = domain.attr("get_constraints")();
    _constraints.clear();
    for (auto c : py_constraints) {
      ConstraintProxy cp;
      cp.bound = c.attr("get_bound")().template cast<double>();
      py::object eval_fn =
          py::reinterpret_borrow<py::object>(c.attr("evaluate"));
      cp.evaluate = [eval_fn](const State &s, const Action &a) -> double {
        return eval_fn(s.pyobj(), a.pyobj(), py::none())
            .template cast<double>();
      };
      _constraints.push_back(std::move(cp));
    }
  }

  std::vector<ConstraintProxy> get_constraints() { return _constraints; }

private:
  std::vector<ConstraintProxy> _constraints;
};

// =========================================================================
// PyIDualSolver — unconstrained SSP (deterministic policy)
// =========================================================================

class PyIDualSolver {
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
    virtual py::int_ get_nb_lp_iterations() = 0;
    virtual py::int_ get_solving_time() = 0;
    virtual py::set get_explored_states() = 0;
    virtual py::str get_callback_event() = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    typedef PyIDualDomain<Texecution> Domain;
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
        const std::function<py::object(const py::object &)> &terminal_value,
        double lp_infinity, double lp_tolerance, double default_dead_end_cost,
        std::size_t lp_callback_interval,
        const std::function<py::bool_(const py::object &)> &callback,
        bool verbose)
        : _goal_checker(goal_checker), _heuristic(heuristic),
          _terminal_value(terminal_value), _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      _domain = std::make_unique<Domain>(domain);

      _solver = std::make_unique<IDualSolver<Domain, Texecution>>(
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
          nullptr, std::vector<double>{}, 0.001, lp_infinity, lp_tolerance,
          default_dead_end_cost, lp_callback_interval,
          [this](const IDualSolver<Domain, Texecution> &, Domain &) -> bool {
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
        Logger::warn(std::string("[IDual.get_next_action] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility(const py::object &s) {
      try {
        return _solver->get_best_value(s).pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[IDual.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::int_ get_nb_explored_states() {
      return _solver->get_nb_explored_states();
    }
    virtual py::int_ get_nb_lp_iterations() {
      return _solver->get_nb_lp_iterations();
    }
    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }
    virtual py::set get_explored_states() {
      py::set result;
      auto &&es = _solver->get_explored_states();
      for (auto &e : es) {
        result.add(e.pyobj());
      }
      return result;
    }
    virtual py::str get_callback_event() {
      using LPCallbackEvent =
          typename IDualSolver<Domain, Texecution>::LPCallbackEvent;
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
    std::unique_ptr<IDualSolver<Domain, Texecution>> _solver;

    std::function<py::object(const py::object &, const py::object &)>
        _goal_checker;
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
  PyIDualSolver(
      py::object &solver, py::object &domain,
      const std::function<py::object(const py::object &, const py::object &)>
          &goal_checker,
      const std::function<py::object(const py::object &, const py::object &)>
          &heuristic,
      const std::function<py::object(const py::object &)> &terminal_value,
      double lp_infinity = 1e20, double lp_tolerance = 1e-15,
      double default_dead_end_cost = 1000.0,
      std::size_t lp_callback_interval = 0, bool parallel = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, goal_checker, heuristic, terminal_value,
                     lp_infinity, lp_tolerance, default_dead_end_cost,
                     lp_callback_interval, callback, verbose);
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
  py::int_ get_nb_lp_iterations() {
    return _implementation->get_nb_lp_iterations();
  }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
  py::set get_explored_states() {
    return _implementation->get_explored_states();
  }
  py::str get_callback_event() { return _implementation->get_callback_event(); }
};

// =========================================================================
// PyCIDualSolver — constrained SSP (stochastic policy)
// =========================================================================

class PyCIDualSolver {
private:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void close() = 0;
    virtual void clear() = 0;
    virtual void solve(const py::object &s) = 0;
    virtual py::bool_ is_solution_defined_for(const py::object &s) = 0;
    virtual py::list get_action_distribution(const py::object &s) = 0;
    virtual py::object get_utility(const py::object &s) = 0;
    virtual py::int_ get_nb_explored_states() = 0;
    virtual py::int_ get_nb_lp_iterations() = 0;
    virtual py::int_ get_solving_time() = 0;
    virtual py::set get_explored_states() = 0;
    virtual py::str get_callback_event() = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    typedef PyConstrainedIDualDomain<Texecution> Domain;
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
        const std::function<py::object(const py::object &)> &terminal_value,
        const std::function<py::object(const py::object &, const py::object &,
                                       int)> &secondary_heuristic,
        py::list dead_end_costs_list, double lp_infinity, double lp_tolerance,
        double default_dead_end_cost, std::size_t lp_callback_interval,
        const std::function<py::bool_(const py::object &)> &callback,
        bool verbose)
        : _goal_checker(goal_checker), _heuristic(heuristic),
          _terminal_value(terminal_value),
          _secondary_heuristic(secondary_heuristic), _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      _pydomain = std::make_unique<py::object>(domain);
      _domain = std::make_unique<Domain>(domain);

      _domain->init_constraints(domain);

      std::vector<double> dead_end_costs;
      for (auto item : dead_end_costs_list) {
        dead_end_costs.push_back(item.template cast<double>());
      }

      typename IDualSolver<Domain, Texecution>::SecondaryHeuristicFunctor
          sec_heur = nullptr;
      if (_secondary_heuristic) {
        sec_heur = [this](Domain &d, const State &s, std::size_t j) -> double {
          try {
            typename GilControl<Texecution>::Acquire acquire;
            py::object result = _secondary_heuristic(*_pydomain, s.pyobj(),
                                                     static_cast<int>(j));
            return result.attr("cost").template cast<double>();
          } catch (const std::exception &e) {
            Logger::error(
                std::string(
                    "SKDECIDE exception when calling secondary_heuristic: ") +
                e.what());
            throw;
          }
        };
      }

      _solver = std::make_unique<IDualSolver<Domain, Texecution>>(
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
          sec_heur, dead_end_costs, 0.001, lp_infinity, lp_tolerance,
          default_dead_end_cost, lp_callback_interval,
          [this](const IDualSolver<Domain, Texecution> &, Domain &) -> bool {
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

    virtual py::list get_action_distribution(const py::object &s) {
      try {
        auto dist = _solver->get_action_distribution(s);
        py::list result;
        for (auto &[action, prob] : dist) {
          result.append(py::make_tuple(action.pyobj(), prob));
        }
        return result;
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[CIDual.get_action_distribution] ") +
                     e.what() + " - returning empty list");
        return py::list();
      }
    }

    virtual py::object get_utility(const py::object &s) {
      try {
        return _solver->get_best_value(s).pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[CIDual.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::int_ get_nb_explored_states() {
      return _solver->get_nb_explored_states();
    }
    virtual py::int_ get_nb_lp_iterations() {
      return _solver->get_nb_lp_iterations();
    }
    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }
    virtual py::set get_explored_states() {
      py::set result;
      auto &&es = _solver->get_explored_states();
      for (auto &e : es) {
        result.add(e.pyobj());
      }
      return result;
    }
    virtual py::str get_callback_event() {
      using LPCallbackEvent =
          typename IDualSolver<Domain, Texecution>::LPCallbackEvent;
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
    std::unique_ptr<py::object> _pydomain;
    std::unique_ptr<Domain> _domain;
    std::unique_ptr<IDualSolver<Domain, Texecution>> _solver;

    std::function<py::object(const py::object &, const py::object &)>
        _goal_checker;
    std::function<py::object(const py::object &, const py::object &)>
        _heuristic;
    std::function<py::object(const py::object &)> _terminal_value;
    std::function<py::object(const py::object &, const py::object &, int)>
        _secondary_heuristic;
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
  PyCIDualSolver(
      py::object &solver, py::object &domain,
      const std::function<py::object(const py::object &, const py::object &)>
          &goal_checker,
      const std::function<py::object(const py::object &, const py::object &)>
          &heuristic,
      const std::function<py::object(const py::object &)> &terminal_value,
      const std::function<py::object(const py::object &, const py::object &,
                                     int)> &secondary_heuristic,
      py::list dead_end_costs, double lp_infinity = 1e20,
      double lp_tolerance = 1e-15, double default_dead_end_cost = 1000.0,
      std::size_t lp_callback_interval = 0, bool parallel = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, goal_checker, heuristic, terminal_value,
                     secondary_heuristic, dead_end_costs, lp_infinity,
                     lp_tolerance, default_dead_end_cost, lp_callback_interval,
                     callback, verbose);
  }

  void close() { _implementation->close(); }
  void clear() { _implementation->clear(); }
  void solve(const py::object &s) { _implementation->solve(s); }
  py::bool_ is_solution_defined_for(const py::object &s) {
    return _implementation->is_solution_defined_for(s);
  }
  py::list get_action_distribution(const py::object &s) {
    return _implementation->get_action_distribution(s);
  }
  py::object get_utility(const py::object &s) {
    return _implementation->get_utility(s);
  }
  py::int_ get_nb_explored_states() {
    return _implementation->get_nb_explored_states();
  }
  py::int_ get_nb_lp_iterations() {
    return _implementation->get_nb_lp_iterations();
  }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
  py::set get_explored_states() {
    return _implementation->get_explored_states();
  }
  py::str get_callback_event() { return _implementation->get_callback_event(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_IDUAL_HH
