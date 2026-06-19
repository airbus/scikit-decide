/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_FRET_HH
#define SKDECIDE_PY_FRET_HH

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>

#include "utils/execution.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/python_gil_control.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include <type_traits>

#include "hub/solver/ldfs/ldfs.hh"
#include "hub/solver/lrtdp/lrtdp.hh"
#include "hub/solver/vi/vi.hh"

#include "fret.hh"
#include "impl/fret_impl.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PyFRETDomain = PythonDomainProxy<Texecution>;

struct FRETLRTDPInnerSolver {};
struct FRETLDFSInnerSolver {};
struct FRETVIInnerSolver {};

class PyFRETSolver {
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
    virtual py::int_ get_nb_fret_iterations() = 0;
    virtual py::int_ get_nb_traps_eliminated() = 0;
    virtual py::int_ get_solving_time() = 0;
    virtual py::set get_explored_states() = 0;
    virtual py::set get_dead_end_states() = 0;
    virtual py::list get_trapped_sccs() = 0;
    virtual py::dict get_policy() = 0;
  };

  template <typename Texecution, typename TinnerSolverTag>
  class Implementation : public BaseImplementation {
  public:
    typedef PyFRETDomain<Texecution> Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Action Action;
    typedef typename Domain::Value Value;

    using SolverType = std::conditional_t<
        std::is_same_v<TinnerSolverTag, FRETLRTDPInnerSolver>,
        FRETSolver<Domain, Texecution, LRTDPSolver>,
        std::conditional_t<std::is_same_v<TinnerSolverTag, FRETLDFSInnerSolver>,
                           FRETSolver<Domain, Texecution, LDFSSolver>,
                           FRETSolver<Domain, Texecution, VISolver>>>;

    Implementation(
        py::object &solver, py::object &domain,
        const std::function<py::object(const py::object &, const py::object &)>
            &goal_checker,
        const std::function<py::object(const py::object &, const py::object &)>
            &heuristic,
        double discount, double epsilon, double dead_end_cost,
        const py::dict &inner_solver_params,
        const std::function<py::bool_(const py::object &)> &callback,
        bool verbose)
        : _goal_checker(goal_checker), _heuristic(heuristic),
          _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      _domain = std::make_unique<Domain>(domain);

      auto make_gc = [this]() {
        return [this](Domain &d, const State &s) -> typename Domain::Predicate {
          try {
            auto fgc = [this](const py::object &dd, const py::object &ss,
                              [[maybe_unused]] const py::object &ii) {
              return _goal_checker(dd, ss);
            };
            std::unique_ptr<py::object> r = d.call(nullptr, fgc, s.pyobj());
            typename GilControl<Texecution>::Acquire acquire;
            bool rr = r->template cast<bool>();
            r.reset();
            return rr;
          } catch (const std::exception &e) {
            Logger::error(
                std::string("SKDECIDE exception when calling goal checker: ") +
                e.what());
            throw;
          }
        };
      };

      auto make_h = [this]() {
        return [this](Domain &d, const State &s) -> Value {
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
        };
      };

      auto make_cb = [this]() {
        return [this](const SolverType &, Domain &) -> bool {
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
        };
      };

      create_solver(inner_solver_params, make_gc(), make_h(), discount, epsilon,
                    dead_end_cost, make_cb(), verbose);

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
        Logger::warn(std::string("[FRET.get_next_action] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility(const py::object &s) {
      try {
        return _solver->get_best_value(s).pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[FRET.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::int_ get_nb_explored_states() {
      return _solver->get_nb_explored_states();
    }

    virtual py::int_ get_nb_fret_iterations() {
      return _solver->get_nb_fret_iterations();
    }

    virtual py::int_ get_nb_traps_eliminated() {
      return _solver->get_nb_traps_eliminated();
    }

    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }

    virtual py::set get_explored_states() {
      py::set result;
      auto &&states = _solver->get_explored_states();
      for (const auto &s : states) {
        result.add(s.pyobj());
      }
      return result;
    }

    virtual py::set get_dead_end_states() {
      py::set result;
      auto &&states = _solver->get_dead_end_states();
      for (const auto &s : states) {
        result.add(s.pyobj());
      }
      return result;
    }

    virtual py::list get_trapped_sccs() {
      py::list result;
      auto &&sccs = _solver->get_trapped_sccs();
      for (const auto &scc : sccs) {
        py::set scc_set;
        for (const auto &s : scc) {
          scc_set.add(s.pyobj());
        }
        result.append(scc_set);
      }
      return result;
    }

    virtual py::dict get_policy() {
      py::dict d;
      auto &&states = _solver->get_explored_states();
      for (const auto &s : states) {
        if (_solver->is_solution_defined_for(s)) {
          try {
            d[s.pyobj()] = py::make_tuple(_solver->get_best_action(s).pyobj(),
                                          _solver->get_best_value(s).pyobj());
          } catch (...) {
          }
        }
      }
      return d;
    }

  private:
    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<Domain> _domain;
    std::unique_ptr<SolverType> _solver;

    std::function<py::object(const py::object &, const py::object &)>
        _goal_checker;
    std::function<py::object(const py::object &, const py::object &)>
        _heuristic;
    std::function<py::bool_(const py::object &)> _callback;

    std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
    std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;

    template <typename T>
    static T dp(const py::dict &d, const char *key, T def) {
      return d.contains(key) ? d[key].cast<T>() : def;
    }

    typedef typename SolverType::GoalCheckerFunctor GC;
    typedef typename SolverType::HeuristicFunctor H;
    typedef typename SolverType::CallbackFunctor CB;

    void create_solver(const py::dict &p, GC gc, H h, double discount,
                       double epsilon, double dead_end_cost, CB cb,
                       bool verbose) {
      if (p.contains("terminal_value")) {
        Logger::warn(
            "FRET overrides the inner solver's terminal_value to manage "
            "dead-end costs internally. The user-provided terminal_value "
            "in inner_solver_params will be ignored.");
      }

      if constexpr (std::is_same_v<TinnerSolverTag, FRETLRTDPInnerSolver>) {
        using IS = LRTDPSolver<Domain, Texecution>;
        auto tv = [](const typename Domain::State &) -> Value {
          return Value(0.0, false);
        };
        _solver = std::make_unique<SolverType>(
            *_domain, gc, h, discount, epsilon, dead_end_cost, cb, verbose, tv,
            dp<bool>(p, "use_labels", true),
            dp<std::size_t>(p, "time_budget", 3600000),
            dp<std::size_t>(p, "rollout_budget", 100000),
            dp<std::size_t>(p, "max_depth", 1000),
            dp<std::size_t>(p, "residual_moving_average_window", 100),
            dp<double>(p, "epsilon", epsilon),
            dp<double>(p, "discount", discount),
            dp<bool>(p, "online_node_garbage", false),
            typename IS::CallbackFunctor(
                [](const IS &, Domain &, const std::size_t *) {
                  return false;
                }),
            dp<bool>(p, "verbose", false));
      } else if constexpr (std::is_same_v<TinnerSolverTag,
                                          FRETLDFSInnerSolver>) {
        using IS = LDFSSolver<Domain, Texecution>;
        auto tv = [](const typename Domain::State &) -> Value {
          return Value(0.0, false);
        };
        _solver = std::make_unique<SolverType>(
            *_domain, gc, h, discount, epsilon, dead_end_cost, cb, verbose, tv,
            dp<double>(p, "discount", discount),
            dp<double>(p, "epsilon", epsilon),
            dp<std::size_t>(p, "max_depth", 0),
            typename IS::CallbackFunctor(
                [](const IS &, Domain &) { return false; }),
            dp<bool>(p, "verbose", false));
      } else if constexpr (std::is_same_v<TinnerSolverTag, FRETVIInnerSolver>) {
        using IS = VISolver<Domain, Texecution>;
        auto tv = [](const typename Domain::State &) -> Value {
          return Value(0.0, false);
        };
        // VI has NO goal_checker parameter — it uses domain.is_terminal()
        _solver = std::make_unique<SolverType>(
            *_domain, gc, h, discount, epsilon, dead_end_cost, cb, verbose, tv,
            dp<double>(p, "discount", discount),
            dp<double>(p, "epsilon", epsilon),
            dp<std::size_t>(p, "max_sweeps", 0),
            typename IS::CallbackFunctor(
                [](const IS &, Domain &) { return false; }),
            dp<bool>(p, "verbose", false));
      }
    }
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

  struct InnerSolverSelector {
    std::string _name;
    InnerSolverSelector(const std::string &name) : _name(name) {}
    template <typename Propagator> struct Select {
      template <typename... Args>
      Select(InnerSolverSelector &This, Args... args) {
        if (This._name == "LRTDP") {
          Propagator::template PushType<FRETLRTDPInnerSolver>::Forward(args...);
        } else if (This._name == "LDFS") {
          Propagator::template PushType<FRETLDFSInnerSolver>::Forward(args...);
        } else if (This._name == "VI") {
          Propagator::template PushType<FRETVIInnerSolver>::Forward(args...);
        } else {
          throw std::runtime_error(
              "Unknown inner solver for FRET: " + This._name +
              ". Use LRTDP, LDFS, or VI (ILAOstar is not supported "
              "because it lacks terminal_value).");
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
  PyFRETSolver(
      py::object &solver, py::object &domain,
      const std::function<py::object(const py::object &, const py::object &)>
          &goal_checker,
      const std::function<py::object(const py::object &, const py::object &)>
          &heuristic,
      double discount = 1.0, double epsilon = 0.001,
      double dead_end_cost = 10000.0,
      const py::dict &inner_solver_params = py::dict(),
      const std::string &inner_solver = "LRTDP", bool parallel = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 InnerSolverSelector(inner_solver),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, goal_checker, heuristic, discount, epsilon,
                     dead_end_cost, inner_solver_params, callback, verbose);
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

  py::int_ get_nb_fret_iterations() {
    return _implementation->get_nb_fret_iterations();
  }

  py::int_ get_nb_traps_eliminated() {
    return _implementation->get_nb_traps_eliminated();
  }

  py::int_ get_solving_time() { return _implementation->get_solving_time(); }

  py::set get_explored_states() {
    return _implementation->get_explored_states();
  }

  py::set get_dead_end_states() {
    return _implementation->get_dead_end_states();
  }

  py::list get_trapped_sccs() { return _implementation->get_trapped_sccs(); }

  py::dict get_policy() { return _implementation->get_policy(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_FRET_HH
