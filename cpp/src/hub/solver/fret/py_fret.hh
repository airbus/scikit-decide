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

#include "hub/solver/inner_solver/meta_inner_solver_proxy.hh"
#include "hub/solver/inner_solver/impl/meta_inner_solver_proxy_impl.hh"
#include "hub/solver/inner_solver/inner_solver_registry.hh"
#include "hub/solver/inner_solver/impl/inner_solver_registry_impl.hh"

#include "fret.hh"
#include "impl/fret_impl.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PyFRETDomain = PythonDomainProxy<Texecution>;

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

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    typedef PyFRETDomain<Texecution> Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Action Action;
    typedef typename Domain::Value Value;

    using SolverType = FRETSolver<Domain, Texecution, MetaInnerSolverProxy>;

    Implementation(
        py::object &solver, py::object &domain,
        const std::function<py::object(const py::object &, const py::object &)>
            &goal_checker,
        const std::function<py::object(const py::object &, const py::object &)>
            &heuristic,
        double discount, double epsilon, double dead_end_cost,
        const std::string &inner_solver, const py::dict &inner_solver_params,
        const std::function<py::bool_(const py::object &)> &callback,
        bool verbose)
        : _goal_checker(goal_checker), _heuristic(heuristic),
          _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      _domain = std::make_unique<Domain>(domain);

      auto gc = [this](Domain &d,
                       const State &s) -> typename Domain::Predicate {
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

      auto h = [this](Domain &d, const State &s) -> Value {
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

      auto cb = [this](const SolverType &, Domain &) -> bool {
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

      py::dict isp(inner_solver_params);
      InnerSolverParams params;
      for (auto &[k, v] : isp) {
        std::string key = k.template cast<std::string>();
        if (py::isinstance<py::bool_>(v))
          params.set(key, v.template cast<bool>());
        else if (py::isinstance<py::int_>(v))
          params.set(key, v.template cast<std::size_t>());
        else if (py::isinstance<py::float_>(v))
          params.set(key, v.template cast<double>());
        else if (py::isinstance<py::str>(v))
          params.set(key, v.template cast<std::string>());
      }

      const auto &entry = find_inner_solver<Domain, Texecution>(inner_solver);
      if (!entry.supports_terminal_value) {
        throw std::runtime_error(
            std::string("Inner solver '") + inner_solver +
            "' is not compatible with FRET (requires terminal_value support).");
      }

      using FretFactory =
          typename MetaInnerSolverProxy<Domain, Texecution>::FretFactory;
      FretFactory fret_factory =
          [&entry, params, verbose](
              Domain &dd,
              std::function<typename Domain::Predicate(Domain &, const State &)>
                  sub_gc,
              std::function<Value(Domain &, const State &)> sub_h,
              std::function<Value(const State &)> sub_tv)
          -> std::unique_ptr<MetaInnerSolverBase<Domain>> {
        return entry.create(dd, sub_gc, sub_h, sub_tv, params, verbose);
      };

      auto dummy_tv = [](const State &) -> Value { return Value(0.0, false); };

      _solver = std::make_unique<SolverType>(*_domain, gc, h, discount, epsilon,
                                             dead_end_cost, cb, verbose,
                                             dummy_tv, std::move(fret_factory));

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
  PyFRETSolver(
      py::object &solver, py::object &domain,
      const std::function<py::object(const py::object &, const py::object &)>
          &goal_checker,
      const std::function<py::object(const py::object &, const py::object &)>
          &heuristic,
      double discount = 1.0, double epsilon = 0.001,
      double dead_end_cost = 10000.0, const std::string &inner_solver = "LRTDP",
      const py::dict &inner_solver_params = py::dict(), bool parallel = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, goal_checker, heuristic, discount, epsilon,
                     dead_end_cost, inner_solver, inner_solver_params, callback,
                     verbose);
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
