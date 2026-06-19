/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_SSPDETHINDSIGHT_HH
#define SKDECIDE_PY_SSPDETHINDSIGHT_HH

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>

#include <string>

#include "utils/execution.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/python_gil_control.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "hub/solver/sspdethindsight/sspdethindsight.hh"
#include "hub/solver/sspdethindsight/impl/sspdethindsight_impl.hh"
#include "hub/solver/determinization/determinized_domain.hh"
#include "hub/solver/determinization/impl/determinized_domain_impl.hh"

#include "hub/solver/inner_solver/inner_solver_registry.hh"
#include "hub/solver/inner_solver/impl/inner_solver_registry_impl.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PySSPDetHindsightDomain = PythonDomainProxy<Texecution>;

class PySSPDetHindsightSolver {
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
    virtual py::int_ get_nb_steps() = 0;
    virtual py::int_ get_solving_time() = 0;
    virtual py::set get_explored_states() = 0;
    virtual py::set get_terminal_states() = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    typedef PySSPDetHindsightDomain<Texecution> Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Action Action;
    typedef typename Domain::Value Value;

    using DetDomain =
        DeterminizedDomain<Domain, RandomOutcomeStrategy, Texecution>;
    using Adapter =
        TransitionDeterminizationAdapter<Domain, RandomOutcomeStrategy,
                                         Texecution>;
    using SolverType = SSPDetHindsightSolver<Domain, Texecution, Adapter>;

    Implementation(
        py::object &solver, py::object &domain,
        const std::function<py::object(const py::object &, const py::object &)>
            &goal_checker,
        const std::function<py::object(const py::object &, const py::object &)>
            &heuristic,
        const std::string &inner_solver, const py::dict &inner_solver_params,
        std::size_t sample_width, double dead_end_cost, std::size_t max_steps,
        double discount, double epsilon,
        const std::function<py::bool_(const py::object &)> &callback,
        bool verbose)
        : _goal_checker(goal_checker), _heuristic(heuristic),
          _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      _domain = std::make_unique<Domain>(domain);

      auto gc = [this](Domain &d, const State &s) -> bool {
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

      auto inner_gc = [this](DetDomain &, const State &s) ->
          typename DetDomain::Predicate {
            try {
              auto fgc = [this](const py::object &dd, const py::object &ss,
                                [[maybe_unused]] const py::object &ii) {
                return _goal_checker(dd, ss);
              };
              std::unique_ptr<py::object> r =
                  _domain->call(nullptr, fgc, s.pyobj());
              typename GilControl<Texecution>::Acquire acquire;
              bool rr = r->template cast<bool>();
              r.reset();
              return rr;
            } catch (const std::exception &e) {
              Logger::error(std::string("SKDECIDE exception when calling inner "
                                        "goal checker: ") +
                            e.what());
              throw;
            }
          };

      auto inner_h = [this](DetDomain &,
                            const State &s) -> typename DetDomain::Value {
        try {
          auto fh = [this](const py::object &dd, const py::object &ss,
                           [[maybe_unused]] const py::object &ii) {
            return _heuristic(dd, ss);
          };
          return
              typename DetDomain::Value(_domain->call(nullptr, fh, s.pyobj()));
        } catch (const std::exception &e) {
          Logger::error(
              std::string("SKDECIDE exception when calling inner heuristic: ") +
              e.what());
          throw;
        }
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

      const auto &entry =
          find_inner_solver<DetDomain, SequentialExecution>(inner_solver);
      auto create_fn = entry.create;
      auto factory = [create_fn, inner_gc, inner_h, params,
                      verbose](DetDomain &det_d)
          -> std::unique_ptr<MetaInnerSolverBase<DetDomain>> {
        auto tv = [](const typename DetDomain::State &) ->
            typename DetDomain::Value {
              return typename DetDomain::Value(0.0, false);
            };
        return create_fn(det_d, inner_gc, inner_h, tv, params, verbose);
      };

      Domain *domain_ptr = _domain.get();
      auto adapter_factory = [domain_ptr]() -> Adapter {
        return Adapter(*domain_ptr);
      };

      _solver = std::make_unique<SolverType>(
          *_domain, std::move(factory), std::move(adapter_factory), gc,
          sample_width, dead_end_cost, max_steps, discount, epsilon, cb,
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
        typename GilControl<Texecution>::Release release;
        const auto &action =
            _solver->get_best_action(typename Domain::State(s));
        typename GilControl<Texecution>::Acquire acquire;
        return action.pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[SSPDetHindsight.get_next_action] ") +
                     e.what() + " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility(const py::object &s) {
      try {
        return py::cast(_solver->get_best_value(typename Domain::State(s)));
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[SSPDetHindsight.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::int_ get_nb_steps() { return _solver->get_nb_steps(); }
    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }

    virtual py::set get_explored_states() {
      py::set s;
      for (auto &e : _solver->get_explored_states()) {
        s.add(e.pyobj());
      }
      return s;
    }

    virtual py::set get_terminal_states() {
      py::set s;
      for (auto &e : _solver->get_terminal_states()) {
        s.add(e.pyobj());
      }
      return s;
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
  PySSPDetHindsightSolver(
      py::object &solver, py::object &domain,
      const std::function<py::object(const py::object &, const py::object &)>
          &goal_checker,
      const std::function<py::object(const py::object &, const py::object &)>
          &heuristic,
      const std::string &inner_solver = "Astar",
      const py::dict &inner_solver_params = py::dict(),
      std::size_t sample_width = 30, double dead_end_cost = 1000.0,
      std::size_t max_steps = 10000, double discount = 0.99,
      double epsilon = 1e-3, bool parallel = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, goal_checker, heuristic, inner_solver,
                     inner_solver_params, sample_width, dead_end_cost,
                     max_steps, discount, epsilon, callback, verbose);
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

  py::int_ get_nb_steps() { return _implementation->get_nb_steps(); }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }

  py::set get_explored_states() {
    return _implementation->get_explored_states();
  }
  py::set get_terminal_states() {
    return _implementation->get_terminal_states();
  }
};

} // namespace skdecide

#endif // SKDECIDE_PY_SSPDETHINDSIGHT_HH
