/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_SSPREPLAN_HH
#define SKDECIDE_PY_SSPREPLAN_HH

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include <string>
#include <type_traits>

#include "utils/execution.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/python_gil_control.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "hub/solver/sspreplan/sspreplan.hh"
#include "hub/solver/sspreplan/impl/sspreplan_impl.hh"
#include "hub/solver/determinization/determinized_domain.hh"
#include "hub/solver/determinization/impl/determinized_domain_impl.hh"

#include "hub/solver/inner_solver/inner_solver_registry.hh"
#include "hub/solver/inner_solver/impl/inner_solver_registry_impl.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PySSPReplanDomain = PythonDomainProxy<Texecution>;

class PySSPReplanSolver {
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
    virtual py::list get_plan() = 0;
    virtual py::int_ get_nb_replans() = 0;
    virtual py::int_ get_nb_steps() = 0;
    virtual py::int_ get_solving_time() = 0;
    virtual py::float_ get_total_cost() = 0;
  };

  template <typename Texecution, typename TstrategyTag>
  class Implementation : public BaseImplementation {
  public:
    typedef PySSPReplanDomain<Texecution> Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Action Action;
    typedef typename Domain::Value Value;

    using DetDomain = DeterminizedDomain<Domain, TstrategyTag, Texecution>;
    using Adapter =
        TransitionDeterminizationAdapter<Domain, TstrategyTag, Texecution>;
    using SolverType = SSPReplanSolver<Domain, Texecution, Adapter>;
    using DetAction = typename DetDomain::Action;

    Implementation(
        py::object &solver, py::object &domain,
        const std::function<py::object(const py::object &, const py::object &)>
            &goal_checker,
        const std::function<py::object(const py::object &, const py::object &)>
            &heuristic,
        std::size_t max_replans, std::size_t max_steps,
        const std::string &inner_solver, const py::dict &inner_solver_params,
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
              Logger::error(
                  std::string(
                      "SKDECIDE exception when calling inner goal checker: ") +
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

      Adapter adapter(*_domain);
      _solver = std::make_unique<SolverType>(
          *_domain, std::move(adapter), std::move(factory), gc, max_replans,
          max_steps, cb, verbose);

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
        Logger::warn(std::string("[SSPReplan.get_next_action] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility(const py::object &s) {
      try {
        return py::cast(_solver->get_best_value(typename Domain::State(s)));
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[SSPReplan.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::list get_plan() override {
      py::list result;
      for (const auto &[state, action] : _solver->get_plan()) {
        result.append(py::make_tuple(state.pyobj(), action.pyobj()));
      }
      return result;
    }

    virtual py::int_ get_nb_replans() { return _solver->get_nb_replans(); }
    virtual py::int_ get_nb_steps() { return _solver->get_nb_steps(); }
    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }
    virtual py::float_ get_total_cost() { return _solver->get_total_cost(); }

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

  struct DeterminizationSelector {
    std::string _name;
    DeterminizationSelector(const std::string &name) : _name(name) {}
    template <typename Propagator> struct Select {
      template <typename... Args>
      Select(DeterminizationSelector &This, Args... args) {
        if (This._name == "all_outcomes") {
          Propagator::template PushType<AllOutcomesStrategy>::Forward(args...);
        } else if (This._name == "most_probable_outcome") {
          Propagator::template PushType<MostProbableOutcomeStrategy>::Forward(
              args...);
        } else if (This._name == "random_outcome") {
          Propagator::template PushType<RandomOutcomeStrategy>::Forward(
              args...);
        } else {
          throw std::runtime_error(
              "Unknown determinization strategy: " + This._name +
              ". Use all_outcomes, most_probable_outcome, or random_outcome.");
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
  PySSPReplanSolver(
      py::object &solver, py::object &domain,
      const std::function<py::object(const py::object &, const py::object &)>
          &goal_checker,
      const std::function<py::object(const py::object &, const py::object &)>
          &heuristic,
      const std::string &determinization = "most_probable_outcome",
      const std::string &inner_solver = "Astar", std::size_t max_replans = 1000,
      std::size_t max_steps = 10000,
      const py::dict &inner_solver_params = py::dict(), bool parallel = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 DeterminizationSelector(determinization),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, goal_checker, heuristic, max_replans,
                     max_steps, inner_solver, inner_solver_params, callback,
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

  py::list get_plan() { return _implementation->get_plan(); }
  py::int_ get_nb_replans() { return _implementation->get_nb_replans(); }
  py::int_ get_nb_steps() { return _implementation->get_nb_steps(); }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
  py::float_ get_total_cost() { return _implementation->get_total_cost(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_SSPREPLAN_HH
