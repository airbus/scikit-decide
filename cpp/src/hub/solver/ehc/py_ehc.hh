/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_EHC_HH
#define SKDECIDE_PY_EHC_HH

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>

#include "utils/execution.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/python_gil_control.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "ehc.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PyEHCDomain = PythonDomainProxy<Texecution>;

class PyEHCSolver {
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
    virtual py::set get_explored_states() = 0;
    virtual py::int_ get_solving_time() = 0;
    virtual py::list get_plan(const py::object &s) = 0;
    virtual py::dict get_policy() = 0;
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
        const std::function<py::object(const py::object &, const py::object &)>
            &preferred_actions,
        const std::function<py::bool_(const py::object &)> &callback = nullptr,
        bool verbose = false)
        : _goal_checker(goal_checker), _heuristic(heuristic),
          _preferred_actions_py(preferred_actions), _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      check_domain(domain);
      _domain = std::make_unique<PyEHCDomain<Texecution>>(domain);

      // Build preferred actions functor (or nullptr)
      typename EHCSolver<PyEHCDomain<Texecution>,
                         Texecution>::PreferredActionsFunctor pa_functor =
          nullptr;
      if (_preferred_actions_py) {
        pa_functor = [this](PyEHCDomain<Texecution> &d,
                            const typename PyEHCDomain<Texecution>::State &s)
            -> std::vector<typename PyEHCDomain<Texecution>::Action> {
          try {
            auto fpa = [this](const py::object &dd, const py::object &ss,
                              [[maybe_unused]] const py::object &ii) {
              return _preferred_actions_py(dd, ss);
            };
            std::unique_ptr<py::object> r = d.call(nullptr, fpa, s.pyobj());
            typename GilControl<Texecution>::Acquire acquire;
            std::vector<typename PyEHCDomain<Texecution>::Action> result;
            for (auto item : r->template cast<py::list>()) {
              result.emplace_back(py::reinterpret_borrow<py::object>(item));
            }
            r.reset();
            return result;
          } catch (const std::exception &e) {
            Logger::error(
                std::string(
                    "SKDECIDE exception when calling preferred actions: ") +
                e.what());
            throw;
          }
        };
      }

      _solver =
          std::make_unique<EHCSolver<PyEHCDomain<Texecution>, Texecution>>(
              *_domain,
              [this](PyEHCDomain<Texecution> &d,
                     const typename PyEHCDomain<Texecution>::State &s) ->
              typename PyEHCDomain<Texecution>::Predicate {
                try {
                  auto fgc = [this](const py::object &dd, const py::object &ss,
                                    [[maybe_unused]] const py::object &ii) {
                    return _goal_checker(dd, ss);
                  };
                  std::unique_ptr<py::object> r =
                      d.call(nullptr, fgc, s.pyobj());
                  typename GilControl<Texecution>::Acquire acquire;
                  bool rr = r->template cast<bool>();
                  r.reset();
                  return rr;
                } catch (const std::exception &e) {
                  Logger::error(std::string("SKDECIDE exception when calling "
                                            "goal checker: ") +
                                e.what());
                  throw;
                }
              },
              [this](PyEHCDomain<Texecution> &d,
                     const typename PyEHCDomain<Texecution>::State &s) ->
              typename PyEHCDomain<Texecution>::Value {
                try {
                  auto fh = [this](const py::object &dd, const py::object &ss,
                                   [[maybe_unused]] const py::object &ii) {
                    return _heuristic(dd, ss);
                  };
                  return typename PyEHCDomain<Texecution>::Value(
                      d.call(nullptr, fh, s.pyobj()));
                } catch (const std::exception &e) {
                  Logger::error(std::string("SKDECIDE exception when calling "
                                            "heuristic estimator: ") +
                                e.what());
                  throw;
                }
              },
              pa_functor,
              [this](const EHCSolver<PyEHCDomain<Texecution>, Texecution> &,
                     PyEHCDomain<Texecution> &) -> bool {
                if (_callback) {
                  try {
                    return _callback(*_pysolver);
                  } catch (const std::exception &e) {
                    Logger::error(std::string("SKDECIDE exception when calling "
                                              "callback function: ") +
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

    void check_domain(py::object &domain) {
      if (!py::hasattr(domain, "get_applicable_actions")) {
        throw std::invalid_argument(
            "SKDECIDE exception: EHC algorithm needs python domain for "
            "implementing get_applicable_actions()");
      }
      if (!py::hasattr(domain, "get_next_state")) {
        throw std::invalid_argument(
            "SKDECIDE exception: EHC algorithm needs python domain for "
            "implementing get_next_state()");
      }
      if (!py::hasattr(domain, "get_transition_value")) {
        throw std::invalid_argument(
            "SKDECIDE exception: EHC algorithm needs python domain for "
            "implementing get_transition_value()");
      }
    }

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
        Logger::warn(std::string("[EHC.get_next_action] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility(const py::object &s) {
      try {
        return _solver->get_best_value(s).pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[EHC.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::int_ get_nb_explored_states() {
      return _solver->get_nb_explored_states();
    }

    virtual py::set get_explored_states() {
      py::set s;
      auto &&es = _solver->get_explored_states();
      for (auto &e : es) {
        s.add(e.pyobj());
      }
      return s;
    }

    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }

    virtual py::list get_plan(const py::object &s) {
      py::list l;
      auto &&p = _solver->get_plan(s);
      for (auto &e : p) {
        l.append(py::make_tuple(std::get<0>(e).pyobj(), std::get<1>(e).pyobj(),
                                std::get<2>(e).pyobj()));
      }
      return l;
    }

    virtual py::dict get_policy() {
      py::dict d;
      auto &&p = _solver->get_policy();
      for (auto &e : p) {
        d[e.first.pyobj()] =
            py::make_tuple(e.second.first.pyobj(), e.second.second.pyobj());
      }
      return d;
    }

  private:
    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<PyEHCDomain<Texecution>> _domain;
    std::unique_ptr<EHCSolver<PyEHCDomain<Texecution>, Texecution>> _solver;

    std::function<py::object(const py::object &, const py::object &)>
        _goal_checker;
    std::function<py::object(const py::object &, const py::object &)>
        _heuristic;
    std::function<py::object(const py::object &, const py::object &)>
        _preferred_actions_py;
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
  PyEHCSolver(
      py::object &solver, py::object &domain,
      const std::function<py::object(const py::object &, const py::object &)>
          &goal_checker,
      const std::function<py::object(const py::object &, const py::object &)>
          &heuristic,
      const std::function<py::object(const py::object &, const py::object &)>
          &preferred_actions,
      bool parallel = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {

    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, goal_checker, heuristic, preferred_actions,
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

  py::int_ get_nb_explored_states() {
    return _implementation->get_nb_explored_states();
  }

  py::set get_explored_states() {
    return _implementation->get_explored_states();
  }

  py::int_ get_solving_time() { return _implementation->get_solving_time(); }

  py::list get_plan(const py::object &s) {
    return _implementation->get_plan(s);
  }

  py::dict get_policy() { return _implementation->get_policy(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_EHC_HH
