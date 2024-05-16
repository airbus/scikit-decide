/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_ILAOSTAR_HH
#define SKDECIDE_PY_ILAOSTAR_HH

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "utils/execution.hh"
#include "utils/python_gil_control.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "ilaostar.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PyILAOStarDomain = PythonDomainProxy<Texecution>;

class PyILAOStarSolver {
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
    virtual py::int_ best_solution_graph_size() = 0;
    virtual py::dict get_policy() = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    Implementation(
        py::object &domain,
        const std::function<py::object(const py::object &, const py::object &)>
            &goal_checker,
        const std::function<py::object(const py::object &, const py::object &)>
            &heuristic,
        double discount = 1.0, double epsilon = 0.001, bool verbose = false)
        : _goal_checker(goal_checker), _heuristic(heuristic) {

      check_domain(domain);
      _domain = std::make_unique<PyILAOStarDomain<Texecution>>(domain);
      _solver = std::make_unique<
          skdecide::ILAOStarSolver<PyILAOStarDomain<Texecution>, Texecution>>(
          *_domain,
          [this](PyILAOStarDomain<Texecution> &d,
                 const typename PyILAOStarDomain<Texecution>::State &s) ->
          typename PyILAOStarDomain<Texecution>::Predicate {
            try {
              auto fgc = [this](const py::object &dd, const py::object &ss,
                                [[maybe_unused]] const py::object &ii) {
                return _goal_checker(dd, ss);
              };
              std::unique_ptr<py::object> r = d.call(nullptr, fgc, s.pyobj());
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
          [this](PyILAOStarDomain<Texecution> &d,
                 const typename PyILAOStarDomain<Texecution>::State &s) ->
          typename PyILAOStarDomain<Texecution>::Value {
            try {
              auto fh = [this](const py::object &dd, const py::object &ss,
                               [[maybe_unused]] const py::object &ii) {
                return _heuristic(dd, ss);
              };
              return typename PyILAOStarDomain<Texecution>::Value(
                  d.call(nullptr, fh, s.pyobj()));
            } catch (const std::exception &e) {
              Logger::error(
                  std::string(
                      "SKDECIDE exception when calling heuristic estimator: ") +
                  e.what());
              throw;
            }
          },
          discount, epsilon, verbose);
      _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(
          std::cout, py::module::import("sys").attr("stdout"));
      _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(
          std::cerr, py::module::import("sys").attr("stderr"));
    }

    virtual ~Implementation() {}

    void check_domain(py::object &domain) {
      if (!py::hasattr(domain, "get_applicable_actions")) {
        throw std::invalid_argument(
            "SKDECIDE exception: AO* algorithm needs python domain for "
            "implementing get_applicable_actions()");
      }
      if (!py::hasattr(domain, "get_next_state_distribution")) {
        throw std::invalid_argument(
            "SKDECIDE exception: AO* algorithm needs python domain for "
            "implementing get_next_state_distribution()");
      }
      if (!py::hasattr(domain, "get_transition_value")) {
        throw std::invalid_argument(
            "SKDECIDE exception: AO* algorithm needs python domain for "
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

    virtual py::int_ best_solution_graph_size() {
      return _solver->best_solution_graph_size();
    }

    virtual py::dict get_policy() {
      py::dict d;
      auto &&p = _solver->policy();
      for (auto &e : p) {
        d[e.first.pyobj()] =
            py::make_tuple(e.second.first.pyobj(), e.second.second);
      }
      return d;
    }

  private:
    std::unique_ptr<PyILAOStarDomain<Texecution>> _domain;
    std::unique_ptr<
        skdecide::ILAOStarSolver<PyILAOStarDomain<Texecution>, Texecution>>
        _solver;

    std::function<py::object(const py::object &, const py::object &)>
        _goal_checker;
    std::function<py::object(const py::object &, const py::object &)>
        _heuristic;

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
  PyILAOStarSolver(
      py::object &domain,
      const std::function<py::object(const py::object &, const py::object &)>
          &goal_checker,
      const std::function<py::object(const py::object &, const py::object &)>
          &heuristic,
      double discount = 1.0, double epsilon = 0.001, bool parallel = false,
      bool verbose = false) {

    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(domain, goal_checker, heuristic, discount, epsilon,
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

  py::float_ get_utility(const py::object &s) {
    return _implementation->get_utility(s);
  }

  py::int_ get_nb_of_explored_states() {
    return _implementation->get_nb_of_explored_states();
  }

  py::int_ best_solution_graph_size() {
    return _implementation->best_solution_graph_size();
  }

  py::dict get_policy() { return _implementation->get_policy(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_ILAOSTAR_HH
