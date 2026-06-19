/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_SARSOP_HH
#define SKDECIDE_PY_SARSOP_HH

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "utils/execution.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/python_gil_control.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "sarsop.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PySARSOPDomain =
    PythonDomainProxy<Texecution, SingleAgent, PartiallyObservable>;

class PySARSOPSolver {
private:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void close() = 0;
    virtual void clear() = 0;
    virtual void solve(const py::object &distribution) = 0;
    // Observation-based interface
    virtual py::object get_next_action(const py::object &o) = 0;
    virtual py::object get_utility(const py::object &o) = 0;
    virtual py::bool_ is_solution_defined_for(const py::object &o) = 0;
    virtual void reset_belief() = 0;
    // Belief-based interface
    virtual py::object get_next_action_from_belief(const py::object &d) = 0;
    virtual py::object get_utility_from_belief(const py::object &d) = 0;
    virtual py::bool_
    is_solution_defined_for_from_belief(const py::object &d) = 0;
    // Statistics
    virtual py::int_ get_nb_alpha_vectors() = 0;
    virtual py::int_ get_nb_explored_beliefs() = 0;
    virtual py::int_ get_solving_time() = 0;
    virtual py::float_ get_lower_bound() = 0;
    virtual py::float_ get_upper_bound() = 0;
    virtual py::float_ get_gap() = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    Implementation(
        py::object &solver, py::object &domain, double epsilon = 0.001,
        double discount = 0.95, std::size_t time_budget = 300000,
        std::size_t max_beliefs = 100000, double pruning_delta = 1e-6,
        std::size_t max_vi_iterations = 1000,
        double vi_convergence_factor = 0.01, std::size_t max_sample_depth = 100,
        double prob_epsilon = 1e-15, double ub_improvement_epsilon = 1e-10,
        std::size_t pruning_interval = 10, std::size_t logging_interval = 50,
        const std::function<py::bool_(const py::object &)> &callback = nullptr,
        bool verbose = false)
        : _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      _domain = std::make_unique<PySARSOPDomain<Texecution>>(domain);
      _solver = std::make_unique<
          SARSOPSolver<PySARSOPDomain<Texecution>, Texecution>>(
          *_domain, epsilon, discount, time_budget, max_beliefs, pruning_delta,
          max_vi_iterations, vi_convergence_factor, max_sample_depth,
          prob_epsilon, ub_improvement_epsilon, pruning_interval,
          logging_interval,
          [this](const SARSOPSolver<PySARSOPDomain<Texecution>, Texecution> &s,
                 PySARSOPDomain<Texecution> &d) -> bool {
            if (_callback) {
              try {
                return _callback(*_pysolver);
              } catch (const std::exception &e) {
                Logger::error(std::string("SKDECIDE exception when calling "
                                          "callback: ") +
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

    virtual void solve(const py::object &distribution) {
      std::vector<std::pair<typename PySARSOPDomain<Texecution>::State, double>>
          dist;
      py::list values = distribution.attr("get_values")();
      for (auto item : values) {
        py::tuple t = item.cast<py::tuple>();
        dist.emplace_back(typename PySARSOPDomain<Texecution>::State(t[0]),
                          t[1].cast<double>());
      }
      typename GilControl<Texecution>::Release release;
      _solver->solve(dist);
    }

    virtual py::object get_next_action(const py::object &o) {
      try {
        return _solver
            ->get_best_action(
                typename PySARSOPDomain<Texecution>::Observation(o))
            .pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[SARSOP.get_next_action] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility(const py::object &o) {
      try {
        return _solver
            ->get_best_value(
                typename PySARSOPDomain<Texecution>::Observation(o))
            .pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[SARSOP.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::bool_ is_solution_defined_for(const py::object &o) {
      return _solver->is_solution_defined_for(
          typename PySARSOPDomain<Texecution>::Observation(o));
    }

    virtual void reset_belief() { _solver->reset_belief(); }

    virtual py::object get_next_action_from_belief(const py::object &d) {
      try {
        auto belief = distribution_to_belief(d);
        return _solver->get_best_action_from_belief(belief).pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[SARSOP.get_next_action_from_belief] ") +
                     e.what() + " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility_from_belief(const py::object &d) {
      try {
        auto belief = distribution_to_belief(d);
        return _solver->get_best_value_from_belief(belief).pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[SARSOP.get_utility_from_belief] ") +
                     e.what() + " - returning None");
        return py::none();
      }
    }

    virtual py::bool_ is_solution_defined_for_from_belief(const py::object &d) {
      auto belief = distribution_to_belief(d);
      return _solver->is_solution_defined_for_from_belief(belief);
    }

    virtual py::int_ get_nb_alpha_vectors() {
      return _solver->get_nb_alpha_vectors();
    }

    virtual py::int_ get_nb_explored_beliefs() {
      return _solver->get_nb_explored_beliefs();
    }

    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }

    virtual py::float_ get_lower_bound() {
      return _solver->get_initial_lower_bound();
    }

    virtual py::float_ get_upper_bound() {
      return _solver->get_initial_upper_bound();
    }

    virtual py::float_ get_gap() { return _solver->get_gap(); }

  private:
    typedef SARSOPSolver<PySARSOPDomain<Texecution>, Texecution> SolverType;

    typename SolverType::Belief distribution_to_belief(const py::object &d) {
      typename SolverType::Belief b;
      py::list values = d.attr("get_values")();
      for (auto item : values) {
        py::tuple t = item.cast<py::tuple>();
        typename PySARSOPDomain<Texecution>::State state(t[0]);
        std::size_t idx = _solver->get_state_index(state);
        b[idx] = t[1].cast<double>();
      }
      return b;
    }

    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<PySARSOPDomain<Texecution>> _domain;
    std::unique_ptr<SARSOPSolver<PySARSOPDomain<Texecution>, Texecution>>
        _solver;

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
  PySARSOPSolver(
      py::object &solver, py::object &domain, double epsilon = 0.001,
      double discount = 0.95, std::size_t time_budget = 300000,
      std::size_t max_beliefs = 100000, double pruning_delta = 1e-6,
      std::size_t max_vi_iterations = 1000, double vi_convergence_factor = 0.01,
      std::size_t max_sample_depth = 100, double prob_epsilon = 1e-15,
      double ub_improvement_epsilon = 1e-10, std::size_t pruning_interval = 10,
      std::size_t logging_interval = 50, bool parallel = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, epsilon, discount, time_budget,
                     max_beliefs, pruning_delta, max_vi_iterations,
                     vi_convergence_factor, max_sample_depth, prob_epsilon,
                     ub_improvement_epsilon, pruning_interval, logging_interval,
                     callback, verbose);
  }

  void close() { _implementation->close(); }
  void clear() { _implementation->clear(); }
  void solve(const py::object &s) { _implementation->solve(s); }

  py::object get_next_action(const py::object &o) {
    return _implementation->get_next_action(o);
  }

  py::object get_utility(const py::object &o) {
    return _implementation->get_utility(o);
  }

  py::bool_ is_solution_defined_for(const py::object &o) {
    return _implementation->is_solution_defined_for(o);
  }

  void reset_belief() { _implementation->reset_belief(); }

  py::object get_next_action_from_belief(const py::object &d) {
    return _implementation->get_next_action_from_belief(d);
  }

  py::object get_utility_from_belief(const py::object &d) {
    return _implementation->get_utility_from_belief(d);
  }

  py::bool_ is_solution_defined_for_from_belief(const py::object &d) {
    return _implementation->is_solution_defined_for_from_belief(d);
  }

  py::int_ get_nb_alpha_vectors() {
    return _implementation->get_nb_alpha_vectors();
  }

  py::int_ get_nb_explored_beliefs() {
    return _implementation->get_nb_explored_beliefs();
  }

  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
  py::float_ get_lower_bound() { return _implementation->get_lower_bound(); }
  py::float_ get_upper_bound() { return _implementation->get_upper_bound(); }
  py::float_ get_gap() { return _implementation->get_gap(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_SARSOP_HH
