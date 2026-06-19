/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_POMCP_HH
#define SKDECIDE_PY_POMCP_HH

#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

#include "utils/execution.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/python_gil_control.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "pomcp.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PyPOMCPDomain =
    PythonDomainProxy<Texecution, SingleAgent, PartiallyObservable>;

class PyPOMCPSolver {
private:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void close() = 0;
    virtual void clear() = 0;
    virtual void solve(const py::object &distribution) = 0;
    virtual py::object get_next_action(const py::object &o) = 0;
    virtual py::object get_utility(const py::object &o) = 0;
    virtual py::bool_ is_solution_defined_for(const py::object &o) = 0;
    virtual void reset_belief() = 0;
    virtual py::object get_next_action_from_belief(const py::object &d) = 0;
    virtual py::object get_utility_from_belief(const py::object &d) = 0;
    virtual py::bool_
    is_solution_defined_for_from_belief(const py::object &d) = 0;
    virtual py::int_ get_nb_tree_nodes() = 0;
    virtual py::int_ get_solving_time() = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    Implementation(
        py::object &solver, py::object &domain,
        double exploration_constant = 1.0 / std::sqrt(2.0),
        double discount = 0.95, std::size_t num_simulations = 1000,
        std::size_t max_depth = 100, double epsilon = 0.001,
        std::size_t time_budget = 0,
        std::size_t num_particles_belief_update = 500,
        double ess_threshold_ratio = 2.0,
        const std::function<py::bool_(const py::object &)> &callback = nullptr,
        bool verbose = false)
        : _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      _domain = std::make_unique<PyPOMCPDomain<Texecution>>(domain);

      _solver =
          std::make_unique<POMCPSolver<PyPOMCPDomain<Texecution>, Texecution>>(
              *_domain, exploration_constant, discount, num_simulations,
              max_depth, epsilon, time_budget, num_particles_belief_update,
              ess_threshold_ratio,
              [this](
                  const POMCPSolver<PyPOMCPDomain<Texecution>, Texecution> &s,
                  PyPOMCPDomain<Texecution> &d) -> bool {
                if (_callback) {
                  try {
                    typename GilControl<Texecution>::Acquire acquire;
                    py::bool_ r = _callback(*_pysolver);
                    bool rr = r.template cast<bool>();
                    return rr;
                  } catch (const std::exception &e) {
                    Logger::error(
                        std::string(
                            "SKDECIDE exception when calling callback: ") +
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
      std::vector<std::pair<typename PyPOMCPDomain<Texecution>::State, double>>
          dist;
      py::list values = distribution.attr("get_values")();
      for (auto item : values) {
        py::tuple t = item.cast<py::tuple>();
        dist.emplace_back(typename PyPOMCPDomain<Texecution>::State(t[0]),
                          t[1].cast<double>());
      }
      typename GilControl<Texecution>::Release release;
      _solver->solve(dist);
    }

    virtual py::object get_next_action(const py::object &o) {
      try {
        typename PyPOMCPDomain<Texecution>::Observation obs(o);
        const typename PyPOMCPDomain<Texecution>::Action *action_ptr;
        {
          typename GilControl<Texecution>::Release release;
          action_ptr = &_solver->get_best_action(obs);
        }
        return action_ptr->pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[POMCP.get_next_action] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility(const py::object &o) {
      try {
        typename PyPOMCPDomain<Texecution>::Observation obs(o);
        typename PyPOMCPDomain<Texecution>::Value val;
        {
          typename GilControl<Texecution>::Release release;
          val = _solver->get_best_value(obs);
        }
        return val.pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[POMCP.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::bool_ is_solution_defined_for(const py::object &o) {
      return _solver->is_solution_defined_for(
          typename PyPOMCPDomain<Texecution>::Observation(o));
    }

    virtual void reset_belief() { _solver->reset_belief(); }

    virtual py::object get_next_action_from_belief(const py::object &d) {
      try {
        auto belief = distribution_to_belief(d);
        const typename PyPOMCPDomain<Texecution>::Action *action_ptr;
        {
          typename GilControl<Texecution>::Release release;
          action_ptr = &_solver->get_best_action_from_belief(belief);
        }
        return action_ptr->pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[POMCP.get_next_action_from_belief] ") +
                     e.what() + " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility_from_belief(const py::object &d) {
      try {
        auto belief = distribution_to_belief(d);
        typename PyPOMCPDomain<Texecution>::Value val;
        {
          typename GilControl<Texecution>::Release release;
          val = _solver->get_best_value_from_belief(belief);
        }
        return val.pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[POMCP.get_utility_from_belief] ") +
                     e.what() + " - returning None");
        return py::none();
      }
    }

    virtual py::bool_ is_solution_defined_for_from_belief(const py::object &d) {
      auto belief = distribution_to_belief(d);
      return _solver->is_solution_defined_for_from_belief(belief);
    }

    virtual py::int_ get_nb_tree_nodes() {
      return _solver->get_nb_tree_nodes();
    }

    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }

  private:
    typedef POMCPSolver<PyPOMCPDomain<Texecution>, Texecution> SolverType;

    typename SolverType::Belief distribution_to_belief(const py::object &d) {
      typename SolverType::Belief b;
      py::list values = d.attr("get_values")();
      for (auto item : values) {
        py::tuple t = item.cast<py::tuple>();
        typename PyPOMCPDomain<Texecution>::State state(t[0]);
        std::size_t idx = _solver->get_state_index(state);
        b[idx] = t[1].cast<double>();
      }
      return b;
    }

    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<PyPOMCPDomain<Texecution>> _domain;
    std::unique_ptr<POMCPSolver<PyPOMCPDomain<Texecution>, Texecution>> _solver;

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
  PyPOMCPSolver(
      py::object &solver, py::object &domain,
      double exploration_constant = 1.0 / std::sqrt(2.0),
      double discount = 0.95, std::size_t num_simulations = 1000,
      std::size_t max_depth = 100, double epsilon = 0.001,
      std::size_t time_budget = 0,
      std::size_t num_particles_belief_update = 500,
      double ess_threshold_ratio = 2.0, bool parallel = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, exploration_constant, discount,
                     num_simulations, max_depth, epsilon, time_budget,
                     num_particles_belief_update, ess_threshold_ratio, callback,
                     verbose);
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

  py::int_ get_nb_tree_nodes() { return _implementation->get_nb_tree_nodes(); }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_POMCP_HH
