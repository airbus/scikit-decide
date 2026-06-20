/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_WITNESS_HH
#define SKDECIDE_PY_WITNESS_HH

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "utils/execution.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/python_gil_control.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "witness.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PyWitnessDomain =
    PythonDomainProxy<Texecution, SingleAgent, PartiallyObservable>;

class PyWitnessSolver {
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
    virtual py::int_ get_nb_alpha_vectors() = 0;
    virtual py::int_ get_nb_iterations() = 0;
    virtual py::int_ get_solving_time() = 0;
    virtual py::str get_callback_event() = 0;
    virtual py::list get_alpha_vectors() = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    Implementation(
        py::object &solver, py::object &domain, double epsilon = 0.001,
        double discount = 0.95, std::size_t max_iterations = 100,
        double lp_infinity = 1e20, double lp_tolerance = 1e-10,
        const std::function<py::bool_(const py::object &)> &callback = nullptr,
        bool verbose = false)
        : _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      _domain = std::make_unique<PyWitnessDomain<Texecution>>(domain);
      _solver = std::make_unique<
          WitnessSolver<PyWitnessDomain<Texecution>, Texecution>>(
          *_domain, epsilon, discount, max_iterations, lp_infinity,
          lp_tolerance,
          [this](
              const WitnessSolver<PyWitnessDomain<Texecution>, Texecution> &s,
              PyWitnessDomain<Texecution> &d) -> bool {
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
      std::vector<
          std::pair<typename PyWitnessDomain<Texecution>::State, double>>
          dist;
      py::list values = distribution.attr("get_values")();
      for (auto item : values) {
        py::tuple t = item.cast<py::tuple>();
        dist.emplace_back(typename PyWitnessDomain<Texecution>::State(t[0]),
                          t[1].cast<double>());
      }
      typename GilControl<Texecution>::Release release;
      _solver->solve(dist);
    }

    virtual py::object get_next_action(const py::object &o) {
      try {
        return _solver
            ->get_best_action(
                typename PyWitnessDomain<Texecution>::Observation(o))
            .pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[Witness.get_next_action] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility(const py::object &o) {
      try {
        return _solver
            ->get_best_value(
                typename PyWitnessDomain<Texecution>::Observation(o))
            .pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[Witness.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::bool_ is_solution_defined_for(const py::object &o) {
      return _solver->is_solution_defined_for(
          typename PyWitnessDomain<Texecution>::Observation(o));
    }

    virtual void reset_belief() { _solver->reset_belief(); }

    virtual py::object get_next_action_from_belief(const py::object &d) {
      try {
        auto belief = distribution_to_belief(d);
        return _solver->get_best_action_from_belief(belief).pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[Witness.get_next_action_from_belief] ") +
                     e.what() + " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility_from_belief(const py::object &d) {
      try {
        auto belief = distribution_to_belief(d);
        return _solver->get_best_value_from_belief(belief).pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[Witness.get_utility_from_belief] ") +
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

    virtual py::int_ get_nb_iterations() {
      return _solver->get_nb_iterations();
    }

    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }

    virtual py::str get_callback_event() {
      using LPCallbackEvent =
          typename WitnessSolver<PyWitnessDomain<Texecution>,
                                 Texecution>::LPCallbackEvent;
      switch (_solver->get_callback_event()) {
      case LPCallbackEvent::SolverIteration:
        return py::str("SolverIteration");
      case LPCallbackEvent::LPProgress:
        return py::str("LPProgress");
      default:
        return py::str("Unknown");
      }
    }

    virtual py::list get_alpha_vectors() {
      py::list result;
      const auto &alphas = _solver->get_alpha_vectors();
      const auto &actions = _solver->get_action_list();
      const auto &index_to_state = _solver->get_index_to_state();
      const auto &state_hash_to_idx = _solver->get_state_hash_to_idx();

      for (const auto &alpha : alphas) {
        py::dict alpha_dict;
        py::dict values_dict;

        // Map each enumerated state to its value in the alpha vector
        for (const auto &[hash, state] : index_to_state) {
          auto idx_it = state_hash_to_idx.find(hash);
          if (idx_it != state_hash_to_idx.end()) {
            std::size_t idx = idx_it->second;
            if (idx < alpha.values.size()) {
              typename PyWitnessDomain<Texecution>::Value val(alpha.values[idx],
                                                              true);
              values_dict[state.pyobj()] = val.pyobj();
            }
          }
        }

        alpha_dict["values"] = values_dict;
        // Convert action index to action
        if (alpha.action_idx < actions.size()) {
          alpha_dict["action"] = actions[alpha.action_idx].pyobj();
        }
        result.append(alpha_dict);
      }

      return result;
    }

  private:
    typedef WitnessSolver<PyWitnessDomain<Texecution>, Texecution> SolverType;

    typename SolverType::Belief distribution_to_belief(const py::object &d) {
      typename SolverType::Belief b;
      py::list values = d.attr("get_values")();
      for (auto item : values) {
        py::tuple t = item.cast<py::tuple>();
        typename PyWitnessDomain<Texecution>::State state(t[0]);
        std::size_t idx = _solver->get_state_index(state);
        b[idx] = t[1].cast<double>();
      }
      return b;
    }

    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<PyWitnessDomain<Texecution>> _domain;
    std::unique_ptr<WitnessSolver<PyWitnessDomain<Texecution>, Texecution>>
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
  PyWitnessSolver(
      py::object &solver, py::object &domain, double epsilon = 0.001,
      double discount = 0.95, std::size_t max_iterations = 100,
      double lp_infinity = 1e20, double lp_tolerance = 1e-10,
      bool parallel = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, epsilon, discount, max_iterations,
                     lp_infinity, lp_tolerance, callback, verbose);
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

  py::int_ get_nb_iterations() { return _implementation->get_nb_iterations(); }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
  py::str get_callback_event() { return _implementation->get_callback_event(); }
  py::list get_alpha_vectors() { return _implementation->get_alpha_vectors(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_WITNESS_HH
