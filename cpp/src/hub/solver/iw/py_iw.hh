/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_IW_HH
#define SKDECIDE_PY_IW_HH

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "utils/execution.hh"
#include "utils/python_gil_control.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/python_container_proxy.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "iw.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution> using PyIWDomain = PythonDomainProxy<Texecution>;

template <typename Texecution>
using PyIWFeatureVector = PythonContainerProxy<Texecution>;

template <typename Texecution>
using PyIWFeatureVector = skdecide::PythonContainerProxy<Texecution>;

class PyIWSolver {
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
    virtual py::int_ get_nb_of_pruned_states() = 0;
    virtual py::list get_intermediate_scores() = 0;
  };

  template <typename Texecution, template <typename...> class Thashing_policy>
  class Implementation : public BaseImplementation {
  public:
    Implementation(
        py::object &domain,
        const std::function<py::object(const py::object &, const py::object &)>
            &state_features,
        const std::function<bool(const double &, const std::size_t &,
                                 const std::size_t &, const double &,
                                 const std::size_t &, const std::size_t &)>
            &node_ordering = nullptr,
        std::size_t time_budget = 0, bool debug_logs = false)
        : _state_features(state_features), _node_ordering(node_ordering) {

      std::function<bool(const double &, const std::size_t &,
                         const std::size_t &, const double &,
                         const std::size_t &, const std::size_t &)>
          pno = nullptr;
      if (_node_ordering != nullptr) {
        pno = [this](const double &a_gscore, const std::size_t &a_novelty,
                     const std::size_t &a_depth, const double &b_gscore,
                     const std::size_t &b_novelty,
                     const std::size_t &b_depth) -> bool {
          typename skdecide::GilControl<Texecution>::Acquire acquire;
          try {
            return _node_ordering(a_gscore, a_novelty, a_depth, b_gscore,
                                  b_novelty, b_depth);
          } catch (const py::error_already_set *e) {
            Logger::error(
                "SKDECIDE exception when calling custom node ordering: " +
                std::string(e->what()));
            std::runtime_error err(e->what());
            delete e;
            throw err;
          }
        };
      }

      check_domain(domain);
      _domain = std::make_unique<PyIWDomain<Texecution>>(domain);
      _solver =
          std::make_unique<skdecide::IWSolver<PyIWDomain<Texecution>,
                                              PyIWFeatureVector<Texecution>,
                                              Thashing_policy, Texecution>>(
              *_domain,
              [this](PyIWDomain<Texecution> &d,
                     const typename PyIWDomain<Texecution>::State &s)
                  -> std::unique_ptr<PyIWFeatureVector<Texecution>> {
                try {
                  auto fsf = [this](const py::object &dd, const py::object &ss,
                                    [[maybe_unused]] const py::object &ii) {
                    return _state_features(dd, ss);
                  };
                  std::unique_ptr<py::object> r =
                      d.call(nullptr, fsf, s.pyobj());
                  typename skdecide::GilControl<Texecution>::Acquire acquire;
                  std::unique_ptr<PyIWFeatureVector<Texecution>> rr =
                      std::make_unique<PyIWFeatureVector<Texecution>>(*r);
                  r.reset();
                  return rr;
                } catch (const std::exception &e) {
                  Logger::error(
                      std::string(
                          "SKDECIDE exception when calling state features: ") +
                      e.what());
                  throw;
                }
              },
              pno, time_budget, debug_logs);
      _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(
          std::cout, py::module::import("sys").attr("stdout"));
      _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(
          std::cerr, py::module::import("sys").attr("stderr"));
    }

    virtual ~Implementation() {}

    void check_domain(py::object &domain) {
      if (!py::hasattr(domain, "get_applicable_actions")) {
        throw std::invalid_argument(
            "SKDECIDE exception: IW algorithm needs python domain for "
            "implementing get_applicable_actions()");
      }
      if (!py::hasattr(domain, "get_next_state")) {
        throw std::invalid_argument(
            "SKDECIDE exception: IW algorithm needs python domain for "
            "implementing get_next_state()");
      }
      if (!py::hasattr(domain, "get_transition_value")) {
        throw std::invalid_argument(
            "SKDECIDE exception: IW algorithm needs python domain for "
            "implementing get_transition_value()");
      }
      if (!py::hasattr(domain, "is_goal")) {
        throw std::invalid_argument("SKDECIDE exception: IW algorithm needs "
                                    "python domain for implementing is_goal()");
      }
      if (!py::hasattr(domain, "is_terminal")) {
        throw std::invalid_argument(
            "SKDECIDE exception: IW algorithm needs python domain for "
            "implementing is_terminal()");
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

    virtual py::int_ get_nb_of_pruned_states() {
      return _solver->get_nb_of_pruned_states();
    }

    virtual py::list get_intermediate_scores() {
      py::list l;
      for (const auto &p : _solver->get_intermediate_scores()) {
        l.append(
            py::make_tuple(std::get<0>(p), std::get<1>(p), std::get<2>(p)));
      }
      return l;
    }

  private:
    std::unique_ptr<PyIWDomain<Texecution>> _domain;
    std::unique_ptr<skdecide::IWSolver<PyIWDomain<Texecution>,
                                       PyIWFeatureVector<Texecution>,
                                       Thashing_policy, Texecution>>
        _solver;

    std::function<py::object(const py::object &, const py::object &)>
        _state_features;
    std::function<bool(const double &, const std::size_t &, const std::size_t &,
                       const double &, const std::size_t &,
                       const std::size_t &)>
        _node_ordering;

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

  struct HashingPolicySelector {
    bool _use_state_feature_hash;

    HashingPolicySelector(bool use_state_feature_hash)
        : _use_state_feature_hash(use_state_feature_hash) {}

    template <typename Propagator> struct Select {
      template <typename... Args>
      Select(HashingPolicySelector &This, Args... args) {
        if (This._use_state_feature_hash) {
          Propagator::template PushTemplate<StateFeatureHash>::Forward(args...);
        } else {
          Propagator::template PushTemplate<DomainStateHash>::Forward(args...);
        }
      }
    };
  };

  struct SolverInstantiator {
    std::unique_ptr<BaseImplementation> &_implementation;

    SolverInstantiator(std::unique_ptr<BaseImplementation> &implementation)
        : _implementation(implementation) {}

    template <typename... TypeInstantiations> struct TypeList {
      template <template <typename...> class... TemplateInstantiations>
      struct TemplateList {
        struct Instantiate {
          template <typename... Args>
          Instantiate(SolverInstantiator &This, Args... args) {
            This._implementation = std::make_unique<Implementation<
                TypeInstantiations..., TemplateInstantiations...>>(args...);
          }
        };
      };
    };
  };

  std::unique_ptr<BaseImplementation> _implementation;

public:
  PyIWSolver(
      py::object &domain,
      const std::function<py::object(const py::object &, const py::object &)>
          &state_features,
      bool use_state_feature_hash = false,
      const std::function<bool(const double &, const std::size_t &,
                               const std::size_t &, const double &,
                               const std::size_t &, const std::size_t &)>
          &node_ordering = nullptr,
      std::size_t time_budget = 0, bool parallel = false,
      bool debug_logs = false) {

    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 HashingPolicySelector(use_state_feature_hash),
                                 SolverInstantiator(_implementation))
        .instantiate(domain, state_features, node_ordering, time_budget,
                     debug_logs);
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

  py::int_ get_nb_of_pruned_states() {
    return _implementation->get_nb_of_pruned_states();
  }

  py::list get_intermediate_scores() {
    return _implementation->get_intermediate_scores();
  }
};

} // namespace skdecide

#endif // SKDECIDE_PY_IW_HH
