/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_HSVI_HH
#define SKDECIDE_PY_HSVI_HH

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include <optional>
#include <type_traits>

#include "utils/execution.hh"
#include "utils/python_gil_control.hh"
#include "utils/python_domain_proxy.hh"
#include "utils/template_instantiator.hh"
#include "utils/impl/python_domain_proxy_call_impl.hh"

#include "hsvi.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
using PyHSVIDomain =
    PythonDomainProxy<Texecution, SingleAgent, PartiallyObservable>;

struct HSVITag {};
struct GoalHSVITag {};

class PyHSVISolverBase {
protected:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void close() = 0;
    virtual void clear() = 0;
    virtual void solve(const py::object &s) = 0;
    virtual py::object get_next_action(const py::object &o) = 0;
    virtual py::object get_utility(const py::object &o) = 0;
    virtual py::bool_ is_solution_defined_for(const py::object &o) = 0;
    virtual void reset_belief() = 0;
    virtual py::object get_next_action_from_belief(const py::object &d) = 0;
    virtual py::object get_utility_from_belief(const py::object &d) = 0;
    virtual py::bool_
    is_solution_defined_for_from_belief(const py::object &d) = 0;
    virtual py::int_ get_nb_alpha_vectors() = 0;
    virtual py::int_ get_nb_bound_points() = 0;
    virtual py::int_ get_solving_time() = 0;
    virtual py::float_ get_gap() = 0;
    virtual py::list get_alpha_vectors() = 0;
  };

  template <typename Texecution, typename SolverTag = HSVITag>
  class Implementation : public BaseImplementation {
  public:
    using SolverType =
        std::conditional_t<std::is_same_v<SolverTag, GoalHSVITag>,
                           GoalHSVISolver<PyHSVIDomain<Texecution>, Texecution>,
                           HSVISolver<PyHSVIDomain<Texecution>, Texecution>>;

    using BaseSolverType = HSVISolver<PyHSVIDomain<Texecution>, Texecution>;

    Implementation(
        py::object &solver, py::object &domain,
        const std::function<py::object(const py::object &, const py::object &)>
            *goal_checker,
        double epsilon, double discount, std::size_t time_budget,
        std::size_t max_sample_depth, bool use_closed_list,
        double depth_bound_eta, std::size_t max_vi_iterations,
        double vi_convergence_factor, double prob_epsilon,
        double belief_hash_resolution,
        const std::function<py::bool_(const py::object &)> &callback,
        bool verbose, std::optional<double> dead_end_cost = std::nullopt)
        : _callback(callback) {

      _pysolver = std::make_unique<py::object>(solver);
      _domain = std::make_unique<PyHSVIDomain<Texecution>>(domain);

      if constexpr (std::is_same_v<SolverTag, GoalHSVITag>) {
        if (goal_checker) {
          _goal_checker = *goal_checker;
        }

        typename SolverType::GoalCheckerFunctor gc =
            [this](PyHSVIDomain<Texecution> &d,
                   const typename PyHSVIDomain<Texecution>::State &s) -> bool {
          auto fgc = [this](const py::object &dd, const py::object &ss,
                            [[maybe_unused]] const py::object &ii) {
            return _goal_checker(dd, ss);
          };
          std::unique_ptr<py::object> r = d.call(nullptr, fgc, s.pyobj());
          typename GilControl<Texecution>::Acquire acquire;
          bool rr = r->template cast<bool>();
          r.reset();
          return rr;
        };

        _solver = std::make_unique<SolverType>(
            *_domain, gc, epsilon, discount, time_budget, max_sample_depth,
            use_closed_list, depth_bound_eta, max_vi_iterations,
            vi_convergence_factor, prob_epsilon, belief_hash_resolution,
            [this](const BaseSolverType &s,
                   PyHSVIDomain<Texecution> &d) -> bool {
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
            verbose, dead_end_cost);
      } else {
        _solver = std::make_unique<SolverType>(
            *_domain, epsilon, discount, time_budget, max_sample_depth,
            use_closed_list, depth_bound_eta, max_vi_iterations,
            vi_convergence_factor, prob_epsilon, belief_hash_resolution,
            [this](const BaseSolverType &s,
                   PyHSVIDomain<Texecution> &d) -> bool {
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
      }

      _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(
          std::cout, py::module::import("sys").attr("stdout"));
      _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(
          std::cerr, py::module::import("sys").attr("stderr"));
    }

    virtual ~Implementation() {}

    virtual void close() { _domain->close(); }
    virtual void clear() { _solver->clear(); }

    virtual void solve(const py::object &distribution) {
      std::vector<std::pair<typename PyHSVIDomain<Texecution>::State, double>>
          dist;
      py::list values = distribution.attr("get_values")();
      for (auto item : values) {
        py::tuple t = item.cast<py::tuple>();
        dist.emplace_back(typename PyHSVIDomain<Texecution>::State(t[0]),
                          t[1].cast<double>());
      }
      typename GilControl<Texecution>::Release release;
      _solver->solve(dist);
    }

    virtual py::object get_next_action(const py::object &o) {
      try {
        return _solver
            ->get_best_action(typename PyHSVIDomain<Texecution>::Observation(o))
            .pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[HSVI.get_next_action] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility(const py::object &o) {
      try {
        return _solver
            ->get_best_value(typename PyHSVIDomain<Texecution>::Observation(o))
            .pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[HSVI.get_utility] ") + e.what() +
                     " - returning None");
        return py::none();
      }
    }

    virtual py::bool_ is_solution_defined_for(const py::object &o) {
      return _solver->is_solution_defined_for(
          typename PyHSVIDomain<Texecution>::Observation(o));
    }

    virtual void reset_belief() { _solver->reset_belief(); }

    virtual py::object get_next_action_from_belief(const py::object &d) {
      try {
        auto belief = distribution_to_belief(d);
        return _solver->get_best_action_from_belief(belief).pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[HSVI.get_next_action_from_belief] ") +
                     e.what() + " - returning None");
        return py::none();
      }
    }

    virtual py::object get_utility_from_belief(const py::object &d) {
      try {
        auto belief = distribution_to_belief(d);
        return _solver->get_best_value_from_belief(belief).pyobj();
      } catch (const std::runtime_error &e) {
        Logger::warn(std::string("[HSVI.get_utility_from_belief] ") + e.what() +
                     " - returning None");
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

    virtual py::int_ get_nb_bound_points() {
      return _solver->get_nb_bound_points();
    }

    virtual py::int_ get_solving_time() { return _solver->get_solving_time(); }

    virtual py::float_ get_gap() { return _solver->get_gap(); }

    virtual py::list get_alpha_vectors() {
      py::list result;
      const auto &alphas = _solver->get_alpha_vectors();
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
              typename PyHSVIDomain<Texecution>::Value val(alpha.values[idx],
                                                           true);
              values_dict[state.pyobj()] = val.pyobj();
            }
          }
        }

        alpha_dict["values"] = values_dict;
        alpha_dict["action"] = alpha.action.pyobj();
        alpha_dict["id"] = alpha.id;
        result.append(alpha_dict);
      }

      return result;
    }

  private:
    typename BaseSolverType::Belief
    distribution_to_belief(const py::object &d) {
      typename BaseSolverType::Belief b;
      py::list values = d.attr("get_values")();
      for (auto item : values) {
        py::tuple t = item.cast<py::tuple>();
        typename PyHSVIDomain<Texecution>::State state(t[0]);
        std::size_t sh =
            typename PyHSVIDomain<Texecution>::State::Hash()(state);
        b[sh] = t[1].cast<double>();
      }
      return b;
    }

    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<PyHSVIDomain<Texecution>> _domain;
    std::unique_ptr<SolverType> _solver;

    std::function<py::bool_(const py::object &)> _callback;
    std::function<py::object(const py::object &, const py::object &)>
        _goal_checker;

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

  std::unique_ptr<BaseImplementation> _implementation;

public:
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

  py::int_ get_nb_bound_points() {
    return _implementation->get_nb_bound_points();
  }

  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
  py::float_ get_gap() { return _implementation->get_gap(); }
  py::list get_alpha_vectors() { return _implementation->get_alpha_vectors(); }
};

class PyHSVISolver : public PyHSVISolverBase {
  struct SolverInstantiator {
    std::unique_ptr<BaseImplementation> &_implementation;
    SolverInstantiator(std::unique_ptr<BaseImplementation> &implementation)
        : _implementation(implementation) {}
    template <typename... TypeInstantiations> struct Instantiate {
      template <typename... Args>
      Instantiate(SolverInstantiator &This, Args... args) {
        This._implementation =
            std::make_unique<Implementation<TypeInstantiations..., HSVITag>>(
                args...);
      }
    };
  };

public:
  PyHSVISolver(
      py::object &solver, py::object &domain, double epsilon = 0.001,
      double discount = 0.95, std::size_t time_budget = 300000,
      std::size_t max_sample_depth = 100, bool use_closed_list = false,
      double depth_bound_eta = 0.1, std::size_t max_vi_iterations = 1000,
      double vi_convergence_factor = 0.01, double prob_epsilon = 1e-15,
      double belief_hash_resolution = 1000.0, bool parallel = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain,
                     static_cast<const std::function<py::object(
                         const py::object &, const py::object &)> *>(nullptr),
                     epsilon, discount, time_budget, max_sample_depth,
                     use_closed_list, depth_bound_eta, max_vi_iterations,
                     vi_convergence_factor, prob_epsilon,
                     belief_hash_resolution, callback, verbose);
  }
};

class PyGoalHSVISolver : public PyHSVISolverBase {
  struct SolverInstantiator {
    std::unique_ptr<BaseImplementation> &_implementation;
    SolverInstantiator(std::unique_ptr<BaseImplementation> &implementation)
        : _implementation(implementation) {}
    template <typename... TypeInstantiations> struct Instantiate {
      template <typename... Args>
      Instantiate(SolverInstantiator &This, Args... args) {
        This._implementation = std::make_unique<
            Implementation<TypeInstantiations..., GoalHSVITag>>(args...);
      }
    };
  };

public:
  PyGoalHSVISolver(
      py::object &solver, py::object &domain,
      const std::function<py::object(const py::object &, const py::object &)>
          &goal_checker,
      double epsilon = 0.001, double discount = 1.0,
      std::size_t time_budget = 300000, std::size_t max_sample_depth = 100,
      bool use_closed_list = true, double depth_bound_eta = 0.1,
      std::size_t max_vi_iterations = 1000, double vi_convergence_factor = 0.01,
      double prob_epsilon = 1e-15, double belief_hash_resolution = 1000.0,
      bool parallel = false,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false,
      std::optional<double> dead_end_cost = std::nullopt) {
    TemplateInstantiator::select(ExecutionSelector(parallel),
                                 SolverInstantiator(_implementation))
        .instantiate(solver, domain, &goal_checker, epsilon, discount,
                     time_budget, max_sample_depth, use_closed_list,
                     depth_bound_eta, max_vi_iterations, vi_convergence_factor,
                     prob_epsilon, belief_hash_resolution, callback, verbose,
                     dead_end_cost);
  }
};

} // namespace skdecide

#endif // SKDECIDE_PY_HSVI_HH
