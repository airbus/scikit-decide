/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_PPDDLPLANMERGER_HH
#define SKDECIDE_PY_PPDDLPLANMERGER_HH

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hub/solver/determinization/determinized_domain.hh"
#include "hub/solver/pddl/inner_solver/impl/pddl_inner_solver_registry_impl.hh"
#include "hub/solver/pddl/inner_solver/pddl_inner_solver_registry.hh"
#include "hub/solver/pddl/ppddlreplan/ff_inner_solver.hh"
#include "impl/ppddlplanmerger_impl.hh"
#include "impl/rff_impl.hh"
#include "ppddlplanmerger.hh"
#include "rff.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace py = pybind11;

namespace skdecide {

class PyRFFSolver {
private:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void solve(const pddl::State &s) = 0;
    virtual void resolve(const pddl::State &s) = 0;
    virtual void clear() = 0;
    virtual bool is_solution_defined_for(const pddl::State &s) = 0;
    virtual py::object get_next_action(const pddl::State &s) = 0;
    virtual std::size_t get_nb_iterations() = 0;
    virtual std::size_t get_nb_plans() = 0;
    virtual std::size_t get_solving_time() = 0;
    virtual std::size_t get_policy_size() = 0;
    virtual double get_best_value(const pddl::State &s) = 0;
    virtual py::set get_explored_states() = 0;
    virtual py::set get_terminal_states() = 0;
    virtual py::dict get_policy() = 0;
  };

  template <typename Texecution, typename TstrategyTag>
  class Implementation : public BaseImplementation {
  public:
    Implementation(py::object &pysolver, const pddl::Task &task,
                   double dead_end_cost, double rho, std::size_t mc_samples,
                   std::size_t max_iterations, std::size_t max_steps,
                   bool optimize_policy_graph, double discount, double epsilon,
                   const std::function<py::bool_(const py::object &)> &callback,
                   bool verbose)
        : _pysolver(std::make_unique<py::object>(pysolver)) {

      typename pddl::RFFSolver<Texecution, TstrategyTag>::CallbackFunctor cb =
          [](const pddl::RFFSolver<Texecution, TstrategyTag> &) {
            return false;
          };

      if (callback) {
        _callback = callback;
        cb = [this](const pddl::RFFSolver<Texecution, TstrategyTag> &) -> bool {
          try {
            return _callback(*_pysolver);
          } catch (const std::exception &e) {
            Logger::error(
                std::string("SKDECIDE exception when calling RFF callback: ") +
                e.what());
            throw;
          }
        };
      }

      _solver = std::make_unique<pddl::RFFSolver<Texecution, TstrategyTag>>(
          task, dead_end_cost, rho, mc_samples, max_iterations, max_steps,
          optimize_policy_graph, discount, epsilon, cb, verbose);
    }

    virtual void solve(const pddl::State &s) override { _solver->solve(s); }
    virtual void resolve(const pddl::State &s) override { _solver->resolve(s); }
    virtual void clear() override { _solver->clear(); }

    virtual bool is_solution_defined_for(const pddl::State &s) override {
      return _solver->is_solution_defined_for(s);
    }

    virtual py::object get_next_action(const pddl::State &s) override {
      try {
        const auto &action = _solver->get_best_action(s);
        return py::cast(static_cast<const pddl::GroundAction &>(action));
      } catch (const std::runtime_error &) {
        return py::none();
      }
    }

    virtual double get_best_value(const pddl::State &s) override {
      return _solver->get_best_value(s);
    }

    virtual std::size_t get_nb_iterations() override {
      return _solver->get_nb_iterations();
    }

    virtual std::size_t get_nb_plans() override {
      return _solver->get_nb_plans();
    }

    virtual std::size_t get_solving_time() override {
      return _solver->get_solving_time();
    }

    virtual std::size_t get_policy_size() override {
      return _solver->get_policy_size();
    }

    virtual py::set get_explored_states() override {
      py::set s;
      for (auto &e : _solver->get_explored_states()) {
        s.add(py::cast(static_cast<const pddl::State &>(e)));
      }
      return s;
    }

    virtual py::set get_terminal_states() override {
      py::set s;
      for (auto &e : _solver->get_terminal_states()) {
        s.add(py::cast(static_cast<const pddl::State &>(e)));
      }
      return s;
    }

    virtual py::dict get_policy() override {
      py::dict d;
      for (auto &e : _solver->get_policy()) {
        d[py::cast(static_cast<const pddl::State &>(e.first))] = py::make_tuple(
            py::cast(static_cast<const pddl::GroundAction &>(e.second.first)),
            e.second.second);
      }
      return d;
    }

  private:
    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<pddl::RFFSolver<Texecution, TstrategyTag>> _solver;
    std::function<py::bool_(const py::object &)> _callback;
  };

  std::unique_ptr<BaseImplementation> _implementation;

  template <typename Texecution>
  void create_impl(const std::string &determinization, py::object &solver,
                   const pddl::Task &task, double dead_end_cost, double rho,
                   std::size_t mc_samples, std::size_t max_iterations,
                   std::size_t max_steps, bool optimize_policy_graph,
                   double discount, double epsilon,
                   const std::function<py::bool_(const py::object &)> &callback,
                   bool verbose) {
    if (determinization == "all_outcomes") {
      _implementation =
          std::make_unique<Implementation<Texecution, AllOutcomesStrategy>>(
              solver, task, dead_end_cost, rho, mc_samples, max_iterations,
              max_steps, optimize_policy_graph, discount, epsilon, callback,
              verbose);
    } else if (determinization == "random_outcome") {
      _implementation =
          std::make_unique<Implementation<Texecution, RandomOutcomeStrategy>>(
              solver, task, dead_end_cost, rho, mc_samples, max_iterations,
              max_steps, optimize_policy_graph, discount, epsilon, callback,
              verbose);
    } else {
      _implementation = std::make_unique<
          Implementation<Texecution, MostProbableOutcomeStrategy>>(
          solver, task, dead_end_cost, rho, mc_samples, max_iterations,
          max_steps, optimize_policy_graph, discount, epsilon, callback,
          verbose);
    }
  }

public:
  PyRFFSolver(
      py::object &solver, const pddl::Task &task,
      const std::string &determinization = "most_probable_outcome",
      bool parallel = false, double dead_end_cost = 1e9, double rho = 0.1,
      std::size_t mc_samples = 100, std::size_t max_iterations = 50,
      std::size_t max_steps = 10000, bool optimize_policy_graph = false,
      double discount = 0.99, double epsilon = 1e-3,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    if (parallel) {
      create_impl<ParallelExecution>(
          determinization, solver, task, dead_end_cost, rho, mc_samples,
          max_iterations, max_steps, optimize_policy_graph, discount, epsilon,
          callback, verbose);
    } else {
      create_impl<SequentialExecution>(
          determinization, solver, task, dead_end_cost, rho, mc_samples,
          max_iterations, max_steps, optimize_policy_graph, discount, epsilon,
          callback, verbose);
    }
  }

  void solve(const pddl::State &s) { _implementation->solve(s); }
  void resolve(const pddl::State &s) { _implementation->resolve(s); }
  void clear() { _implementation->clear(); }

  py::bool_ is_solution_defined_for(const pddl::State &s) {
    return _implementation->is_solution_defined_for(s);
  }

  py::object get_next_action(const pddl::State &s) {
    return _implementation->get_next_action(s);
  }

  py::float_ get_best_value(const pddl::State &s) {
    return _implementation->get_best_value(s);
  }

  py::int_ get_nb_iterations() { return _implementation->get_nb_iterations(); }
  py::int_ get_nb_plans() { return _implementation->get_nb_plans(); }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
  py::int_ get_policy_size() { return _implementation->get_policy_size(); }

  py::set get_explored_states() {
    return _implementation->get_explored_states();
  }
  py::set get_terminal_states() {
    return _implementation->get_terminal_states();
  }
  py::dict get_policy() { return _implementation->get_policy(); }
};

class PyPPDDLPlanMergerSolver {
private:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void solve(const pddl::State &s) = 0;
    virtual void resolve(const pddl::State &s) = 0;
    virtual void clear() = 0;
    virtual bool is_solution_defined_for(const pddl::State &s) = 0;
    virtual py::object get_next_action(const pddl::State &s) = 0;
    virtual std::size_t get_nb_iterations() = 0;
    virtual std::size_t get_nb_plans() = 0;
    virtual std::size_t get_solving_time() = 0;
    virtual std::size_t get_policy_size() = 0;
    virtual double get_best_value(const pddl::State &s) = 0;
    virtual py::set get_explored_states() = 0;
    virtual py::set get_terminal_states() = 0;
    virtual py::dict get_policy() = 0;
  };

  template <typename Texecution, typename TstrategyTag>
  class Implementation : public BaseImplementation {
  public:
    Implementation(py::object &pysolver, const pddl::Task &task,
                   const std::string &inner_solver_name,
                   const py::dict &inner_solver_params, double dead_end_cost,
                   double rho, std::size_t mc_samples,
                   std::size_t max_iterations, std::size_t max_steps,
                   bool optimize_policy_graph, double discount, double epsilon,
                   const std::function<py::bool_(const py::object &)> &callback,
                   bool verbose)
        : _pysolver(std::make_unique<py::object>(pysolver)) {

      using SolverType = pddl::PPDDLPlanMergerSolver<Texecution, TstrategyTag>;

      InnerSolverParams params;
      for (auto &[k, v] : py::dict(inner_solver_params)) {
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
      params.set(std::string("dead_end_cost"), dead_end_cost);

      const auto &entry =
          pddl::find_pddl_inner_solver<Texecution>(inner_solver_name);

      typename SolverType::InnerSolverFactory factory =
          [&entry, params, verbose](pddl::PddlDeterministicDomain &det_d)
          -> std::unique_ptr<
              MetaInnerSolverBase<pddl::PddlDeterministicDomain>> {
        return entry.create(det_d, params, verbose);
      };

      typename SolverType::CallbackFunctor cb = [](const SolverType &) {
        return false;
      };

      if (callback) {
        _callback = callback;
        cb = [this](const SolverType &) -> bool {
          try {
            return _callback(*_pysolver);
          } catch (const std::exception &e) {
            Logger::error(std::string("SKDECIDE exception when calling "
                                      "PPDDLPlanMerger callback: ") +
                          e.what());
            throw;
          }
        };
      }

      _solver = std::make_unique<SolverType>(
          task, std::move(factory), rho, mc_samples, max_iterations, max_steps,
          dead_end_cost, optimize_policy_graph, discount, epsilon, cb, verbose);
    }

    virtual void solve(const pddl::State &s) override { _solver->solve(s); }
    virtual void resolve(const pddl::State &s) override { _solver->resolve(s); }
    virtual void clear() override { _solver->clear(); }

    virtual bool is_solution_defined_for(const pddl::State &s) override {
      return _solver->is_solution_defined_for(s);
    }

    virtual py::object get_next_action(const pddl::State &s) override {
      try {
        const auto &action = _solver->get_best_action(s);
        return py::cast(static_cast<const pddl::GroundAction &>(action));
      } catch (const std::runtime_error &) {
        return py::none();
      }
    }

    virtual double get_best_value(const pddl::State &s) override {
      return _solver->get_best_value(s);
    }

    virtual std::size_t get_nb_iterations() override {
      return _solver->get_nb_iterations();
    }

    virtual std::size_t get_nb_plans() override {
      return _solver->get_nb_plans();
    }

    virtual std::size_t get_solving_time() override {
      return _solver->get_solving_time();
    }

    virtual std::size_t get_policy_size() override {
      return _solver->get_policy_size();
    }

    virtual py::set get_explored_states() override {
      py::set s;
      for (auto &e : _solver->get_explored_states()) {
        s.add(py::cast(static_cast<const pddl::State &>(e)));
      }
      return s;
    }

    virtual py::set get_terminal_states() override {
      py::set s;
      for (auto &e : _solver->get_terminal_states()) {
        s.add(py::cast(static_cast<const pddl::State &>(e)));
      }
      return s;
    }

    virtual py::dict get_policy() override {
      py::dict d;
      for (auto &e : _solver->get_policy()) {
        d[py::cast(static_cast<const pddl::State &>(e.first))] = py::make_tuple(
            py::cast(static_cast<const pddl::GroundAction &>(e.second.first)),
            e.second.second);
      }
      return d;
    }

  private:
    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<pddl::PPDDLPlanMergerSolver<Texecution, TstrategyTag>>
        _solver;
    std::function<py::bool_(const py::object &)> _callback;
  };

  std::unique_ptr<BaseImplementation> _implementation;

  template <typename Texecution>
  void create_impl(const std::string &determinization,
                   const std::string &inner_solver_name,
                   const py::dict &inner_solver_params, py::object &solver,
                   const pddl::Task &task, double dead_end_cost, double rho,
                   std::size_t mc_samples, std::size_t max_iterations,
                   std::size_t max_steps, bool optimize_policy_graph,
                   double discount, double epsilon,
                   const std::function<py::bool_(const py::object &)> &callback,
                   bool verbose) {
    if (determinization == "all_outcomes") {
      _implementation =
          std::make_unique<Implementation<Texecution, AllOutcomesStrategy>>(
              solver, task, inner_solver_name, inner_solver_params,
              dead_end_cost, rho, mc_samples, max_iterations, max_steps,
              optimize_policy_graph, discount, epsilon, callback, verbose);
    } else if (determinization == "random_outcome") {
      _implementation =
          std::make_unique<Implementation<Texecution, RandomOutcomeStrategy>>(
              solver, task, inner_solver_name, inner_solver_params,
              dead_end_cost, rho, mc_samples, max_iterations, max_steps,
              optimize_policy_graph, discount, epsilon, callback, verbose);
    } else {
      _implementation = std::make_unique<
          Implementation<Texecution, MostProbableOutcomeStrategy>>(
          solver, task, inner_solver_name, inner_solver_params, dead_end_cost,
          rho, mc_samples, max_iterations, max_steps, optimize_policy_graph,
          discount, epsilon, callback, verbose);
    }
  }

public:
  PyPPDDLPlanMergerSolver(
      py::object &solver, const pddl::Task &task,
      const std::string &inner_solver_name = "FF",
      const std::string &determinization = "most_probable_outcome",
      bool parallel = false, double dead_end_cost = 1e9, double rho = 0.1,
      std::size_t mc_samples = 100, std::size_t max_iterations = 50,
      std::size_t max_steps = 10000, bool optimize_policy_graph = false,
      double discount = 0.99, double epsilon = 1e-3,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false, const py::dict &inner_solver_params = py::dict()) {
    if (parallel) {
      create_impl<ParallelExecution>(
          determinization, inner_solver_name, inner_solver_params, solver, task,
          dead_end_cost, rho, mc_samples, max_iterations, max_steps,
          optimize_policy_graph, discount, epsilon, callback, verbose);
    } else {
      create_impl<SequentialExecution>(
          determinization, inner_solver_name, inner_solver_params, solver, task,
          dead_end_cost, rho, mc_samples, max_iterations, max_steps,
          optimize_policy_graph, discount, epsilon, callback, verbose);
    }
  }

  void solve(const pddl::State &s) { _implementation->solve(s); }
  void resolve(const pddl::State &s) { _implementation->resolve(s); }
  void clear() { _implementation->clear(); }

  py::bool_ is_solution_defined_for(const pddl::State &s) {
    return _implementation->is_solution_defined_for(s);
  }

  py::object get_next_action(const pddl::State &s) {
    return _implementation->get_next_action(s);
  }

  py::float_ get_best_value(const pddl::State &s) {
    return _implementation->get_best_value(s);
  }

  py::int_ get_nb_iterations() { return _implementation->get_nb_iterations(); }
  py::int_ get_nb_plans() { return _implementation->get_nb_plans(); }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
  py::int_ get_policy_size() { return _implementation->get_policy_size(); }

  py::set get_explored_states() {
    return _implementation->get_explored_states();
  }
  py::set get_terminal_states() {
    return _implementation->get_terminal_states();
  }
  py::dict get_policy() { return _implementation->get_policy(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_PPDDLPLANMERGER_HH
