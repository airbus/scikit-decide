/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_PPDDLREPLAN_HH
#define SKDECIDE_PY_PPDDLREPLAN_HH

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ff_inner_solver.hh"
#include "ffreplan.hh"
#include "ppddlreplan.hh"
#include "hub/solver/pddl/inner_solver/pddl_inner_solver_registry.hh"
#include "hub/solver/pddl/inner_solver/impl/pddl_inner_solver_registry_impl.hh"
#include "hub/solver/determinization/determinized_domain.hh"
#include "impl/ffreplan_impl.hh"
#include "impl/ppddlreplan_impl.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace py = pybind11;

namespace skdecide {

class PyFFReplanSolver {
private:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void solve(const pddl::State &s) = 0;
    virtual void clear() = 0;
    virtual bool is_solution_defined_for(const pddl::State &s) = 0;
    virtual py::object get_next_action(const pddl::State &s) = 0;
    virtual py::list get_plan() = 0;
    virtual std::size_t get_nb_replans() = 0;
    virtual std::size_t get_nb_steps() = 0;
    virtual std::size_t get_solving_time() = 0;
    virtual double get_total_cost() = 0;
  };

  template <typename Texecution, typename TstrategyTag>
  class Implementation : public BaseImplementation {
  public:
    Implementation(py::object &pysolver, const pddl::Task &task,
                   double dead_end_cost, std::size_t max_replans,
                   std::size_t max_steps,
                   const std::function<py::bool_(const py::object &)> &callback,
                   bool verbose)
        : _pysolver(std::make_unique<py::object>(pysolver)) {

      typename pddl::FFReplanSolver<Texecution, TstrategyTag>::CallbackFunctor
          cb = [](const pddl::FFReplanSolver<Texecution, TstrategyTag> &) {
            return false;
          };

      if (callback) {
        _callback = callback;
        cb = [this](const pddl::FFReplanSolver<Texecution, TstrategyTag> &)
            -> bool {
          try {
            return _callback(*_pysolver);
          } catch (const std::exception &e) {
            Logger::error(
                std::string(
                    "SKDECIDE exception when calling FFReplan callback: ") +
                e.what());
            throw;
          }
        };
      }

      _solver =
          std::make_unique<pddl::FFReplanSolver<Texecution, TstrategyTag>>(
              task, dead_end_cost, max_replans, max_steps, cb, verbose);
    }

    virtual void solve(const pddl::State &s) override { _solver->solve(s); }

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

    virtual py::list get_plan() override {
      py::list result;
      for (const auto &[state, action] : _solver->get_plan()) {
        result.append(py::make_tuple(
            py::cast(static_cast<const pddl::State &>(state)),
            py::cast(static_cast<const pddl::GroundAction &>(action))));
      }
      return result;
    }

    virtual std::size_t get_nb_replans() override {
      return _solver->get_nb_replans();
    }

    virtual std::size_t get_nb_steps() override {
      return _solver->get_nb_steps();
    }

    virtual std::size_t get_solving_time() override {
      return _solver->get_solving_time();
    }

    virtual double get_total_cost() override {
      return _solver->get_total_cost();
    }

  private:
    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<pddl::FFReplanSolver<Texecution, TstrategyTag>> _solver;
    std::function<py::bool_(const py::object &)> _callback;
  };

  std::unique_ptr<BaseImplementation> _implementation;

  template <typename Texecution>
  void create_impl(const std::string &determinization, py::object &solver,
                   const pddl::Task &task, double dead_end_cost,
                   std::size_t max_replans, std::size_t max_steps,
                   const std::function<py::bool_(const py::object &)> &callback,
                   bool verbose) {
    if (determinization == "all_outcomes") {
      _implementation =
          std::make_unique<Implementation<Texecution, AllOutcomesStrategy>>(
              solver, task, dead_end_cost, max_replans, max_steps, callback,
              verbose);
    } else if (determinization == "random_outcome") {
      _implementation =
          std::make_unique<Implementation<Texecution, RandomOutcomeStrategy>>(
              solver, task, dead_end_cost, max_replans, max_steps, callback,
              verbose);
    } else {
      _implementation = std::make_unique<
          Implementation<Texecution, MostProbableOutcomeStrategy>>(
          solver, task, dead_end_cost, max_replans, max_steps, callback,
          verbose);
    }
  }

public:
  PyFFReplanSolver(
      py::object &solver, const pddl::Task &task,
      const std::string &determinization = "most_probable_outcome",
      bool parallel = false, double dead_end_cost = 1e9,
      std::size_t max_replans = 1000, std::size_t max_steps = 10000,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    if (parallel) {
      create_impl<ParallelExecution>(determinization, solver, task,
                                     dead_end_cost, max_replans, max_steps,
                                     callback, verbose);
    } else {
      create_impl<SequentialExecution>(determinization, solver, task,
                                       dead_end_cost, max_replans, max_steps,
                                       callback, verbose);
    }
  }

  void solve(const pddl::State &s) { _implementation->solve(s); }
  void clear() { _implementation->clear(); }

  py::bool_ is_solution_defined_for(const pddl::State &s) {
    return _implementation->is_solution_defined_for(s);
  }

  py::object get_next_action(const pddl::State &s) {
    return _implementation->get_next_action(s);
  }

  py::list get_plan() { return _implementation->get_plan(); }
  py::int_ get_nb_replans() { return _implementation->get_nb_replans(); }
  py::int_ get_nb_steps() { return _implementation->get_nb_steps(); }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
  py::float_ get_total_cost() { return _implementation->get_total_cost(); }
};

class PyPPDDLReplanSolver {
private:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void solve(const pddl::State &s) = 0;
    virtual void clear() = 0;
    virtual bool is_solution_defined_for(const pddl::State &s) = 0;
    virtual py::object get_next_action(const pddl::State &s) = 0;
    virtual py::list get_plan() = 0;
    virtual std::size_t get_nb_replans() = 0;
    virtual std::size_t get_nb_steps() = 0;
    virtual std::size_t get_solving_time() = 0;
    virtual double get_total_cost() = 0;
  };

  template <typename Texecution, typename TstrategyTag>
  class Implementation : public BaseImplementation {
  public:
    Implementation(py::object &pysolver, const pddl::Task &task,
                   const std::string &inner_solver_name,
                   const py::dict &inner_solver_params, double dead_end_cost,
                   std::size_t max_replans, std::size_t max_steps,
                   const std::function<py::bool_(const py::object &)> &callback,
                   bool verbose)
        : _pysolver(std::make_unique<py::object>(pysolver)) {

      using SolverType = pddl::PPDDLReplanSolver<Texecution, TstrategyTag>;

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
            Logger::error(
                std::string(
                    "SKDECIDE exception when calling PPDDLReplan callback: ") +
                e.what());
            throw;
          }
        };
      }

      _solver = std::make_unique<SolverType>(
          task, std::move(factory), max_replans, max_steps, cb, verbose);
    }

    virtual void solve(const pddl::State &s) override { _solver->solve(s); }

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

    virtual py::list get_plan() override {
      py::list result;
      for (const auto &[state, action] : _solver->get_plan()) {
        result.append(py::make_tuple(
            py::cast(static_cast<const pddl::State &>(state)),
            py::cast(static_cast<const pddl::GroundAction &>(action))));
      }
      return result;
    }

    virtual std::size_t get_nb_replans() override {
      return _solver->get_nb_replans();
    }

    virtual std::size_t get_nb_steps() override {
      return _solver->get_nb_steps();
    }

    virtual std::size_t get_solving_time() override {
      return _solver->get_solving_time();
    }

    virtual double get_total_cost() override {
      return _solver->get_total_cost();
    }

  private:
    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<pddl::PPDDLReplanSolver<Texecution, TstrategyTag>> _solver;
    std::function<py::bool_(const py::object &)> _callback;
  };

  std::unique_ptr<BaseImplementation> _implementation;

  template <typename Texecution>
  void create_impl(const std::string &determinization,
                   const std::string &inner_solver_name,
                   const py::dict &inner_solver_params, py::object &solver,
                   const pddl::Task &task, double dead_end_cost,
                   std::size_t max_replans, std::size_t max_steps,
                   const std::function<py::bool_(const py::object &)> &callback,
                   bool verbose) {
    if (determinization == "all_outcomes") {
      _implementation =
          std::make_unique<Implementation<Texecution, AllOutcomesStrategy>>(
              solver, task, inner_solver_name, inner_solver_params,
              dead_end_cost, max_replans, max_steps, callback, verbose);
    } else if (determinization == "random_outcome") {
      _implementation =
          std::make_unique<Implementation<Texecution, RandomOutcomeStrategy>>(
              solver, task, inner_solver_name, inner_solver_params,
              dead_end_cost, max_replans, max_steps, callback, verbose);
    } else {
      _implementation = std::make_unique<
          Implementation<Texecution, MostProbableOutcomeStrategy>>(
          solver, task, inner_solver_name, inner_solver_params, dead_end_cost,
          max_replans, max_steps, callback, verbose);
    }
  }

public:
  PyPPDDLReplanSolver(
      py::object &solver, const pddl::Task &task,
      const std::string &inner_solver_name = "FF",
      const std::string &determinization = "most_probable_outcome",
      bool parallel = false, double dead_end_cost = 1e9,
      std::size_t max_replans = 1000, std::size_t max_steps = 10000,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false, const py::dict &inner_solver_params = py::dict()) {
    if (parallel) {
      create_impl<ParallelExecution>(
          determinization, inner_solver_name, inner_solver_params, solver, task,
          dead_end_cost, max_replans, max_steps, callback, verbose);
    } else {
      create_impl<SequentialExecution>(
          determinization, inner_solver_name, inner_solver_params, solver, task,
          dead_end_cost, max_replans, max_steps, callback, verbose);
    }
  }

  void solve(const pddl::State &s) { _implementation->solve(s); }
  void clear() { _implementation->clear(); }

  py::bool_ is_solution_defined_for(const pddl::State &s) {
    return _implementation->is_solution_defined_for(s);
  }

  py::object get_next_action(const pddl::State &s) {
    return _implementation->get_next_action(s);
  }

  py::list get_plan() { return _implementation->get_plan(); }
  py::int_ get_nb_replans() { return _implementation->get_nb_replans(); }
  py::int_ get_nb_steps() { return _implementation->get_nb_steps(); }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
  py::float_ get_total_cost() { return _implementation->get_total_cost(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_PPDDLREPLAN_HH
