/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_PPDDLDETHINDSIGHT_HH
#define SKDECIDE_PY_PPDDLDETHINDSIGHT_HH

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ffdethindsight.hh"
#include "hub/solver/pddl/inner_solver/pddl_inner_solver_registry.hh"
#include "hub/solver/pddl/inner_solver/impl/pddl_inner_solver_registry_impl.hh"
#include "hub/solver/pddl/ppddlreplan/ff_inner_solver.hh"
#include "impl/ffdethindsight_impl.hh"
#include "ppddldethindsight.hh"
#include "impl/ppddldethindsight_impl.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace py = pybind11;

namespace skdecide {

// --- PPDDLDetHindsight pybind wrapper ---

class PyPPDDLDetHindsightSolver {
private:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void solve(const pddl::State &s) = 0;
    virtual void clear() = 0;
    virtual bool is_solution_defined_for(const pddl::State &s) = 0;
    virtual py::object get_next_action(const pddl::State &s) = 0;
    virtual double get_best_value(const pddl::State &s) = 0;
    virtual std::size_t get_nb_steps() = 0;
    virtual std::size_t get_solving_time() = 0;
    virtual py::set get_explored_states() = 0;
    virtual py::set get_terminal_states() = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    Implementation(py::object &pysolver, const pddl::Task &task,
                   const std::string &inner_solver_name,
                   const py::dict &inner_solver_params,
                   std::size_t sample_width, double dead_end_cost,
                   std::size_t max_steps, double discount, double epsilon,
                   const std::function<py::bool_(const py::object &)> &callback,
                   bool verbose)
        : _pysolver(std::make_unique<py::object>(pysolver)) {

      using SolverType = pddl::PPDDLDetHindsightSolver<Texecution>;

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
          pddl::find_pddl_inner_solver<SequentialExecution>(inner_solver_name);

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
                                      "PPDDLDetHindsight callback: ") +
                          e.what());
            throw;
          }
        };
      }

      _solver = std::make_unique<SolverType>(
          task, std::move(factory), sample_width, dead_end_cost, max_steps,
          discount, epsilon, cb, verbose);
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

    virtual double get_best_value(const pddl::State &s) override {
      return _solver->get_best_value(s);
    }

    virtual std::size_t get_nb_steps() override {
      return _solver->get_nb_steps();
    }

    virtual std::size_t get_solving_time() override {
      return _solver->get_solving_time();
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

  private:
    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<pddl::PPDDLDetHindsightSolver<Texecution>> _solver;
    std::function<py::bool_(const py::object &)> _callback;
  };

  std::unique_ptr<BaseImplementation> _implementation;

public:
  PyPPDDLDetHindsightSolver(
      py::object &solver, const pddl::Task &task,
      const std::string &inner_solver_name = "FF", bool parallel = false,
      std::size_t sample_width = 30, double dead_end_cost = 1e9,
      std::size_t max_steps = 10000, double discount = 0.99,
      double epsilon = 1e-3,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false, const py::dict &inner_solver_params = py::dict()) {
    if (parallel) {
      _implementation = std::make_unique<Implementation<ParallelExecution>>(
          solver, task, inner_solver_name, inner_solver_params, sample_width,
          dead_end_cost, max_steps, discount, epsilon, callback, verbose);
    } else {
      _implementation = std::make_unique<Implementation<SequentialExecution>>(
          solver, task, inner_solver_name, inner_solver_params, sample_width,
          dead_end_cost, max_steps, discount, epsilon, callback, verbose);
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

  py::float_ get_best_value(const pddl::State &s) {
    return _implementation->get_best_value(s);
  }
  py::int_ get_nb_steps() { return _implementation->get_nb_steps(); }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }

  py::set get_explored_states() {
    return _implementation->get_explored_states();
  }
  py::set get_terminal_states() {
    return _implementation->get_terminal_states();
  }
};

// --- FFDetHindsight pybind wrapper ---

class PyFFDetHindsightSolver {
private:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void solve(const pddl::State &s) = 0;
    virtual void clear() = 0;
    virtual bool is_solution_defined_for(const pddl::State &s) = 0;
    virtual py::object get_next_action(const pddl::State &s) = 0;
    virtual double get_best_value(const pddl::State &s) = 0;
    virtual std::size_t get_nb_steps() = 0;
    virtual std::size_t get_solving_time() = 0;
    virtual py::set get_explored_states() = 0;
    virtual py::set get_terminal_states() = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    Implementation(py::object &pysolver, const pddl::Task &task,
                   std::size_t sample_width, double dead_end_cost,
                   std::size_t max_steps, double discount, double epsilon,
                   const std::function<py::bool_(const py::object &)> &callback,
                   bool verbose)
        : _pysolver(std::make_unique<py::object>(pysolver)) {

      typename pddl::FFDetHindsightSolver<Texecution>::CallbackFunctor cb =
          [](const pddl::FFDetHindsightSolver<Texecution> &) { return false; };

      if (callback) {
        _callback = callback;
        cb = [this](const pddl::FFDetHindsightSolver<Texecution> &) -> bool {
          try {
            return _callback(*_pysolver);
          } catch (const std::exception &e) {
            Logger::error(std::string("SKDECIDE exception when calling "
                                      "FFDetHindsight callback: ") +
                          e.what());
            throw;
          }
        };
      }

      _solver = std::make_unique<pddl::FFDetHindsightSolver<Texecution>>(
          task, sample_width, dead_end_cost, max_steps, discount, epsilon, cb,
          verbose);
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

    virtual double get_best_value(const pddl::State &s) override {
      return _solver->get_best_value(s);
    }

    virtual std::size_t get_nb_steps() override {
      return _solver->get_nb_steps();
    }

    virtual std::size_t get_solving_time() override {
      return _solver->get_solving_time();
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

  private:
    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<pddl::FFDetHindsightSolver<Texecution>> _solver;
    std::function<py::bool_(const py::object &)> _callback;
  };

  std::unique_ptr<BaseImplementation> _implementation;

public:
  PyFFDetHindsightSolver(
      py::object &solver, const pddl::Task &task, bool parallel = false,
      std::size_t sample_width = 30, double dead_end_cost = 1e9,
      std::size_t max_steps = 10000, double discount = 0.99,
      double epsilon = 1e-3,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    if (parallel) {
      _implementation = std::make_unique<Implementation<ParallelExecution>>(
          solver, task, sample_width, dead_end_cost, max_steps, discount,
          epsilon, callback, verbose);
    } else {
      _implementation = std::make_unique<Implementation<SequentialExecution>>(
          solver, task, sample_width, dead_end_cost, max_steps, discount,
          epsilon, callback, verbose);
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

  py::float_ get_best_value(const pddl::State &s) {
    return _implementation->get_best_value(s);
  }
  py::int_ get_nb_steps() { return _implementation->get_nb_steps(); }
  py::int_ get_solving_time() { return _implementation->get_solving_time(); }

  py::set get_explored_states() {
    return _implementation->get_explored_states();
  }
  py::set get_terminal_states() {
    return _implementation->get_terminal_states();
  }
};

} // namespace skdecide

#endif // SKDECIDE_PY_PPDDLDETHINDSIGHT_HH
