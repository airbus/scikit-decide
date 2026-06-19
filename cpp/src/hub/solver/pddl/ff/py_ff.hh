/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_PDDL_FF_HH
#define SKDECIDE_PY_PDDL_FF_HH

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "ff.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace py = pybind11;

namespace skdecide {

class PyFFSolver {
private:
  class BaseImplementation {
  public:
    virtual ~BaseImplementation() {}
    virtual void solve(const pddl::State &s) = 0;
    virtual void clear() = 0;
    virtual bool is_solution_defined_for(const pddl::State &s) = 0;
    virtual py::object get_next_action(const pddl::State &s) = 0;
    virtual py::list get_plan() = 0;
    virtual std::size_t get_nb_explored_states() = 0;
    virtual py::list get_explored_states() = 0;
    virtual std::size_t get_solving_time() = 0;
  };

  template <typename Texecution>
  class Implementation : public BaseImplementation {
  public:
    Implementation(py::object &pysolver, const pddl::Task &task,
                   double dead_end_cost,
                   const std::function<py::bool_(const py::object &)> &callback,
                   bool verbose)
        : _pysolver(std::make_unique<py::object>(pysolver)) {

      typename pddl::FFSolver<Texecution>::CallbackFunctor cb =
          [](const pddl::FFSolver<Texecution> &) { return false; };

      if (callback) {
        _callback = callback;
        cb = [this](const pddl::FFSolver<Texecution> &) -> bool {
          try {
            return _callback(*_pysolver);
          } catch (const std::exception &e) {
            Logger::error(std::string("SKDECIDE exception when calling "
                                      "FF callback: ") +
                          e.what());
            throw;
          }
        };
      }

      _solver = std::make_unique<pddl::FFSolver<Texecution>>(
          task, dead_end_cost, cb, verbose);
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
      auto plan = _solver->get_plan();
      for (auto &[state, action] : plan) {
        result.append(py::make_tuple(py::cast(state), py::cast(action)));
      }
      return result;
    }

    virtual std::size_t get_nb_explored_states() override {
      return _solver->get_nb_explored_states();
    }

    virtual py::list get_explored_states() override {
      py::list result;
      auto states = _solver->get_explored_states();
      for (auto &s : states) {
        result.append(py::cast(s));
      }
      return result;
    }

    virtual std::size_t get_solving_time() override {
      return _solver->get_solving_time();
    }

  private:
    std::unique_ptr<py::object> _pysolver;
    std::unique_ptr<pddl::FFSolver<Texecution>> _solver;
    std::function<py::bool_(const py::object &)> _callback;
  };

  std::unique_ptr<BaseImplementation> _implementation;

public:
  PyFFSolver(
      py::object &solver, const pddl::Task &task, bool parallel = false,
      double dead_end_cost = 1e9,
      const std::function<py::bool_(const py::object &)> &callback = nullptr,
      bool verbose = false) {
    if (parallel) {
      _implementation = std::make_unique<Implementation<ParallelExecution>>(
          solver, task, dead_end_cost, callback, verbose);
    } else {
      _implementation = std::make_unique<Implementation<SequentialExecution>>(
          solver, task, dead_end_cost, callback, verbose);
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

  py::int_ get_nb_explored_states() {
    return _implementation->get_nb_explored_states();
  }

  py::list get_explored_states() {
    return _implementation->get_explored_states();
  }

  py::int_ get_solving_time() { return _implementation->get_solving_time(); }
};

} // namespace skdecide

#endif // SKDECIDE_PY_PDDL_FF_HH
