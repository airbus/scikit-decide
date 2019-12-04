/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "mcts.hh"
#include "core.hh"

#include "utils/python_gil_control.hh"
#include "utils/python_hash_eq.hh"
#include "utils/python_domain_adapter.hh"

namespace py = pybind11;


template <typename Texecution>
class PyMCTSDomain : public airlaps::PythonDomainAdapter<Texecution> {
public :

    PyMCTSDomain(const py::object& domain)
    : airlaps::PythonDomainAdapter<Texecution>(domain) {
        if (!py::hasattr(domain, "get_applicable_actions")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing get_applicable_actions()");
        }
        if (!py::hasattr(domain, "get_next_state_distribution")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing get_next_state_distribution()");
        }
        if (!py::hasattr(domain, "sample")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing sample()");
        }
        if (!py::hasattr(domain, "get_transition_value")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing get_transition_value()");
        }
        if (!py::hasattr(domain, "is_terminal")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing is_terminal()");
        }
    }

};


class PyMCTSSolver {
public :
    PyMCTSSolver(py::object& domain,
                 std::size_t time_budget = 3600000,
                 std::size_t rollout_budget = 100000,
                 std::size_t max_depth = 1000,
                 double discount = 1.0,
                 bool uct_mode = true,
                 double ucb_constant = 1.0 / std::sqrt(2.0),
                 bool parallel = true,
                 bool debug_logs = false) {
        if (parallel) {
            if (uct_mode) {
                _implementation = std::make_unique<Implementation<airlaps::ParallelExecution>>(
                    domain, time_budget, rollout_budget, max_depth, discount, debug_logs
                );
                // _implementation->ucb_constant(ucb_constant);
            } else {
                spdlog::error("MCTS only supports MCTS at the moment.");
                throw std::runtime_error("MCTS only supports MCTS at the moment.");
            }
        } else {
            if (uct_mode) {
                _implementation = std::make_unique<Implementation<airlaps::SequentialExecution>>(
                    domain, time_budget, rollout_budget, max_depth, discount, debug_logs
                );
                // _implementation->ucb_constant(ucb_constant);
            } else {
                spdlog::error("MCTS only supports MCTS at the moment.");
                throw std::runtime_error("MCTS only supports MCTS at the moment.");
            }
        }
    }

    void clear() {
        _implementation->clear();
    }

    void solve(const py::object& s) {
        _implementation->solve(s);
    }

    py::bool_ is_solution_defined_for(const py::object& s) {
        return _implementation->is_solution_defined_for(s);
    }

    py::object get_next_action(const py::object& s) {
        return _implementation->get_next_action(s);
    }

    py::float_ get_utility(const py::object& s) {
        return _implementation->get_utility(s);
    }

    virtual py::int_ get_nb_of_explored_states() {
        return _implementation->get_nb_of_explored_states();
    }

    virtual py::int_ get_nb_rollouts() {
        return _implementation->get_nb_rollouts();
    }

private :

    class BaseImplementation {
    public :
        virtual void clear() =0;
        virtual void solve(const py::object& s) =0;
        virtual py::bool_ is_solution_defined_for(const py::object& s) =0;
        virtual py::object get_next_action(const py::object& s) =0;
        virtual py::float_ get_utility(const py::object& s) =0;
        virtual void ucb_constant(double ucb_constant) =0;
        virtual py::int_ get_nb_of_explored_states() =0;
        virtual py::int_ get_nb_rollouts() =0;
    };

    template <typename Texecution>
    class Implementation : public BaseImplementation {
    public :
        Implementation(py::object& domain,
                       std::size_t time_budget = 3600000,
                       std::size_t rollout_budget = 100000,
                       std::size_t max_depth = 1000,
                       double discount = 1.0,
                       bool debug_logs = false) {

            _domain = std::make_unique<PyMCTSDomain<Texecution>>(domain);
            _solver = std::make_unique<airlaps::MCTSSolver<PyMCTSDomain<Texecution>, Texecution>>(
                        *_domain,
                        time_budget,
                        rollout_budget,
                        max_depth,
                        discount,
                        debug_logs);
            _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(std::cout,
                                                                            py::module::import("sys").attr("stdout"));
            _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(std::cerr,
                                                                            py::module::import("sys").attr("stderr"));
        }

        virtual void clear() {
            _solver->clear();
        }

        virtual void solve(const py::object& s) {
            typename airlaps::GilControl<Texecution>::Release release;
            _solver->solve(s);
        }

        virtual py::bool_ is_solution_defined_for(const py::object& s) {
            return _solver->is_solution_defined_for(s);
        }

        virtual py::object get_next_action(const py::object& s) {
            return _solver->get_best_action(s).get();
        }

        virtual py::float_ get_utility(const py::object& s) {
            return _solver->get_best_value(s);
        }

        virtual py::int_ get_nb_of_explored_states() {
            return _solver->nb_of_explored_states();
        }

        virtual py::int_ get_nb_rollouts() {
            return _solver->nb_rollouts();
        }

        virtual void ucb_constant(double ucb_constant) {
            // _solver->action_selector().ucb_constant() = ucb_constant;
        }

    private :
        std::unique_ptr<PyMCTSDomain<Texecution>> _domain;
        std::unique_ptr<airlaps::MCTSSolver<PyMCTSDomain<Texecution>, Texecution>> _solver;

        std::function<bool (const py::object&)> _goal_checker;
        std::function<double (const py::object&)> _heuristic;

        std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
        std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};


void init_pymcts(py::module& m) {
    py::class_<PyMCTSSolver> py_mcts_solver(m, "_MCTSSolver_");
        py_mcts_solver
            .def(py::init<py::object&,
                          std::size_t,
                          std::size_t,
                          std::size_t,
                          double,
                          bool,
                          double,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("time_budget")=3600000,
                 py::arg("rollout_budget")=100000,
                 py::arg("max_depth")=1000,
                 py::arg("discount")=1.0,
                 py::arg("uct_mode")=true,
                 py::arg("ucb_constant")=1.0/std::sqrt(2.0),
                 py::arg("parallel")=true,
                 py::arg("debug_logs")=false)
            .def("clear", &PyMCTSSolver::clear)
            .def("solve", &PyMCTSSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &PyMCTSSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &PyMCTSSolver::get_next_action, py::arg("state"))
            .def("get_utility", &PyMCTSSolver::get_utility, py::arg("state"))
            .def("get_nb_of_explored_states", &PyMCTSSolver::get_nb_of_explored_states)
            .def("get_nb_rollouts", &PyMCTSSolver::get_nb_rollouts)
        ;
}
