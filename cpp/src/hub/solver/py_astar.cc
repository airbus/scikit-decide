/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "astar.hh"
#include "core.hh"

#include "utils/python_gil_control.hh"
#include "utils/python_hash_eq.hh"
#include "utils/python_domain_adapter.hh"

namespace py = pybind11;


template <typename Texecution>
class PyAStarDomain : public airlaps::PythonDomainAdapter<Texecution> {
public :

    PyAStarDomain(const py::object& domain)
    : airlaps::PythonDomainAdapter<Texecution>(domain) {
        if (!py::hasattr(domain, "get_applicable_actions")) {
            throw std::invalid_argument("AIRLAPS exception: A* algorithm needs python domain for implementing get_applicable_actions()");
        }
        if (!py::hasattr(domain, "compute_next_state")) {
            throw std::invalid_argument("AIRLAPS exception: A* algorithm needs python domain for implementing compute_next_state()");
        }
        if (!py::hasattr(domain, "get_next_state")) {
            throw std::invalid_argument("AIRLAPS exception: A* algorithm needs python domain for implementing get_next_state()");
        }
        if (!py::hasattr(domain, "get_transition_value")) {
            throw std::invalid_argument("AIRLAPS exception: A* algorithm needs python domain for implementing get_transition_value()");
        }
    }

};


class PyAStarSolver {
public :
    PyAStarSolver(py::object& domain,
                  const std::function<bool (const py::object&)>& goal_checker,
                  const std::function<double (const py::object&)>& heuristic,
                  bool parallel = true,
                  bool debug_logs = false) {
        if (parallel) {
            _implementation = std::make_unique<Implementation<airlaps::ParallelExecution>>(
                domain, goal_checker, heuristic, debug_logs
            );
        } else {
            _implementation = std::make_unique<Implementation<airlaps::SequentialExecution>>(
                domain, goal_checker, heuristic, debug_logs
            );
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

private :

    class BaseImplementation {
    public :
        virtual void clear() =0;
        virtual void solve(const py::object& s) =0;
        virtual py::bool_ is_solution_defined_for(const py::object& s) =0;
        virtual py::object get_next_action(const py::object& s) =0;
        virtual py::float_ get_utility(const py::object& s) =0;
    };

    template <typename Texecution>
    class Implementation : public BaseImplementation {
    public :
        Implementation(py::object& domain,
                       const std::function<bool (const py::object&)>& goal_checker,
                       const std::function<double (const py::object&)>& heuristic,
                       bool debug_logs = false)
        : _goal_checker(goal_checker), _heuristic(heuristic) {
            _domain = std::make_unique<PyAStarDomain<Texecution>>(domain);
            _solver = std::make_unique<airlaps::AStarSolver<PyAStarDomain<Texecution>, Texecution>>(
                                                                            *_domain,
                                                                            [this](const typename PyAStarDomain<Texecution>::State& s)->bool {
                                                                                typename airlaps::GilControl<Texecution>::Acquire acquire;
                                                                                return _goal_checker(s._state);
                                                                            },
                                                                            [this](const typename PyAStarDomain<Texecution>::State& s)->double {
                                                                                typename airlaps::GilControl<Texecution>::Acquire acquire;
                                                                                return _heuristic(s._state);
                                                                            },
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

    private :
        std::unique_ptr<PyAStarDomain<Texecution>> _domain;
        std::unique_ptr<airlaps::AStarSolver<PyAStarDomain<Texecution>, Texecution>> _solver;

        std::function<bool (const py::object&)> _goal_checker;
        std::function<double (const py::object&)> _heuristic;

        std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
        std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};


void init_pyastar(py::module& m) {
    py::class_<PyAStarSolver> py_astar_solver(m, "_AStarSolver_");
        py_astar_solver
            .def(py::init<py::object&,
                          const std::function<bool (const py::object&)>&,
                          const std::function<double (const py::object&)>&,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("goal_checker"),
                 py::arg("heuristic"),
                 py::arg("parallel")=true,
                 py::arg("debug_logs")=false)
            .def("clear", &PyAStarSolver::clear)
            .def("solve", &PyAStarSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &PyAStarSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &PyAStarSolver::get_next_action, py::arg("state"))
            .def("get_utility", &PyAStarSolver::get_utility, py::arg("state"))
        ;
}
