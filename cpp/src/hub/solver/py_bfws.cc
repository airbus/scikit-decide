/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "bfws.hh"
#include "core.hh"

#include "utils/python_gil_control.hh"
#include "utils/python_hash_eq.hh"
#include "utils/python_domain_adapter.hh"

namespace py = pybind11;


template <typename Texecution>
class PyBFWSDomain : public airlaps::PythonDomainAdapter<Texecution> {
public :

    PyBFWSDomain(const py::object& domain)
    : airlaps::PythonDomainAdapter<Texecution>(domain) {
        if (!py::hasattr(domain, "get_applicable_actions")) {
            throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python domain for implementing get_applicable_actions()");
        }
        if (!py::hasattr(domain, "compute_next_state")) {
            throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python domain for implementing compute_sample()");
        }
        if (!py::hasattr(domain, "get_next_state")) {
            throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python domain for implementing get_sample()");
        }
        if (!py::hasattr(domain, "get_transition_value")) {
            throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python domain for implementing get_transition_value()");
        }
        if (!py::hasattr(domain, "is_terminal")) {
            throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python domain for implementing is_terminal()");
        }
    }

};


class PyBFWSSolver {
public :
    PyBFWSSolver(py::object& domain,
                 const std::function<void (const py::object&, const std::function<void (const py::int_&)>&)>& state_binarizer,
                 const std::function<double (const py::object&)>& heuristic,
                 const std::function<bool (const py::object&)>& termination_checker,
                 bool parallel = true,
                 bool debug_logs = false) {
        if (parallel) {
            _implementation = std::make_unique<Implementation<airlaps::ParallelExecution>>(
                domain, state_binarizer, heuristic, termination_checker, debug_logs
            );
        } else {
            _implementation = std::make_unique<Implementation<airlaps::SequentialExecution>>(
                domain, state_binarizer, heuristic, termination_checker, debug_logs
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
                       const std::function<void (const py::object&, const std::function<void (const py::int_&)>&)>& state_binarizer,
                       const std::function<double (const py::object&)>& heuristic,
                       const std::function<bool (const py::object&)>& termination_checker,
                       bool debug_logs = false)
        : _state_binarizer(state_binarizer), _heuristic(heuristic), _termination_checker(termination_checker) {
            _domain = std::make_unique<PyBFWSDomain<Texecution>>(domain);
            _solver = std::make_unique<airlaps::BFWSSolver<PyBFWSDomain<Texecution>, Texecution>>(
                                                                            *_domain,
                                                                            [this](const typename PyBFWSDomain<Texecution>::State& s,
                                                                                const std::function<void (const py::int_&)>& f)->void {
                                                                                typename airlaps::GilControl<Texecution>::Acquire acquire;
                                                                                _state_binarizer(s._state, f);
                                                                            },
                                                                            [this](const typename PyBFWSDomain<Texecution>::State& s)->double {
                                                                                typename airlaps::GilControl<Texecution>::Acquire acquire;
                                                                                return _heuristic(s._state);
                                                                            },
                                                                            [this](const typename PyBFWSDomain<Texecution>::State& s)->bool {
                                                                                typename airlaps::GilControl<Texecution>::Acquire acquire;
                                                                                return _termination_checker(s._state);
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
        std::unique_ptr<PyBFWSDomain<Texecution>> _domain;
        std::unique_ptr<airlaps::BFWSSolver<PyBFWSDomain<Texecution>, Texecution>> _solver;
        
        std::function<void (const py::object&, const std::function<void (const py::int_&)>&)> _state_binarizer;
        std::function<double (const py::object&)> _heuristic;
        std::function<bool (const py::object&)> _termination_checker;

        std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
        std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};


void init_pybfws(py::module& m) {
    py::class_<PyBFWSSolver> py_bfws_solver(m, "_BFWSSolver_");
        py_bfws_solver
            .def(py::init<py::object&,
                          const std::function<void (const py::object&, const std::function<void (const py::int_&)>&)>&,
                          const std::function<double (const py::object&)>&,
                          const std::function<bool (const py::object&)>&,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("state_binarizer"),
                 py::arg("heuristic"),
                 py::arg("termination_checker"),
                 py::arg("parallel")=true,
                 py::arg("debug_logs")=false)
            .def("clear", &PyBFWSSolver::clear)
            .def("solve", &PyBFWSSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &PyBFWSSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &PyBFWSSolver::get_next_action, py::arg("state"))
            .def("get_utility", &PyBFWSSolver::get_utility, py::arg("state"))
        ;
}
