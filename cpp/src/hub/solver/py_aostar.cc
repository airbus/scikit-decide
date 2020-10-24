/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "aostar.hh"
#include "core.hh"

#include "utils/python_gil_control.hh"
#include "utils/python_hash_eq.hh"
#include "utils/python_domain_adapter.hh"

namespace py = pybind11;


template <typename Texecution>
class PyAOStarDomain : public skdecide::PythonDomainAdapter<Texecution> {
public :
    
    PyAOStarDomain(const py::object& domain)
    : skdecide::PythonDomainAdapter<Texecution>(domain) {
        if (!py::hasattr(domain, "get_applicable_actions")) {
            throw std::invalid_argument("SKDECIDE exception: AO* algorithm needs python domain for implementing get_applicable_actions()");
        }
        if (!py::hasattr(domain, "get_next_state_distribution")) {
            throw std::invalid_argument("SKDECIDE exception: AO* algorithm needs python domain for implementing get_next_state_distribution()");
        }
        if (!py::hasattr(domain, "get_transition_value")) {
            throw std::invalid_argument("SKDECIDE exception: AO* algorithm needs python domain for implementing get_transition_value()");
        }
    }

};


class PyAOStarSolver {
public :
    PyAOStarSolver(py::object& domain,
                   const std::function<py::object (const py::object&, const py::object&)>& goal_checker,
                   const std::function<py::object (const py::object&, const py::object&)>& heuristic,
                   double discount = 1.0,
                   std::size_t max_tip_expansions = 1,
                   bool detect_cycles = false,
                   bool parallel = false,
                   bool debug_logs = false) {
        if (parallel) {
            _implementation = std::make_unique<Implementation<skdecide::ParallelExecution>>(
                domain, goal_checker, heuristic, discount, max_tip_expansions, detect_cycles, debug_logs
            );
        } else {
            _implementation = std::make_unique<Implementation<skdecide::SequentialExecution>>(
                domain, goal_checker, heuristic, discount, max_tip_expansions, detect_cycles, debug_logs
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
        virtual ~BaseImplementation() {}
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
                       const std::function<py::object (const py::object&, const py::object&)>& goal_checker,
                       const std::function<py::object (const py::object&, const py::object&)>& heuristic,
                       double discount = 1.0,
                       std::size_t max_tip_expansions = 1,
                       bool detect_cycles = false,
                       bool debug_logs = false)
        : _goal_checker(goal_checker), _heuristic(heuristic) {
            _domain = std::make_unique<PyAOStarDomain<Texecution>>(domain);
            _solver = std::make_unique<skdecide::AOStarSolver<PyAOStarDomain<Texecution>, Texecution>>(
                                                                            *_domain,
                                                                            [this](PyAOStarDomain<Texecution>& d, const typename PyAOStarDomain<Texecution>::State& s)->bool {
                                                                                try {
                                                                                    auto fgc = [this](const py::object& dd, const py::object& ss, [[maybe_unused]] const py::object& ii) {
                                                                                        return _goal_checker(dd, ss);
                                                                                    };
                                                                                    std::unique_ptr<py::object> r = d.call(nullptr, fgc, s.pyobj());
                                                                                    typename skdecide::GilControl<Texecution>::Acquire acquire;
                                                                                    bool rr = r->template cast<bool>();
                                                                                    r.reset();
                                                                                    return  rr;
                                                                                } catch (const std::exception& e) {
                                                                                    spdlog::error(std::string("SKDECIDE exception when calling goal checker: ") + e.what());
                                                                                    throw;
                                                                                }
                                                                            },
                                                                            [this](PyAOStarDomain<Texecution>& d, const typename PyAOStarDomain<Texecution>::State& s)->double {
                                                                                try {
                                                                                    auto fh = [this](const py::object& dd, const py::object& ss, [[maybe_unused]] const py::object& ii) {
                                                                                        return _heuristic(dd, ss);
                                                                                    };
                                                                                    std::unique_ptr<py::object> r = d.call(nullptr, fh, s.pyobj());
                                                                                    typename skdecide::GilControl<Texecution>::Acquire acquire;
                                                                                    double rr = r->template cast<double>();
                                                                                    r.reset();
                                                                                    return  rr;
                                                                                } catch (const std::exception& e) {
                                                                                    spdlog::error(std::string("SKDECIDE exception when calling heuristic estimator: ") + e.what());
                                                                                    throw;
                                                                                }
                                                                            },
                                                                            discount,
                                                                            max_tip_expansions,
                                                                            detect_cycles,
                                                                            debug_logs);
            _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(std::cout,
                                                                            py::module::import("sys").attr("stdout"));
            _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(std::cerr,
                                                                            py::module::import("sys").attr("stderr"));
        }

        virtual ~Implementation() {}

        virtual void clear() {
            _solver->clear();
        }

        virtual void solve(const py::object& s) {
            typename skdecide::GilControl<Texecution>::Release release;
            _solver->solve(s);
        }

        virtual py::bool_ is_solution_defined_for(const py::object& s) {
            return _solver->is_solution_defined_for(s);
        }

        virtual py::object get_next_action(const py::object& s) {
            return _solver->get_best_action(s).pyobj();
        }

        virtual py::float_ get_utility(const py::object& s) {
            return _solver->get_best_value(s);
        }

    private :
        std::unique_ptr<PyAOStarDomain<Texecution>> _domain;
        std::unique_ptr<skdecide::AOStarSolver<PyAOStarDomain<Texecution>, Texecution>> _solver;

        std::function<py::object (const py::object&, const py::object&)> _goal_checker;
        std::function<py::object (const py::object&, const py::object&)> _heuristic;

        std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
        std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};


void init_pyaostar(py::module& m) {
    py::class_<PyAOStarSolver> py_aostar_solver(m, "_AOStarSolver_");
        py_aostar_solver
            .def(py::init<py::object&,
                          const std::function<py::object (const py::object&, const py::object&)>&,
                          const std::function<py::object (const py::object&, const py::object&)>&,
                          double,
                          std::size_t,
                          bool,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("goal_checker"),
                 py::arg("heuristic"),
                 py::arg("discount")=1.0,
                 py::arg("max_tip_expansions")=1,
                 py::arg("detect_cycles")=false,
                 py::arg("parallel")=false,
                 py::arg("debug_logs")=false)
            .def("clear", &PyAOStarSolver::clear)
            .def("solve", &PyAOStarSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &PyAOStarSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &PyAOStarSolver::get_next_action, py::arg("state"))
            .def("get_utility", &PyAOStarSolver::get_utility, py::arg("state"))
        ;
}
