/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "ilaostar.hh"
#include "core.hh"

#include "utils/python_gil_control.hh"
#include "utils/python_hash_eq.hh"
#include "utils/python_domain_adapter.hh"

namespace py = pybind11;


template <typename Texecution>
class PyILAOStarDomain : public skdecide::PythonDomainAdapter<Texecution> {
public :
    
    PyILAOStarDomain(const py::object& domain)
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


class PyILAOStarSolver {
public :
    PyILAOStarSolver(py::object& domain,
                   const std::function<py::object (const py::object&, const py::object&)>& goal_checker,
                   const std::function<py::object (const py::object&, const py::object&)>& heuristic,
                   double discount = 1.0,
                   double epsilon = 0.001,
                   bool parallel = false,
                   bool debug_logs = false) {
        if (parallel) {
            _implementation = std::make_unique<Implementation<skdecide::ParallelExecution>>(
                domain, goal_checker, heuristic, discount, epsilon, debug_logs
            );
        } else {
            _implementation = std::make_unique<Implementation<skdecide::SequentialExecution>>(
                domain, goal_checker, heuristic, discount, epsilon, debug_logs
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

    py::int_ get_nb_of_explored_states() {
        return _implementation->get_nb_of_explored_states();
    }

    py::int_ best_solution_graph_size() {
        return _implementation->best_solution_graph_size();
    }

    py::dict get_policy() {
        return _implementation->get_policy();
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
        virtual py::int_ get_nb_of_explored_states() =0;
        virtual py::int_ best_solution_graph_size() =0;
        virtual py::dict get_policy() =0;
    };

    template <typename Texecution>
    class Implementation : public BaseImplementation {
    public :
        Implementation(py::object& domain,
                       const std::function<py::object (const py::object&, const py::object&)>& goal_checker,
                       const std::function<py::object (const py::object&, const py::object&)>& heuristic,
                       double discount = 1.0,
                       double epsilon = 0.001,
                       bool debug_logs = false)
        : _goal_checker(goal_checker), _heuristic(heuristic) {
            _domain = std::make_unique<PyILAOStarDomain<Texecution>>(domain);
            _solver = std::make_unique<skdecide::ILAOStarSolver<PyILAOStarDomain<Texecution>, Texecution>>(
                                                                            *_domain,
                                                                            [this](PyILAOStarDomain<Texecution>& d, const typename PyILAOStarDomain<Texecution>::State& s)->bool {
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
                                                                            [this](PyILAOStarDomain<Texecution>& d, const typename PyILAOStarDomain<Texecution>::State& s)->double {
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
                                                                            epsilon,
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
            try {
                return _solver->get_best_action(s).pyobj();
            } catch (const std::runtime_error&) {
                return py::none();
            }
        }

        virtual py::float_ get_utility(const py::object& s) {
            try {
                return _solver->get_best_value(s);
            } catch (const std::runtime_error&) {
                return py::none();
            }
        }

        virtual py::int_ get_nb_of_explored_states() {
            return _solver->get_nb_of_explored_states();
        }

        virtual py::int_ best_solution_graph_size() {
            return _solver->best_solution_graph_size();
        }

        virtual py::dict get_policy() {
            py::dict d;
            auto&& p = _solver->policy();
            for (auto& e : p) {
                d[e.first.pyobj()] = py::make_tuple(e.second.first.pyobj(), e.second.second);
            }
            return d;
        }

    private :
        std::unique_ptr<PyILAOStarDomain<Texecution>> _domain;
        std::unique_ptr<skdecide::ILAOStarSolver<PyILAOStarDomain<Texecution>, Texecution>> _solver;

        std::function<py::object (const py::object&, const py::object&)> _goal_checker;
        std::function<py::object (const py::object&, const py::object&)> _heuristic;

        std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
        std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};


void init_pyilaostar(py::module& m) {
    py::class_<PyILAOStarSolver> py_ilaostar_solver(m, "_ILAOStarSolver_");
        py_ilaostar_solver
            .def(py::init<py::object&,
                          const std::function<py::object (const py::object&, const py::object&)>&,
                          const std::function<py::object (const py::object&, const py::object&)>&,
                          double,
                          double,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("goal_checker"),
                 py::arg("heuristic"),
                 py::arg("discount")=1.0,
                 py::arg("epsilon")=0.001,
                 py::arg("parallel")=false,
                 py::arg("debug_logs")=false)
            .def("clear", &PyILAOStarSolver::clear)
            .def("solve", &PyILAOStarSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &PyILAOStarSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &PyILAOStarSolver::get_next_action, py::arg("state"))
            .def("get_utility", &PyILAOStarSolver::get_utility, py::arg("state"))
            .def("get_nb_of_explored_states", &PyILAOStarSolver::get_nb_of_explored_states)
            .def("best_solution_graph_size", &PyILAOStarSolver::best_solution_graph_size)
            .def("get_policy", &PyILAOStarSolver::get_policy)
        ;
}
