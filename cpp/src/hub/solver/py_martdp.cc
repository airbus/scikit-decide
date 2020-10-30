/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "martdp.hh"

#include "utils/python_domain_proxy.hh"

namespace py = pybind11;


template <typename Texecution>
class PyMARTDPDomain : public skdecide::PythonDomainProxy<Texecution> {
public :
    
    PyMARTDPDomain(const py::object& domain)
    : skdecide::PythonDomainProxy<Texecution>(domain) {
        if (!py::hasattr(domain, "get_applicable_actions")) {
            throw std::invalid_argument("SKDECIDE exception: MARTDP algorithm needs python domain for implementing get_applicable_actions()");
        }
        if (!py::hasattr(domain, "get_next_state_distribution")) {
            throw std::invalid_argument("SKDECIDE exception: MARTDP algorithm needs python domain for implementing get_next_state_distribution()");
        }
        if (!py::hasattr(domain, "get_transition_value")) {
            throw std::invalid_argument("SKDECIDE exception: MARTDP algorithm needs python domain for implementing get_transition_value()");
        }
    }

};


class PyMARTDPSolver {
public :

    PyMARTDPSolver(py::object& domain,
                  const std::function<py::object (py::object&, const py::object&, const py::object&)>& goal_checker,  // last arg used for optional thread_id
                  const std::function<py::object (py::object&, const py::object&, const py::object&)>& heuristic,  // last arg used for optional thread_id
                  std::size_t time_budget = 3600000,
                  std::size_t rollout_budget = 100000,
                  std::size_t max_depth = 1000,
                  double discount = 1.0,
                  double epsilon = 0.001,
                  bool online_node_garbage = false,
                  bool parallel = false,
                  bool debug_logs = false) {

        if (parallel) {
            _implementation = std::make_unique<Implementation<skdecide::ParallelExecution>>(
                domain, goal_checker, heuristic,
                time_budget, rollout_budget, max_depth,
                discount, epsilon, online_node_garbage, debug_logs
            );
        } else {
            _implementation = std::make_unique<Implementation<skdecide::SequentialExecution>>(
                domain, goal_checker, heuristic,
                time_budget, rollout_budget, max_depth,
                discount, epsilon, online_node_garbage, debug_logs
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

    py::int_ get_nb_rollouts() {
        return _implementation->get_nb_rollouts();
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
        virtual py::int_ get_nb_rollouts() =0;
        virtual py::dict get_policy() =0;
    };

    template <typename Texecution>
    class Implementation : public BaseImplementation {
    public :

        Implementation(py::object& domain,
                       const std::function<py::object (py::object&, const py::object&, const py::object&)>& goal_checker,  // last arg used for optional thread_id
                       const std::function<py::object (py::object&, const py::object&, const py::object&)>& heuristic,  // last arg used for optional thread_id
                       std::size_t time_budget = 3600000,
                       std::size_t rollout_budget = 100000,
                       std::size_t max_depth = 1000,
                       double discount = 1.0,
                       double epsilon = 0.001,
                       bool online_node_garbage = false,
                       bool debug_logs = false)
        : _goal_checker(goal_checker), _heuristic(heuristic) {
            
            _domain = std::make_unique<PyMARTDPDomain<Texecution>>(domain);
            _solver = std::make_unique<skdecide::MARTDPSolver<PyMARTDPDomain<Texecution>, Texecution>>(
                                                                            *_domain,
                                                                            [this](PyMARTDPDomain<Texecution>& d, const typename PyMARTDPDomain<Texecution>::State& s, const std::size_t* thread_id)->bool {
                                                                                try {
                                                                                    std::unique_ptr<py::object> r = d.call(thread_id, _goal_checker, s.pyobj());
                                                                                    typename skdecide::GilControl<Texecution>::Acquire acquire;
                                                                                    bool rr = r->template cast<bool>();
                                                                                    r.reset();
                                                                                    return  rr;
                                                                                } catch (const std::exception& e) {
                                                                                    spdlog::error(std::string("SKDECIDE exception when calling goal checker: ") + e.what());
                                                                                    throw;
                                                                                }
                                                                            },
                                                                            [this](PyMARTDPDomain<Texecution>& d, const typename PyMARTDPDomain<Texecution>::State& s, const std::size_t* thread_id) -> PyMARTDPDomain<Texecution>::Value {
                                                                                try {
                                                                                    return PyMARTDPDomain<Texecution>::Value(d.call(thread_id, _heuristic, s.pyobj()));
                                                                                } catch (const std::exception& e) {
                                                                                    spdlog::error(std::string("SKDECIDE exception when calling heuristic: ") + e.what());
                                                                                    throw;
                                                                                }
                                                                            },
                                                                            time_budget,
                                                                            rollout_budget,
                                                                            max_depth,
                                                                            discount,
                                                                            epsilon,
                                                                            online_node_garbage,
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

        virtual py::int_ get_nb_rollouts() {
            return _solver->get_nb_rollouts();
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
        std::unique_ptr<PyMARTDPDomain<Texecution>> _domain;
        std::unique_ptr<skdecide::MARTDPSolver<PyMARTDPDomain<Texecution>, Texecution>> _solver;
        
        std::function<py::object (py::object&, const py::object&, const py::object&)> _goal_checker;  // last arg used for optional thread_id
        std::function<py::object (py::object&, const py::object&, const py::object&)> _heuristic;  // last arg used for optional thread_id

        std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
        std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};


void init_pymartdp(py::module& m) {
    py::class_<PyMARTDPSolver> py_martdp_solver(m, "_MARTDPSolver_");
        py_martdp_solver
            .def(py::init<py::object&,
                          const std::function<py::object (py::object&, const py::object&, const py::object&)>&,  // last arg used for optional thread_id
                          const std::function<py::object (py::object&, const py::object&, const py::object&)>&,  // last arg used for optional thread_id
                          std::size_t,
                          std::size_t,
                          std::size_t,
                          double,
                          double,
                          bool,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("goal_checker"),
                 py::arg("heuristic"),
                 py::arg("time_budget")=3600000,
                 py::arg("rollout_budget")=100000,
                 py::arg("max_depth")=1000,
                 py::arg("discount")=1.0,
                 py::arg("epsilon")=0.001,
                 py::arg("online_node_garbage")=false,
                 py::arg("parallel")=false,
                 py::arg("debug_logs")=false)
            .def("clear", &PyMARTDPSolver::clear)
            .def("solve", &PyMARTDPSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &PyMARTDPSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &PyMARTDPSolver::get_next_action, py::arg("state"))
            .def("get_utility", &PyMARTDPSolver::get_utility, py::arg("state"))
            .def("get_nb_of_explored_states", &PyMARTDPSolver::get_nb_of_explored_states)
            .def("get_nb_rollouts", &PyMARTDPSolver::get_nb_rollouts)
            .def("get_policy", &PyMARTDPSolver::get_policy)
        ;
}
