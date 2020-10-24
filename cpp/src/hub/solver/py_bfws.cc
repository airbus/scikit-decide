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
class PyBFWSDomain : public skdecide::PythonDomainAdapter<Texecution> {
public :

    PyBFWSDomain(const py::object& domain)
    : skdecide::PythonDomainAdapter<Texecution>(domain) {
        if (!py::hasattr(domain, "get_applicable_actions")) {
            throw std::invalid_argument("SKDECIDE exception: BFWS algorithm needs python domain for implementing get_applicable_actions()");
        }
        if (!py::hasattr(domain, "get_next_state")) {
            throw std::invalid_argument("SKDECIDE exception: BFWS algorithm needs python domain for implementing get_sample()");
        }
        if (!py::hasattr(domain, "get_transition_value")) {
            throw std::invalid_argument("SKDECIDE exception: BFWS algorithm needs python domain for implementing get_transition_value()");
        }
        if (!py::hasattr(domain, "is_terminal")) {
            throw std::invalid_argument("SKDECIDE exception: BFWS algorithm needs python domain for implementing is_terminal()");
        }
    }

};


template <typename Texecution>
using PyBFWSFeatureVector = skdecide::PythonContainerAdapter<Texecution>;


class PyBFWSSolver {
public :
    PyBFWSSolver(py::object& domain,
                 const std::function<py::object (const py::object&, const py::object&)>& state_features,
                 const std::function<py::object (const py::object&, const py::object&)>& heuristic,
                 const std::function<py::object (const py::object&, const py::object&)>& termination_checker,
                 bool use_state_feature_hash = false,
                 bool parallel = false,
                 bool debug_logs = false) {
        if (parallel) {
            if (use_state_feature_hash) {
                _implementation = std::make_unique<Implementation<skdecide::ParallelExecution, skdecide::StateFeatureHash>>(
                    domain, state_features, heuristic, termination_checker, debug_logs
                );
            } else {
                _implementation = std::make_unique<Implementation<skdecide::ParallelExecution, skdecide::DomainStateHash>>(
                    domain, state_features, heuristic, termination_checker, debug_logs
                );
            }
        } else {
            if (use_state_feature_hash) {
                _implementation = std::make_unique<Implementation<skdecide::SequentialExecution, skdecide::StateFeatureHash>>(
                    domain, state_features, heuristic, termination_checker, debug_logs
                );
            } else {
                _implementation = std::make_unique<Implementation<skdecide::SequentialExecution, skdecide::DomainStateHash>>(
                    domain, state_features, heuristic, termination_checker, debug_logs
                );
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

    template <typename Texecution, template <typename...> class Thashing_policy>
    class Implementation : public BaseImplementation {
    public :
        Implementation(py::object& domain,
                       const std::function<py::object (const py::object&, const py::object&)>& state_features,
                       const std::function<py::object (const py::object&, const py::object&)>& heuristic,
                       const std::function<py::object (const py::object&, const py::object&)>& termination_checker,
                       bool debug_logs = false)
        : _state_features(state_features), _heuristic(heuristic), _termination_checker(termination_checker) {
            _domain = std::make_unique<PyBFWSDomain<Texecution>>(domain);
            _solver = std::make_unique<skdecide::BFWSSolver<PyBFWSDomain<Texecution>, PyBFWSFeatureVector<Texecution>, Thashing_policy, Texecution>>(
                                                                            *_domain,
                                                                            [this](PyBFWSDomain<Texecution>& d, const typename PyBFWSDomain<Texecution>::State& s)->std::unique_ptr<PyBFWSFeatureVector<Texecution>> {
                                                                                try {
                                                                                    auto fsf = [this](const py::object& dd, const py::object& ss, [[maybe_unused]] const py::object& ii) {
                                                                                        return _state_features(dd, ss);
                                                                                    };
                                                                                    std::unique_ptr<py::object> r = d.call(nullptr, fsf, s.pyobj());
                                                                                    typename skdecide::GilControl<Texecution>::Acquire acquire;
                                                                                    std::unique_ptr<PyBFWSFeatureVector<Texecution>> rr = std::make_unique<PyBFWSFeatureVector<Texecution>>(*r);
                                                                                    r.reset();
                                                                                    return rr;
                                                                                } catch (const std::exception& e) {
                                                                                    spdlog::error(std::string("SKDECIDE exception when calling state features: ") + e.what());
                                                                                    throw;
                                                                                }
                                                                            },
                                                                            [this](PyBFWSDomain<Texecution>& d, const typename PyBFWSDomain<Texecution>::State& s)->double {
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
                                                                            [this](PyBFWSDomain<Texecution>& d, const typename PyBFWSDomain<Texecution>::State& s)->bool {
                                                                                try {
                                                                                    auto ftc = [this](const py::object& dd, const py::object& ss, [[maybe_unused]] const py::object& ii) {
                                                                                        return _termination_checker(dd, ss);
                                                                                    };
                                                                                    std::unique_ptr<py::object> r = d.call(nullptr, ftc, s.pyobj());
                                                                                    typename skdecide::GilControl<Texecution>::Acquire acquire;
                                                                                    bool rr = r->template cast<bool>();
                                                                                    r.reset();
                                                                                    return  rr;
                                                                                } catch (const std::exception& e) {
                                                                                    spdlog::error(std::string("SKDECIDE exception when calling termination checker: ") + e.what());
                                                                                    throw;
                                                                                }
                                                                            },
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
        std::unique_ptr<PyBFWSDomain<Texecution>> _domain;
        std::unique_ptr<skdecide::BFWSSolver<PyBFWSDomain<Texecution>, PyBFWSFeatureVector<Texecution>, Thashing_policy, Texecution>> _solver;
        
        std::function<py::object (const py::object&, const py::object&)> _state_features;
        std::function<py::object (const py::object&, const py::object&)> _heuristic;
        std::function<py::object (const py::object&, const py::object&)> _termination_checker;

        std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
        std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};


void init_pybfws(py::module& m) {
    py::class_<PyBFWSSolver> py_bfws_solver(m, "_BFWSSolver_");
        py_bfws_solver
            .def(py::init<py::object&,
                          const std::function<py::object (const py::object&, const py::object&)>&,
                          const std::function<py::object (const py::object&, const py::object&)>&,
                          const std::function<py::object (const py::object&, const py::object&)>&,
                          bool,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("state_features"),
                 py::arg("heuristic"),
                 py::arg("termination_checker"),
                 py::arg("use_state_feature_hash")=false,
                 py::arg("parallel")=false,
                 py::arg("debug_logs")=false)
            .def("clear", &PyBFWSSolver::clear)
            .def("solve", &PyBFWSSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &PyBFWSSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &PyBFWSSolver::get_next_action, py::arg("state"))
            .def("get_utility", &PyBFWSSolver::get_utility, py::arg("state"))
        ;
}
