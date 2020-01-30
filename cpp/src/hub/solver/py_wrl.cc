/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "wrl.hh"
#include "core.hh"

#include "utils/python_gil_control.hh"
#include "utils/python_hash_eq.hh"
#include "utils/python_domain_adapter.hh"

namespace py = pybind11;

class PyWRLDomain : public skdecide::PythonDomainAdapter<skdecide::SequentialExecution> {
public :

    PyWRLDomain(const py::object& domain)
    : skdecide::PythonDomainAdapter<skdecide::SequentialExecution>(domain) {
        if (!py::hasattr(domain, "reset")) {
            throw std::invalid_argument("SKDECIDE exception: width-based proxy domain needs python domain for implementing reset()");
        }
        if (!py::hasattr(domain, "step") && !py::hasattr(domain, "sample")) {
            throw std::invalid_argument("SKDECIDE exception: width-based proxy domain needs python domain for implementing step() or sample()");
        }
    }

};


using PyWRLFeatureVector = skdecide::PythonContainerAdapter<skdecide::SequentialExecution>;


class PyWRLUnderlyingSolver;


class PyWRLDomainFilter {
public :

    PyWRLDomainFilter(py::object& domain,
                      const std::function<py::object (const py::object&)>& observation_features,
                      double initial_pruning_probability = 0.999,
                      double temperature_increase_rate = 0.01,
                      std::size_t width_increase_resilience = 10,
                      std::size_t max_depth = 1000,
                      bool use_observation_feature_hash = false,
                      bool cache_transitions = false,
                      bool debug_logs = false) {
        if (use_observation_feature_hash) {
            _implementation = std::make_unique<Implementation<skdecide::ObservationFeatureHash>>(
                domain, observation_features, initial_pruning_probability, temperature_increase_rate,
                width_increase_resilience, max_depth, cache_transitions, debug_logs);
        } else {
            _implementation = std::make_unique<Implementation<skdecide::DomainObservationHash>>(
                domain, observation_features, initial_pruning_probability, temperature_increase_rate,
                width_increase_resilience, max_depth, cache_transitions, debug_logs);
        }
    }

    template <typename TWRLSolverDomainFilterPtr,
              std::enable_if_t<std::is_same<typename TWRLSolverDomainFilterPtr::element_type::Solver::HashingPolicy,
                                            skdecide::DomainObservationHash<typename TWRLSolverDomainFilterPtr::element_type::Solver::Domain,
                                                                     typename TWRLSolverDomainFilterPtr::element_type::Solver::FeatureVector>>::value, int> = 0>
    PyWRLDomainFilter(TWRLSolverDomainFilterPtr domain_filter) {
        _implementation = std::make_unique<Implementation<skdecide::DomainObservationHash>>(std::move(domain_filter));
    }

    template <typename TWRLSolverDomainFilterPtr,
              std::enable_if_t<std::is_same<typename TWRLSolverDomainFilterPtr::element_type::Solver::HashingPolicy,
                                            skdecide::ObservationFeatureHash<typename TWRLSolverDomainFilterPtr::element_type::Solver::Domain,
                                                                      typename TWRLSolverDomainFilterPtr::element_type::Solver::FeatureVector>>::value, int> = 0>
    PyWRLDomainFilter(TWRLSolverDomainFilterPtr domain_filter) {
        _implementation = std::make_unique<Implementation<skdecide::ObservationFeatureHash>>(std::move(domain_filter));
    }

    py::object reset() {
        return _implementation->reset();
    }

    py::object step(const py::object& action) {
        return _implementation->step(action);
    }

    py::object sample(const py::object& observation, const py::object& action) {
        return _implementation->sample(observation, action);
    }

private :

    class BaseImplementation {
    public :
        virtual ~BaseImplementation() {}
        virtual py::object reset() =0;
        virtual py::object step(const py::object& action) =0;
        virtual py::object sample(const py::object& observation, const py::object& action) =0;
    };

    template <template <typename...> class Thashing_policy>
    class Implementation : public BaseImplementation {
    public :
        Implementation(py::object& domain,
                       const std::function<py::object (const py::object&)>& observation_features,
                       double initial_pruning_probability = 0.999,
                       double temperature_increase_rate = 0.01,
                       std::size_t width_increase_resilience = 10,
                       std::size_t max_depth = 1000,
                       bool cache_transitions = false,
                       bool debug_logs = false)
            : _observation_features(observation_features) {
            
            std::unique_ptr<PyWRLDomain> wrl_domain = std::make_unique<PyWRLDomain>(domain);

            if (cache_transitions) {
                _domain_filter = std::make_unique<typename _WRLSolver::WRLCachedDomainFilter>(
                    std::move(wrl_domain),
                    [this](const typename PyWRLDomain::Observation& s)->std::unique_ptr<PyWRLFeatureVector> {
                        try {
                            return std::make_unique<PyWRLFeatureVector>(_observation_features(s._state));
                        } catch (const py::error_already_set& e) {
                            spdlog::error(std::string("SKDECIDE exception when calling observation features: ") + e.what());
                            throw;
                        }
                    },
                    initial_pruning_probability, temperature_increase_rate,
                    width_increase_resilience, max_depth, debug_logs
                );
            } else {
                _domain_filter = std::make_unique<typename _WRLSolver::WRLUncachedDomainFilter>(
                    std::move(wrl_domain),
                    [this](const typename PyWRLDomain::Observation& s)->std::unique_ptr<PyWRLFeatureVector> {
                        try {
                            return std::make_unique<PyWRLFeatureVector>(_observation_features(s._state));
                        } catch (const py::error_already_set& e) {
                            spdlog::error(std::string("SKDECIDE exception when calling observation features: ") + e.what());
                            throw;
                        }
                    },
                    initial_pruning_probability, temperature_increase_rate,
                    width_increase_resilience, max_depth, debug_logs
                );
            }

            _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(std::cout,
                                                                             py::module::import("sys").attr("stdout"));
            _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(std::cerr,
                                                                             py::module::import("sys").attr("stderr"));
        }

        Implementation(std::unique_ptr<typename skdecide::WRLSolver<PyWRLDomain,
                                                                   PyWRLUnderlyingSolver,
                                                                   PyWRLFeatureVector,
                                                                   Thashing_policy>::WRLDomainFilter> domain_filter) {
            _domain_filter = std::move(domain_filter);
            _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(std::cout,
                                                                             py::module::import("sys").attr("stdout"));
            _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(std::cerr,
                                                                             py::module::import("sys").attr("stderr"));
        }

        virtual ~Implementation() {}

        virtual py::object reset() {
            return _domain_filter->reset()->_state;
        }

        virtual py::object step(const py::object& action) {
            return _domain_filter->step(action)->_outcome;
        }

        virtual py::object sample(const py::object& observation, const py::object& action) {
            return _domain_filter->sample(observation, action)->_outcome;
        }
    
    private :
        typedef skdecide::WRLSolver<PyWRLDomain, PyWRLUnderlyingSolver, PyWRLFeatureVector, Thashing_policy> _WRLSolver;
        std::unique_ptr<typename _WRLSolver::WRLDomainFilter> _domain_filter;

        std::function<py::object (const py::object&)> _observation_features;

        std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
        std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};


class PyWRLUnderlyingSolver {
public :
    PyWRLUnderlyingSolver(py::object& solver)
    : _solver(solver) {
        if (!py::hasattr(solver, "reset")) {
            throw std::invalid_argument("SKDECIDE exception: RWL algorithm needs the original solver to provide the 'reset' method");
        }
        if (!py::hasattr(solver, "solve")) {
            throw std::invalid_argument("SKDECIDE exception: RWL algorithm needs the original solver to provide the 'solve' method");
        }
    }

    void reset() {
        if (py::hasattr(_solver, "reset")) {
            _solver.attr("reset")();
        }
    }

    template <typename TWRLSolverDomainFilterFactory>
    void solve(const TWRLSolverDomainFilterFactory& domain_factory) {
        try {
            _solver.attr("solve")([&domain_factory]() -> std::unique_ptr<PyWRLDomainFilter> {
                return std::make_unique<PyWRLDomainFilter>(domain_factory());
            });
        } catch(const py::error_already_set& e) {
            spdlog::error(std::string("SKDECIDE exception when calling the original solve method: ") + e.what());
            throw;
        }
    }

private :
    py::object _solver;
};


class PyWRLSolver {
public :

    PyWRLSolver(py::object& solver,
                const std::function<py::object (const py::object&)>& observation_features,
                double initial_pruning_probability = 0.999,
                double temperature_increase_rate = 0.01,
                std::size_t width_increase_resilience = 10,
                std::size_t max_depth = 1000,
                bool use_observation_feature_hash = false,
                bool cache_transitions = false,
                bool debug_logs = false) {
        
        if (use_observation_feature_hash) {
            _implementation = std::make_unique<Implementation<skdecide::ObservationFeatureHash>>(
                solver, observation_features, initial_pruning_probability, temperature_increase_rate,
                width_increase_resilience, max_depth, cache_transitions, debug_logs
            );
        } else {
            _implementation = std::make_unique<Implementation<skdecide::DomainObservationHash>>(
                solver, observation_features, initial_pruning_probability, temperature_increase_rate,
                width_increase_resilience, max_depth, cache_transitions, debug_logs
            );
        }
    }

    void reset() {
        _implementation->reset();
    }

    void solve(const std::function<py::object ()>& domain_factory) {
        _implementation->solve(domain_factory);
    }

private :

    class BaseImplementation {
    public :
        virtual ~BaseImplementation() {}
        virtual void reset() =0;
        virtual void solve(const std::function<py::object ()>& domain_factory) =0;
    };

    template <template <typename...> class Thashing_policy>
    class Implementation : public BaseImplementation {
    public :

        Implementation(py::object& solver,
                       const std::function<py::object (const py::object&)>& observation_features,
                       double initial_pruning_probability = 0.999,
                       double temperature_increase_rate = 0.1,
                       std::size_t width_increase_resilience = 10,
                       std::size_t max_depth = 1000,
                       bool cache_transitions = false,
                       bool debug_logs = false)
            : _observation_features(observation_features) {

            _underlying_solver = std::make_unique<PyWRLUnderlyingSolver>(solver);
            
            _solver = std::make_unique<skdecide::WRLSolver<PyWRLDomain, PyWRLUnderlyingSolver, PyWRLFeatureVector, Thashing_policy>>(
                *_underlying_solver,
                [this](const typename PyWRLDomain::Observation& s)->std::unique_ptr<PyWRLFeatureVector> {
                    try {
                        return std::make_unique<PyWRLFeatureVector>(_observation_features(s._state));
                    } catch (const py::error_already_set& e) {
                        spdlog::error(std::string("SKDECIDE exception when calling observation features: ") + e.what());
                        throw;
                    }
                },
                initial_pruning_probability,
                temperature_increase_rate,
                width_increase_resilience,
                max_depth,
                cache_transitions,
                debug_logs
            );
            _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(std::cout,
                                                                             py::module::import("sys").attr("stdout"));
            _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(std::cerr,
                                                                             py::module::import("sys").attr("stderr"));
        }

        virtual ~Implementation() {}

        virtual void reset() {
            _solver->reset();
        }

        virtual void solve(const std::function<py::object ()>& domain_factory) {
            _solver->solve([&domain_factory] () -> std::unique_ptr<PyWRLDomain> {
                return std::make_unique<PyWRLDomain>(domain_factory());
            });
        }

    private :
        typedef skdecide::WRLSolver<PyWRLDomain, PyWRLUnderlyingSolver, PyWRLFeatureVector, Thashing_policy> _WRLSolver;
        std::unique_ptr<_WRLSolver> _solver;
        std::unique_ptr<PyWRLUnderlyingSolver> _underlying_solver;
        
        std::function<py::object (const py::object&)> _observation_features;

        std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
        std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};


void init_pywrl(py::module& m) {
    py::class_<PyWRLDomainFilter> py_wrl_domain_filter(m, "_WRLDomainFilter_");
        py_wrl_domain_filter
            .def(py::init<py::object&,
                          const std::function<py::object (const py::object&)>&,
                          double,
                          double,
                          std::size_t,
                          std::size_t,
                          bool,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("observation_features"),
                 py::arg("initial_pruning_probability")=0.999,
                 py::arg("temperature_increase_rate")=0.01,
                 py::arg("width_increase_resilience")=10,
                 py::arg("max_depth")=1000,
                 py::arg("use_observation_feature_hash")=false,
                 py::arg("cache_transitions")=false,
                 py::arg("debug_logs")=false)
            .def("reset", &PyWRLDomainFilter::reset)
            .def("step", &PyWRLDomainFilter::step, py::arg("action"))
            .def("sample", &PyWRLDomainFilter::sample, py::arg("observation"), py::arg("action"))
        ;
    
    py::class_<PyWRLSolver> py_wrl_solver(m, "_WRLSolver_");
        py_wrl_solver
            .def(py::init<py::object&,
                          const std::function<py::object (const py::object&)>&,
                          double,
                          double,
                          std::size_t,
                          std::size_t,
                          bool,
                          bool,
                          bool>(),
                 py::arg("solver"),
                 py::arg("observation_features"),
                 py::arg("initial_pruning_probability")=0.999,
                 py::arg("temperature_increase_rate")=0.01,
                 py::arg("width_increase_resilience")=10,
                 py::arg("max_depth")=1000,
                 py::arg("use_observation_feature_hash")=false,
                 py::arg("cache_transitions")=false,
                 py::arg("debug_logs")=false)
            .def("reset", &PyWRLSolver::reset)
            .def("solve", &PyWRLSolver::solve, py::arg("domain_factory"))
        ;
}