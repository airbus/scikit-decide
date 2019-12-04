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
#include "utils/python_hash_eq.hh"

namespace py = pybind11;

class PyWRLDomain {
public :
    struct Observation {
        py::object _observation;

        Observation() {}
        Observation(const py::object& s) : _observation(s) {}

        std::string print() const {
            return py::str(_observation);
        }

        struct Hash {
            std::size_t operator()(const Observation& s) const {
                try {
                    return airlaps::python_hash(s._observation);
                } catch(const py::error_already_set& e) {
                    spdlog::error(std::string("AIRLAPS exception when hashing observations: ") + e.what());
                    throw;
                }
            }
        };

        struct Equal {
            bool operator()(const Observation& s1, const Observation& s2) const {
                try {
                    return airlaps::python_equal(s1._observation, s2._observation);
                } catch(const py::error_already_set& e) {
                    spdlog::error(std::string("AIRLAPS exception when testing observations equality: ") + e.what());
                    throw;
                }
            }
        };
    };

    struct Event {
        py::object _event;

        Event() {}
        Event(const py::object& e) : _event(e) {}
        
        const py::object& get() const { return _event; }

        std::string print() const {
            return py::str(_event);
        }

        struct Hash {
            std::size_t operator()(const Event& e) const {
                try {
                    return airlaps::python_hash(e._event);
                } catch(const py::error_already_set& ex) {
                    spdlog::error(std::string("AIRLAPS exception when hashing actions: ") + ex.what());
                    throw;
                }
            }
        };

        struct Equal {
            bool operator()(const Event& e1, const Event& e2) const {
                try {
                    return airlaps::python_equal(e1._event, e2._event);
                } catch(const py::error_already_set& ex) {
                    spdlog::error(std::string("AIRLAPS exception when testing actions equality: ") + ex.what());
                    throw;
                }
            }
        };
    };

    struct EnvironmentOutcome {
        py::object _outcome;

        EnvironmentOutcome(const py::object& outcome)
        : _outcome(outcome) {
            if (!py::hasattr(_outcome, "observation")) {
                throw std::invalid_argument("AIRLAPS exception: width-based proxy domain needs python environment outcome object for providing 'observation'");
            }
            if (!py::hasattr(_outcome, "value")) {
                throw std::invalid_argument("AIRLAPS exception: width-based proxy domain needs python transition outcome object for providing 'value'");
            }
            if (!py::hasattr(_outcome, "termination")) {
                throw std::invalid_argument("AIRLAPS exception: width-based proxy domain needs python transition outcome object for providing 'termination'");
            }
        }

        EnvironmentOutcome(const EnvironmentOutcome& other) {
            _outcome = py::module::import("copy").attr("deepcopy")(other._outcome);
        }

        EnvironmentOutcome& operator= (const EnvironmentOutcome& other) {
            this->_outcome = py::module::import("copy").attr("deepcopy")(other._outcome);
            return *this;
        }

        py::object observation() {
            try {
                return _outcome.attr("observation");
            } catch(const py::error_already_set& e) {
                spdlog::error(std::string("AIRLAPS exception when getting outcome's observation: ") + e.what());
                throw;
            }
        }

        void observation(const py::object& s) {
            try {
                _outcome.attr("observation") = s;
            } catch(const py::error_already_set& e) {
                spdlog::error(std::string("AIRLAPS exception when setting outcome's observation: ") + e.what());
                throw;
            }
        }

        double reward() {
            try {
                return py::cast<double>(_outcome.attr("value").attr("reward"));
            } catch(const py::error_already_set& e) {
                spdlog::error(std::string("AIRLAPS exception when getting outcome's reward: ") + e.what());
                throw;
            }
        }

        void reward(double r) {
            try {
                _outcome.attr("value").attr("reward") = py::float_(r);
            } catch(const py::error_already_set& e) {
                spdlog::error(std::string("AIRLAPS exception when setting outcome's reward: ") + e.what());
                throw;
            }
        }

        bool termination() {
            try {
                return py::cast<bool>(_outcome.attr("termination"));
            } catch(const py::error_already_set& e) {
                spdlog::error(std::string("AIRLAPS exception when getting outcome's observation: ") + e.what());
                throw;
            }
        }

        void termination(bool t) {
            try {
                _outcome.attr("termination") = py::bool_(t);
            } catch(const py::error_already_set& e) {
                spdlog::error(std::string("AIRLAPS exception when setting outcome's observation: ") + e.what());
                throw;
            }
        }

        py::object info() {
            try {
                return _outcome.attr("info");
            } catch(const py::error_already_set& e) {
                spdlog::error(std::string("AIRLAPS exception when getting outcome's info: ") + e.what());
                throw;
            }
        }

        void info(const py::object& i) {
            try {
                _outcome.attr("info") = i;
            } catch(const py::error_already_set& e) {
                spdlog::error(std::string("AIRLAPS exception when setting outcome's info: ") + e.what());
                throw;
            }
        }

        static std::size_t get_depth(const py::object& info) {
            if (py::hasattr(info, "depth")) {
                return py::cast<std::size_t>(info.attr("depth")());
            } else {
                return 0;
            }
        }
    };

    PyWRLDomain(const py::object& domain)
    : _domain(domain) {
        if (!py::hasattr(domain, "reset")) {
            throw std::invalid_argument("AIRLAPS exception: width-based proxy domain needs python domain for implementing reset()");
        }
        if (!py::hasattr(domain, "step") && !py::hasattr(domain, "sample")) {
            throw std::invalid_argument("AIRLAPS exception: width-based proxy domain needs python domain for implementing step() or sample()");
        }
    }

    std::unique_ptr<Observation> reset() {
        try {
            return std::make_unique<Observation>(_domain.attr("reset")());
        } catch(const py::error_already_set& ex) {
            spdlog::error(std::string("AIRLAPS exception when resetting the domain: ") + ex.what());
            throw;
        }
    }

    std::unique_ptr<EnvironmentOutcome> step(const Event& e) {
        try {
            return std::make_unique<EnvironmentOutcome>(_domain.attr("step")(e._event));
        } catch(const py::error_already_set& ex) {
            spdlog::error(std::string("AIRLAPS exception when stepping with action ") +
                          e.print() + ": " + ex.what());
            throw;
        }
    }

    std::unique_ptr<EnvironmentOutcome> sample(const Observation& s, const Event& e) {
        try {
            return std::make_unique<EnvironmentOutcome>(_domain.attr("sample")(s._observation, e._event));
        } catch(const py::error_already_set& ex) {
            spdlog::error(std::string("AIRLAPS exception when sampling from observation ") + s.print() +
                          " with action " + e.print() + ": " + ex.what());
            throw;
        }
    }

private :
    py::object _domain;
};


class PyWRLFeatureVector {
public :

    class value_type {
    public :
        value_type() {}
        
        value_type(const py::object& value) {
            _value = std::make_unique<ObjectType>(value);
        }

        template <typename T,
                  typename  = typename std::enable_if<std::is_fundamental<T>::value>::type>
        value_type(const T& value) {
            _value = std::make_unique<PrimitiveType<T>>(value);
        }

        value_type(const value_type& other) {
            other._value->copy(_value);
        }

        value_type& operator= (const value_type& other) {
            other._value->copy(_value);
            return *this;
        }

        std::size_t hash() const {
            return _value->hash();
        }

        bool operator== (const value_type& other) const {
            return _value->equal(*(other._value));
        }

    private :
        class BaseType {
        public :
            virtual void copy(std::unique_ptr<BaseType>& other) const =0;
            virtual std::size_t hash() const =0;
            virtual bool equal(const BaseType& other) const =0;
        };

        template <typename T>
        class PrimitiveType : public BaseType {
        public :
            PrimitiveType(const T& value) : _value(value) {}

            virtual void copy(std::unique_ptr<BaseType>& other) const {
                other = std::make_unique<PrimitiveType<T>>(_value);
            }

            virtual std::size_t hash() const {
                return boost::hash_value(_value);
            }

            virtual bool equal(const BaseType& other) const {
                const PrimitiveType<T>* o = dynamic_cast<const PrimitiveType<T>*>(&other);
                return ((o != nullptr) && (o->_value == _value));
            }

        private :
            T _value;
        };

        class ObjectType : public BaseType {
        public :
            ObjectType(const py::object& value) : _value(value) {}

            ~ObjectType() {
                _value = py::object();
            }

            virtual void copy(std::unique_ptr<BaseType>& other) const {
                other = std::make_unique<ObjectType>(_value);
            }

            virtual std::size_t hash() const {
                try {
                    return airlaps::python_hash(_value);
                } catch(const py::error_already_set& ex) {
                    spdlog::error(std::string("AIRLAPS exception when hashing observation feature items: ") + ex.what());
                    throw;
                }
            }

            virtual bool equal(const BaseType& other) const {
                try {
                    const ObjectType* o = dynamic_cast<const ObjectType*>(&other);
                    return  ((o != nullptr) && airlaps::python_equal(_value, o->_value));
                } catch(const py::error_already_set& ex) {
                    spdlog::error(std::string("AIRLAPS exception when testing observation feature items equality: ") + ex.what());
                    throw;
                }
            }

        private :
            py::object _value;
        };

        std::unique_ptr<BaseType> _value;
    };

    PyWRLFeatureVector() {}

    PyWRLFeatureVector(const py::object& vector) {
        if (py::isinstance<py::list>(vector)) {
            _implementation = std::make_unique<SequenceImplementation<py::list>>(vector);
        } else if (py::isinstance<py::tuple>(vector)) {
            _implementation = std::make_unique<SequenceImplementation<py::tuple>>(vector);
        } else if (py::isinstance<py::array>(vector)) {
            std::string dtype = py::str(vector.attr("dtype"));
            if (dtype == "float_") {
                _implementation = std::make_unique<NumpyImplementation<double>>(vector);
            } else if (dtype == "float32") {
                _implementation = std::make_unique<NumpyImplementation<float>>(vector);
            } else if (dtype == "float64") {
                _implementation = std::make_unique<NumpyImplementation<double>>(vector);
            } else if (dtype == "bool_") {
                _implementation = std::make_unique<NumpyImplementation<bool>>(vector);
            } else if (dtype == "int_") {
                _implementation = std::make_unique<NumpyImplementation<long int>>(vector);
            } else if (dtype == "intc") {
                _implementation = std::make_unique<NumpyImplementation<int>>(vector);
            } else if (dtype == "intp") {
                _implementation = std::make_unique<NumpyImplementation<std::size_t>>(vector);
            } else if (dtype == "int8") {
                _implementation = std::make_unique<NumpyImplementation<std::int8_t>>(vector);
            } else if (dtype == "int16") {
                _implementation = std::make_unique<NumpyImplementation<std::int16_t>>(vector);
            } else if (dtype == "int32") {
                _implementation = std::make_unique<NumpyImplementation<std::int32_t>>(vector);
            } else if (dtype == "int64") {
                _implementation = std::make_unique<NumpyImplementation<std::int64_t>>(vector);
            } else if (dtype == "uint8") {
                _implementation = std::make_unique<NumpyImplementation<std::uint8_t>>(vector);
            } else if (dtype == "uint16") {
                _implementation = std::make_unique<NumpyImplementation<std::uint16_t>>(vector);
            } else if (dtype == "uint32") {
                _implementation = std::make_unique<NumpyImplementation<std::uint32_t>>(vector);
            } else if (dtype == "uint64") {
                _implementation = std::make_unique<NumpyImplementation<std::uint64_t>>(vector);
            } else {
                spdlog::error("Unhandled array dtype '" + dtype + "' when parsing observation features as numpy array");
                throw std::invalid_argument("AIRLAPS exception: Unhandled array dtype '" + dtype +
                                            "' when parsing observation features as numpy array");
            }
        } else {
            spdlog::error("Unhandled observation feature type '" + std::string(py::str(vector.attr("__class__").attr("__name__"))) +
                           " (expecting list, tuple or numpy array)");
            throw std::invalid_argument("Unhandled observation feature type '" + std::string(py::str(vector.attr("__class__").attr("__name__"))) +
                                        " (expecting list, tuple or numpy array)");
        }
    }

    std::size_t size() const {
        return _implementation->size();
    }

    value_type operator[](std::size_t index) const {
        return _implementation->at(index);
    }

private :

    class BaseImplementation {
    public :
        virtual std::size_t size() const =0;
        virtual value_type at(std::size_t index) const =0;
    };

    template <typename Tsequence>
    class SequenceImplementation : public BaseImplementation {
    public :
        SequenceImplementation(const py::object& vector)
        : _vector(static_cast<const Tsequence&>(vector)) {}

        virtual std::size_t size() const {
            return _vector.size();
        }

        virtual value_type at(std::size_t index) const {
            return value_type(_vector[index]);
        }

    private :
        Tsequence _vector;
    };

    template <typename T>
    class NumpyImplementation : public BaseImplementation {
    public :
        NumpyImplementation(const py::object& vector)
        : _vector(static_cast<const py::array_t<T>&>(vector)) {
            _buffer = _vector.request();
        }

        virtual std::size_t size() const {
            return _vector.size();
        }

        virtual value_type at(std::size_t index) const {
            return value_type(((T*) _buffer.ptr)[index]);
        }
    
    private :
        py::array_t<T> _vector;
        py::buffer_info _buffer;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};


std::size_t hash_value(const PyWRLFeatureVector::value_type& o) {
    return o.hash();
}


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
            _implementation = std::make_unique<Implementation<airlaps::ObservationFeatureHash>>(
                domain, observation_features, initial_pruning_probability, temperature_increase_rate,
                width_increase_resilience, max_depth, cache_transitions, debug_logs);
        } else {
            _implementation = std::make_unique<Implementation<airlaps::DomainObservationHash>>(
                domain, observation_features, initial_pruning_probability, temperature_increase_rate,
                width_increase_resilience, max_depth, cache_transitions, debug_logs);
        }
    }

    template <typename TWRLSolverDomainFilterPtr,
              std::enable_if_t<std::is_same<typename TWRLSolverDomainFilterPtr::element_type::Solver::HashingPolicy,
                                            airlaps::DomainObservationHash<typename TWRLSolverDomainFilterPtr::element_type::Solver::Domain,
                                                                     typename TWRLSolverDomainFilterPtr::element_type::Solver::FeatureVector>>::value, int> = 0>
    PyWRLDomainFilter(TWRLSolverDomainFilterPtr domain_filter) {
        _implementation = std::make_unique<Implementation<airlaps::DomainObservationHash>>(std::move(domain_filter));
    }

    template <typename TWRLSolverDomainFilterPtr,
              std::enable_if_t<std::is_same<typename TWRLSolverDomainFilterPtr::element_type::Solver::HashingPolicy,
                                            airlaps::ObservationFeatureHash<typename TWRLSolverDomainFilterPtr::element_type::Solver::Domain,
                                                                      typename TWRLSolverDomainFilterPtr::element_type::Solver::FeatureVector>>::value, int> = 0>
    PyWRLDomainFilter(TWRLSolverDomainFilterPtr domain_filter) {
        _implementation = std::make_unique<Implementation<airlaps::ObservationFeatureHash>>(std::move(domain_filter));
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
                            return std::make_unique<PyWRLFeatureVector>(_observation_features(s._observation));
                        } catch (const py::error_already_set& e) {
                            spdlog::error(std::string("AIRLAPS exception when calling observation features: ") + e.what());
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
                            return std::make_unique<PyWRLFeatureVector>(_observation_features(s._observation));
                        } catch (const py::error_already_set& e) {
                            spdlog::error(std::string("AIRLAPS exception when calling observation features: ") + e.what());
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

        Implementation(std::unique_ptr<typename airlaps::WRLSolver<PyWRLDomain,
                                                                   PyWRLUnderlyingSolver,
                                                                   PyWRLFeatureVector,
                                                                   Thashing_policy>::WRLDomainFilter> domain_filter) {
            _domain_filter = std::move(domain_filter);
            _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(std::cout,
                                                                             py::module::import("sys").attr("stdout"));
            _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(std::cerr,
                                                                             py::module::import("sys").attr("stderr"));
        }

        virtual py::object reset() {
            return _domain_filter->reset()->_observation;
        }

        virtual py::object step(const py::object& action) {
            return _domain_filter->step(action)->_outcome;
        }

        virtual py::object sample(const py::object& observation, const py::object& action) {
            return _domain_filter->sample(observation, action)->_outcome;
        }
    
    private :
        typedef airlaps::WRLSolver<PyWRLDomain, PyWRLUnderlyingSolver, PyWRLFeatureVector, Thashing_policy> _WRLSolver;
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
            throw std::invalid_argument("AIRLAPS exception: RWL algorithm needs the original solver to provide the 'reset' method");
        }
        if (!py::hasattr(solver, "solve")) {
            throw std::invalid_argument("AIRLAPS exception: RWL algorithm needs the original solver to provide the 'solve' method");
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
            spdlog::error(std::string("AIRLAPS exception when calling the original solve method: ") + e.what());
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
            _implementation = std::make_unique<Implementation<airlaps::ObservationFeatureHash>>(
                solver, observation_features, initial_pruning_probability, temperature_increase_rate,
                width_increase_resilience, max_depth, cache_transitions, debug_logs
            );
        } else {
            _implementation = std::make_unique<Implementation<airlaps::DomainObservationHash>>(
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
            
            _solver = std::make_unique<airlaps::WRLSolver<PyWRLDomain, PyWRLUnderlyingSolver, PyWRLFeatureVector, Thashing_policy>>(
                *_underlying_solver,
                [this](const typename PyWRLDomain::Observation& s)->std::unique_ptr<PyWRLFeatureVector> {
                    try {
                        return std::make_unique<PyWRLFeatureVector>(_observation_features(s._observation));
                    } catch (const py::error_already_set& e) {
                        spdlog::error(std::string("AIRLAPS exception when calling observation features: ") + e.what());
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

        virtual void reset() {
            _solver->reset();
        }

        virtual void solve(const std::function<py::object ()>& domain_factory) {
            _solver->solve([&domain_factory] () -> std::unique_ptr<PyWRLDomain> {
                return std::make_unique<PyWRLDomain>(domain_factory());
            });
        }

    private :
        typedef airlaps::WRLSolver<PyWRLDomain, PyWRLUnderlyingSolver, PyWRLFeatureVector, Thashing_policy> _WRLSolver;
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