/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PYTHON_DOMAIN_ADAPTER_HH
#define AIRLAPS_PYTHON_DOMAIN_ADAPTER_HH

#include <pybind11/pybind11.h>

#include "utils/python_gil_control.hh"
#include "utils/python_hash_eq.hh"
#include "utils/execution.hh"

namespace py = pybind11;

namespace airlaps {

template <typename Texecution>
class PythonDomainAdapter {
public :
    struct State {
        py::object _state;

        State() {}

        State(const py::object& s) {
             typename GilControl<Texecution>::Acquire acquire;
             this->_state = s;
        }

        State(const State& other) {
             typename GilControl<Texecution>::Acquire acquire;
             this->_state = other._state;
        }

        State& operator=(const State& other) {
             typename GilControl<Texecution>::Acquire acquire;
             this->_state = other._state;
             return *this;
        }

        ~State() {
            typename GilControl<Texecution>::Acquire acquire;
            _state = py::object();
        }

        std::string print() const {
            typename GilControl<Texecution>::Acquire acquire;
            return py::str(_state);
        }

        struct Hash {
            std::size_t operator()(const State& s) const {
                try {
                    return airlaps::PythonHash<Texecution>()(s._state);
                } catch(const std::exception& e) {
                    spdlog::error(std::string("AIRLAPS exception when hashing states: ") + e.what());
                    throw;
                }
            }
        };

        struct Equal {
            bool operator()(const State& s1, const State& s2) const {
                try {
                    return airlaps::PythonEqual<Texecution>()(s1._state, s2._state);
                } catch(const std::exception& e) {
                    spdlog::error(std::string("AIRLAPS exception when testing states equality: ") + e.what());
                    throw;
                }
            }
        };
    };

    typedef State Observation;

    struct Event {
        py::object _event;

        Event() {}

        Event(const py::object& e) {
             typename GilControl<Texecution>::Acquire acquire;
             this->_event = e;
        }

        Event(const py::handle& e) {
             typename GilControl<Texecution>::Acquire acquire;
             this->_event = py::reinterpret_borrow<py::object>(e);
        }

        Event(const Event& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_event = other._event;
        }

        Event& operator=(const Event& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_event = other._event;
            return *this;
        }

        ~Event() {
            typename GilControl<Texecution>::Acquire acquire;
            _event = py::object();
        }
        
        const py::object& get() const { return _event; }

        std::string print() const {
            typename GilControl<Texecution>::Acquire acquire;
            return py::str(_event);
        }

        struct Hash {
            std::size_t operator()(const Event& e) const {
                try {
                    return airlaps::PythonHash<Texecution>()(e._event);
                } catch(const std::exception& ex) {
                    spdlog::error(std::string("AIRLAPS exception when hashing events: ") + ex.what());
                    throw;
                }
            }
        };

        struct Equal {
            bool operator()(const Event& e1, const Event& e2) const {
                try {
                    return airlaps::PythonEqual<Texecution>()(e1._event, e2._event);
                } catch(const std::exception& ex) {
                    spdlog::error(std::string("AIRLAPS exception when testing events equality: ") + ex.what());
                    throw;
                }
            }
        };
    };

    typedef Event Action;

    struct ApplicableActionSpace { // don't inherit from airlaps::EnumerableSpace since otherwise we would need to copy the applicable action python object into a c++ iterable object
        py::object _applicable_actions;

        ApplicableActionSpace(const py::object& applicable_actions) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_applicable_actions = applicable_actions;
        }

        ApplicableActionSpace(const ApplicableActionSpace& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_applicable_actions = other.applicable_actions;
        }

        ApplicableActionSpace& operator=(const ApplicableActionSpace& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_applicable_actions = other.applicable_actions;
            return *this;
        }

        ~ApplicableActionSpace() {
            typename GilControl<Texecution>::Acquire acquire;
            _applicable_actions = py::object();
        }

        struct ApplicableActionSpaceElements {
            py::object _elements;
            
            ApplicableActionSpaceElements(const py::object& elements) {
                typename GilControl<Texecution>::Acquire acquire;
                this->_elements = elements;
            }

            ApplicableActionSpaceElements(const ApplicableActionSpaceElements& other) {
                typename GilControl<Texecution>::Acquire acquire;
                this->_elements = other._elements;
            }

            ApplicableActionSpaceElements& operator=(const ApplicableActionSpaceElements& other) {
                typename GilControl<Texecution>::Acquire acquire;
                this->_elements = other._elements;
                return *this;
            }

            ~ApplicableActionSpaceElements() {
                typename GilControl<Texecution>::Acquire acquire;
                _elements = py::object();
            }

            py::iterator begin() const {
                typename GilControl<Texecution>::Acquire acquire;
                return _elements.begin();
            }

            py::iterator end() const {
                typename GilControl<Texecution>::Acquire acquire;
                return _elements.end();
            }
        };

        ApplicableActionSpaceElements get_elements() const {
            typename GilControl<Texecution>::Acquire acquire;
            if (!py::hasattr(_applicable_actions, "get_elements")) {
                throw std::invalid_argument("AIRLAPS exception: python applicable action object must implement get_elements()");
            } else {
                return ApplicableActionSpaceElements(_applicable_actions.attr("get_elements")());
            }
        }

        std::unique_ptr<Event> sample() const {
            typename GilControl<Texecution>::Acquire acquire;
            if (!py::hasattr(_applicable_actions, "sample")) {
                throw std::invalid_argument("AIRLAPS exception: python applicable action object must implement sample()");
            } else {
                return std::make_unique<Event>(_applicable_actions.attr("sample")());
            }
        }
    };

    struct TransitionOutcome {
        py::object _outcome;
        py::object _state;

        TransitionOutcome(const py::object& outcome) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_outcome = outcome;
            if (py::hasattr(_outcome, "state")) {
                _state = _outcome.attr("state");
            } else if (py::hasattr(_outcome, "observation")) {
                _state = _outcome.attr("observation");
            } else {
                throw std::invalid_argument("AIRLAPS exception: python transition outcome object must provide 'state' or 'observation'");
            }
            if (!py::hasattr(_outcome, "value")) {
                throw std::invalid_argument("AIRLAPS exception: python transition outcome object must provide 'value'");
            }
            if (!py::hasattr(_outcome, "termination")) {
                throw std::invalid_argument("AIRLAPS exception: python transition outcome object must provide 'termination'");
            }
            if (!py::hasattr(_outcome, "info")) {
                throw std::invalid_argument("AIRLAPS exception: python transition outcome object must provide 'info'");
            }
        }

        TransitionOutcome(const TransitionOutcome& other) {
            typename GilControl<Texecution>::Acquire acquire;
            _outcome = py::module::import("copy").attr("deepcopy")(other._outcome);
            _state = other._state;
        }

        TransitionOutcome& operator= (const TransitionOutcome& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_outcome = py::module::import("copy").attr("deepcopy")(other._outcome);
            this->_state = other._state;
            return *this;
        }

        ~TransitionOutcome() {
            typename GilControl<Texecution>::Acquire acquire;
            _outcome = py::object();
            _state = py::object();
        }

        py::object state() {
            return _state;
        }

        py::object observation() {
            return state();
        }

        void state(const py::object& s) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                if (py::hasattr(_outcome, "state")) {
                    _outcome.attr("state") = s;
                } else {
                    _outcome.attr("observation") = s;
                }
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("AIRLAPS exception when setting outcome's state: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void observation(const py::object& o) {
            state(o);
        }

        double cost() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return py::cast<double>(_outcome.attr("value").attr("cost"));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("AIRLAPS exception when getting outcome's cost: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void cost(double c) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                _outcome.attr("value").attr("cost") = py::float_(c);
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("AIRLAPS exception when setting outcome's cost: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        double reward() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return py::cast<double>(_outcome.attr("value").attr("reward"));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("AIRLAPS exception when getting outcome's reward: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void reward(double r) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                _outcome.attr("value").attr("reward") = py::float_(r);
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("AIRLAPS exception when setting outcome's reward: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        bool terminal() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return py::cast<bool>(_outcome.attr("termination"));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("AIRLAPS exception when getting outcome's state: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void termination(bool t) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                _outcome.attr("termination") = py::bool_(t);
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("AIRLAPS exception when setting outcome's observation: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        py::object info() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return _outcome.attr("info");
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("AIRLAPS exception when getting outcome's info: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void info(const py::object& i) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                _outcome.attr("info") = i;
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("AIRLAPS exception when setting outcome's info: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        static std::size_t get_depth(const py::object& info) {
            typename GilControl<Texecution>::Acquire acquire;
            if (py::hasattr(info, "depth")) {
                return py::cast<std::size_t>(info.attr("depth")());
            } else {
                return 0;
            }
        }
    };

    typedef TransitionOutcome EnvironmentOutcome;

    struct NextStateDistribution {
        py::object _next_state_distribution;

        NextStateDistribution(const py::object& next_state_distribution) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_next_state_distribution = next_state_distribution;
            if (!py::hasattr(_next_state_distribution, "get_values")) {
                throw std::invalid_argument("AIRLAPS exception: python next state distribution object must implement get_values()");
            }
        }

        NextStateDistribution(const NextStateDistribution& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_next_state_distribution = other._next_state_distribution;
        }

         NextStateDistribution& operator=(const NextStateDistribution& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_next_state_distribution = other._next_state_distribution;
            return *this;
        }

        ~NextStateDistribution() {
            typename GilControl<Texecution>::Acquire acquire;
            _next_state_distribution = py::object();
        }

        struct NextStateDistributionValues {
            py::object _values;

            NextStateDistributionValues(const py::object& values) {
                typename GilControl<Texecution>::Acquire acquire;
                this->_values = values;
            }

            NextStateDistributionValues(const NextStateDistributionValues& other) {
                typename GilControl<Texecution>::Acquire acquire;
                this->_values = other._values;
            }

            NextStateDistributionValues& operator=(const NextStateDistributionValues& other) {
                typename GilControl<Texecution>::Acquire acquire;
                this->_values = other._values;
                return *this;
            }

            ~NextStateDistributionValues() {
                typename GilControl<Texecution>::Acquire acquire;
                _values = py::object();
            }

            py::iterator begin() const {
                typename GilControl<Texecution>::Acquire acquire;
                return _values.begin();
            }

            py::iterator end() const {
                typename GilControl<Texecution>::Acquire acquire;
                return _values.end();
            }
        };

        NextStateDistributionValues get_values() const {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return NextStateDistributionValues(_next_state_distribution.attr("get_values")());
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("AIRLAPS exception when getting next state's distribution values: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }
    };

    struct OutcomeExtractor {
        py::object _state;
        double _probability;

        OutcomeExtractor(const py::handle& o) {
            typename GilControl<Texecution>::Acquire acquire;
            if (!py::isinstance<py::tuple>(o)) {
                throw std::invalid_argument("AIRLAPS exception: python next state distribution returned value should be an iterable over tuple objects");
            }
            py::tuple t = o.cast<py::tuple>();
            _state = t[0];
            _probability = t[1].cast<double>();
        }

        OutcomeExtractor(const OutcomeExtractor& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->state = other._state;
            this->probability = other._probability;
        }

        OutcomeExtractor& operator=(const OutcomeExtractor& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->state = other._state;
            this->probability = other._probability;
            return *this;
        }

        ~OutcomeExtractor() {
            typename GilControl<Texecution>::Acquire acquire;
            _state = py::object();
        }

        const py::object& state() const { return _state; }
        const double& probability() const { return _probability; }
    };

    PythonDomainAdapter(const py::object& domain) {
        _implementation = std::make_unique<Implementation<Texecution>>(domain);
    }

    std::unique_ptr<ApplicableActionSpace> get_applicable_actions(const State& s) {
        return _implementation->get_applicable_actions(s);
    }

    std::unique_ptr<State> reset() {
        return _implementation->reset();
    }

    std::unique_ptr<TransitionOutcome> step(const Event& e) {
        return _implementation->step(e);
    }

    std::unique_ptr<TransitionOutcome> sample(const State& s, const Event& e) {
        return _implementation->sample(s, e);
    }

    void compute_next_state(const State& s, const Event& e) {
        throw std::runtime_error("DEPRECATED");
    }

    py::object get_next_state(const State& s, const Event& e) {
        return _implementation->get_next_state(s, e);
    }

    void compute_next_state_distribution(const State& s, const Event& e) {
        throw std::runtime_error("DEPRECATED");
    }

    std::unique_ptr<NextStateDistribution> get_next_state_distribution(const State& s, const Event& e) {
        return _implementation->get_next_state_distribution(s, e);
    }

    double get_transition_cost(const State& s, const Event& e, const State& sp) {
        return _implementation->get_transition_cost(s, e, sp);
    }

    double get_transition_reward(const State& s, const Event& e, const State& sp) {
        return _implementation->get_transition_reward(s, e, sp);
    }

    bool is_goal(const State& s) {
        return _implementation->is_goal(s);
    }

    // Used only if the domain provides is_goal, which is not the case of simulation of environment
    // domains that are mosts expected for RIW (but sometimes the domain can b a planning domain)
    bool is_optional_goal(const State& s) {
        return _implementation->is_optional_goal(s);
    }

    bool is_terminal(const State& s) {
        return _implementation->is_terminal(s);
    }

protected :

    template <typename TexecutionPolicy, typename Enable = void>
    struct Implementation;

    template <typename TexecutionPolicy>
    struct Implementation<TexecutionPolicy,
                          typename std::enable_if<std::is_same<TexecutionPolicy, SequentialExecution>::value>::type> {
        Implementation(const py::object& domain)
        : _domain(domain) {}

        std::unique_ptr<ApplicableActionSpace> get_applicable_actions(const State& s) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return std::make_unique<ApplicableActionSpace>(_domain.attr("get_applicable_actions")(s._state));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("AIRLAPS exception when getting applicable actions in state ") + s.print() + ": " + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        std::unique_ptr<State> reset() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return std::make_unique<State>(_domain.attr("reset")());
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("AIRLAPS exception when resetting the domain: ") + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        std::unique_ptr<TransitionOutcome> step(const Event& e) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return std::make_unique<TransitionOutcome>(_domain.attr("step")(e._event));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("AIRLAPS exception when stepping with action ") +
                            e.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        std::unique_ptr<TransitionOutcome> sample(const State& s, const Event& e) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return std::make_unique<TransitionOutcome>(_domain.attr("sample")(s._state, e._event));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("AIRLAPS exception when sampling from state ") + s.print() +
                              " with action " + e.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        py::object get_next_state(const State& s, const Event& e) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return _domain.attr("get_next_state")(s._state, e._event);
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("AIRLAPS exception when getting next state from state ") +
                              s.print() + " and applying action " + e.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        std::unique_ptr<NextStateDistribution> get_next_state_distribution(const State& s, const Event& e) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return std::make_unique<NextStateDistribution>(_domain.attr("get_next_state_distribution")(s._state, e._event));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("AIRLAPS exception when getting next state distribution from state ") +
                              s.print() + " and applying action " + e.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        double get_transition_cost(const State& s, const Event& e, const State& sp) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return py::cast<double>(_domain.attr("get_transition_value")(s._state, e._event, sp._state).attr("cost"));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("AIRLAPS exception when getting value of transition (") +
                            s.print() + ", " + e.print() + ") -> " + sp.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        double get_transition_reward(const State& s, const Event& e, const State& sp) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return py::cast<double>(_domain.attr("get_transition_value")(s._state, e._event, sp._state).attr("reward"));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("AIRLAPS exception when getting value of transition (") +
                            s.print() + ", " + e.print() + ") -> " + sp.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        bool is_goal(const State& s) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return py::cast<bool>(_domain.attr("is_goal")(s._state));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("AIRLAPS exception when testing goal condition of state ") +
                            s.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        // Used only if the domain provides is_goal, which is not the case of simulation of environment
        // domains that are mosts expected for RIW (but sometimes the domain can b a planning domain)
        bool is_optional_goal(const State& s) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                if (py::hasattr(_domain, "is_goal")) {
                    return py::cast<bool>(_domain.attr("is_goal")(s._state));
                } else {
                    return false;
                }
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("AIRLAPS exception when testing goal condition of state ") +
                            s.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        bool is_terminal(const State& s) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return py::cast<bool>(_domain.attr("is_terminal")(s._state));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("AIRLAPS exception when testing terminal condition of state ") +
                            s.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        py::object _domain;
    };

    template <typename TexecutionPolicy>
    struct Implementation<TexecutionPolicy,
                          typename std::enable_if<std::is_same<TexecutionPolicy, ParallelExecution>::value>::type> {
        static constexpr std::chrono::milliseconds PERIOD = std::chrono::milliseconds(10);

        Implementation(const py::object& domain) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_domain = domain;
        }

        Implementation(const Implementation& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_domain = other._domain;
        }

        Implementation& operator=(const Implementation& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_domain = other._domain;
            return *this;
        }

        ~Implementation() {
            typename GilControl<Texecution>::Acquire acquire;
            _domain = py::object();
        }

        template <typename ... Types>
        py::object launch(const char* name, Types ... args) {
            py::object id;
            {
                typename GilControl<Texecution>::Acquire acquire;
                try {
                    id = _domain.attr(name)(args...);
                } catch(py::error_already_set* e) {
                    std::runtime_error err(e->what());
                    id = py::object();
                    delete e;
                    throw err;
                }
            }
            while (true) {
                std::this_thread::sleep_for(PERIOD);
                typename GilControl<Texecution>::Acquire acquire;
                try {
                    py::object r = _domain.attr("get_result")(id);
                    if (!r.is_none()) {
                        return r;
                    }
                } catch(const py::error_already_set* e) {
                    std::runtime_error err(e->what());
                    id = py::object();
                    delete e;
                    throw err;
                }
            }
            typename GilControl<Texecution>::Acquire acquire;
            return py::none();
        }

        std::unique_ptr<ApplicableActionSpace> get_applicable_actions(const State& s) {
            try {
                return std::make_unique<ApplicableActionSpace>(launch("get_applicable_actions", s._state));
            } catch(const std::exception& e) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("AIRLAPS exception when getting applicable actions in state ") + s.print() + ": " + e.what());
                throw;
            }
        }

        std::unique_ptr<State> reset() {
            try {
                return std::make_unique<State>(launch("reset"));
            } catch(const std::exception& e) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("AIRLAPS exception when resetting the domain: ") + e.what());
                throw;
            }
        }

        std::unique_ptr<TransitionOutcome> step(const Event& e) {
            try {
                return std::make_unique<TransitionOutcome>(launch("step", e._event));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("AIRLAPS exception when stepping with action ") +
                              e.print() + ": " + ex.what());
                throw;
            }
        }

        std::unique_ptr<TransitionOutcome> sample(const State& s, const Event& e) {
            try {
                return std::make_unique<TransitionOutcome>(launch("sample", s._state, e._event));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("AIRLAPS exception when sampling from state ") + s.print() +
                            " with action " + e.print() + ": " + ex.what());
                throw;
            }
        }

        py::object get_next_state(const State& s, const Event& e) {
            try {
                return launch("get_next_state", s._state, e._event);
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("AIRLAPS exception when getting next state from state ") +
                            s.print() + " and applying action " + e.print() + ": " + ex.what());
                throw;
            }
        }

        std::unique_ptr<NextStateDistribution> get_next_state_distribution(const State& s, const Event& e) {
            try {
                return std::make_unique<NextStateDistribution>(launch("get_next_state_distribution", s._state, e._event));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("AIRLAPS exception when getting next state distribution from state ") +
                            s.print() + " and applying action " + e.print() + ": " + ex.what());
                throw;
            }
        }

        double get_transition_cost(const State& s, const Event& e, const State& sp) {
            try {
                py::object r = launch("get_transition_value", s._state, e._event, sp._state);
                typename GilControl<Texecution>::Acquire acquire;
                return py::cast<double>(r.attr("cost"));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("AIRLAPS exception when getting value of transition (") +
                              s.print() + ", " + e.print() + ") -> " + sp.print() + ": " + ex.what());
                throw;
            }
        }

        double get_transition_reward(const State& s, const Event& e, const State& sp) {
            try {
                py::object r = launch("get_transition_value", s._state, e._event, sp._state);
                typename GilControl<Texecution>::Acquire acquire;
                return py::cast<double>(r.attr("reward"));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("AIRLAPS exception when getting value of transition (") +
                              s.print() + ", " + e.print() + ") -> " + sp.print() + ": " + ex.what());
                throw;
            }
        }

        bool is_goal(const State& s) {
            try {
                py::object r = launch("is_goal", s._state);
                typename GilControl<Texecution>::Acquire acquire;
                return py::cast<bool>(r);
            } catch(const std::exception& e) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("AIRLAPS exception when testing goal condition of state ") +
                              s.print() + ": " + e.what());
                throw;
            }
        }

        bool is_optional_goal(const State& s) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                bool has_goal;
                {
                    typename GilControl<Texecution>::Acquire acquire;
                    has_goal = py::hasattr(_domain, "is_goal");
                }
                if (has_goal) {
                    py::object r = launch("is_goal", s._state);
                    typename GilControl<Texecution>::Acquire acquire;
                    return py::cast<bool>(r);
                } else {
                    return false;
                }
            } catch(py::error_already_set* e) {
                    std::runtime_error err(e->what());
                    delete e;
                    throw err;
            } catch(const std::exception& e) {
                spdlog::error(std::string("AIRLAPS exception when testing goal condition of state ") +
                              s.print() + ": " + e.what());
                throw;
            }
        }

        bool is_terminal(const State& s) {
            try {
                py::object r = launch("is_terminal", s._state);
                typename GilControl<Texecution>::Acquire acquire;
                return py::cast<bool>(r);
            } catch(const std::exception& e) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("AIRLAPS exception when testing terminal condition of state ") +
                              s.print() + ": " + e.what());
                throw;
            }
        }

        py::object _domain;
    };

    std::unique_ptr<Implementation<Texecution>> _implementation;
};

} // namespace airlaps

#endif // AIRLAPS_PYTHON_DOMAIN_ADAPTER_HH
