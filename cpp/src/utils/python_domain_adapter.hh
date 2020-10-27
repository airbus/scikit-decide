/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_DOMAIN_ADAPTER_HH
#define SKDECIDE_PYTHON_DOMAIN_ADAPTER_HH

#include <pybind11/pybind11.h>

#include <nngpp/nngpp.h>
#include <nngpp/protocol/pull0.h>

#include "utils/python_gil_control.hh"
#include "utils/python_hash_eq.hh"
#include "utils/execution.hh"

namespace py = pybind11;

namespace skdecide {

struct SingleAgent { SingleAgent() {} };
struct MultiAgent { MultiAgent() {} };

struct PartiallyObservable { PartiallyObservable() {} };
struct FullyObservable { FullyObservable() {} };

template <typename Texecution,
          typename Tagent = SingleAgent,
          typename Tobservability = FullyObservable>
class PythonDomainAdapter {
public :
    template <typename Derived>
    struct PyObj {
        std::unique_ptr<py::object> _pyobj;

        PyObj() {
            typename GilControl<Texecution>::Acquire acquire;
            _pyobj = std::make_unique<py::object>();
        }

        PyObj(std::unique_ptr<py::object>&& o) : _pyobj(std::move(o)) {}

        PyObj(const py::object& o) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_pyobj = std::make_unique<py::object>(o);
        }

        PyObj(const PyObj& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_pyobj = std::make_unique<py::object>(*other._pyobj);
        }

        PyObj& operator=(const PyObj& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_pyobj = std::make_unique<py::object>(*other._pyobj);
            return *this;
        }

        virtual ~PyObj() {
            typename GilControl<Texecution>::Acquire acquire;
            _pyobj.reset();
        }

        const py::object& pyobj() const { return *_pyobj; }

        std::string print() const {
            typename GilControl<Texecution>::Acquire acquire;
            return py::str(*_pyobj);
        }

        struct Hash {
            std::size_t operator()(const PyObj<Derived>& o) const {
                try {
                    return skdecide::PythonHash<Texecution>()(*o._pyobj);
                } catch(const std::exception& e) {
                    spdlog::error(std::string("SKDECIDE exception when hashing ") +
                                  Derived::class_name + "s: " + e.what());
                    throw;
                }
            }
        };

        struct Equal {
            bool operator()(const PyObj<Derived>& o1, const PyObj<Derived>& o2) const {
                try {
                    return skdecide::PythonEqual<Texecution>()(*o1._pyobj, *o2._pyobj);
                } catch(const std::exception& e) {
                    spdlog::error(std::string("SKDECIDE exception when testing ") +
                                  Derived::class_name + "s equality: " + e.what());
                    throw;
                }
            }
        };
    };

    template<typename T>
    struct PyIter {
        std::unique_ptr<py::iterator> _iterator;

        PyIter(std::unique_ptr<py::iterator>&& iterator) : _iterator(std::move(iterator)) {}

        PyIter(const py::iterator& iterator) {
            typename GilControl<Texecution>::Acquire acquire;
            _iterator = std::make_unique<py::iterator>(iterator);
        }

        PyIter(const PyIter& other) {
            typename GilControl<Texecution>::Acquire acquire;
            _iterator = std::make_unique<py::iterator>(*other._iterator);
        }

        PyIter& operator=(const PyIter& other) {
            typename GilControl<Texecution>::Acquire acquire;
            _iterator = std::make_unique<py::iterator>(*other._iterator);
            return *this;
        }

        ~PyIter() {
            typename GilControl<Texecution>::Acquire acquire;
            this->_iterator.reset();
        }

        PyIter<T>& operator++() {
            typename GilControl<Texecution>::Acquire acquire;
            ++(*(this->_iterator));
            return *this;
        }

        PyIter<T> operator++(int) {
            typename GilControl<Texecution>::Acquire acquire;
            py::iterator rv = (*(this->_iterator))++;
            return PyIter<T>(rv);
        }

        T operator*() const {
            typename GilControl<Texecution>::Acquire acquire;
            return T(py::reinterpret_borrow<py::object>(**(this->_iterator)));
        }

        std::unique_ptr<T> operator->() const {
            typename GilControl<Texecution>::Acquire acquire;
            return std::make_unique<T>(py::reinterpret_borrow<py::object>(**(this->_iterator)));
        }

        bool operator==(const PyIter<T>& other) const {
            typename GilControl<Texecution>::Acquire acquire;
            return *(this->_iterator) == *(other._iterator);
        }

        bool operator!=(const PyIter<T>& other) const {
            typename GilControl<Texecution>::Acquire acquire;
            return *(this->_iterator) != *(other._iterator);
        }
    };

    template <typename Inherited, typename TTagent, typename Enable = void>
    struct AgentData {};

    template <typename Inherited, typename TTagent>
    struct AgentData<Inherited, TTagent,
                     typename std::enable_if<std::is_same<TTagent, SingleAgent>::value>::type> : public Inherited {
        AgentData() : Inherited() {}
        AgentData(std::unique_ptr<py::object>&& s) : Inherited(std::move(s)) {}
        AgentData(const py::object& s) : Inherited(s) {}
        AgentData(const AgentData& other) : Inherited(other) {}
        AgentData& operator=(const AgentData& other) { dynamic_cast<Inherited&>(*this) = other; return *this; }
        virtual ~AgentData() {}
    };

    template <typename Inherited, typename TTagent>
    struct AgentData<Inherited, TTagent,
                     typename std::enable_if<std::is_same<TTagent, MultiAgent>::value>::type> : public PyObj<AgentData<Inherited, TTagent>> {
        static constexpr char class_name[] = Inherited::class_name;
        AgentData() : PyObj<AgentData<Inherited, TTagent>>() {}
        AgentData(std::unique_ptr<py::object>&& s) : PyObj<AgentData<Inherited, TTagent>>(std::move(s)) {}
        AgentData(const py::object& s) : PyObj<AgentData<Inherited, TTagent>>(s) {}
        AgentData(const AgentData& other) : PyObj<AgentData<Inherited, TTagent>>(other) {}
        AgentData& operator=(const AgentData& other) { dynamic_cast<PyObj<AgentData<Inherited, TTagent>>&>(*this) = other; return *this; }
        virtual ~AgentData() {}

        struct Element : public Inherited {
            static constexpr char class_name[] = std::string("agent ") + Inherited::class_name;
            Element() : Inherited() {}
            Element(std::unique_ptr<py::object>&& s) : Inherited(std::move(s)) {}
            Element(const py::object& s) : Inherited(s) {}
            Element(const Element& other) : Inherited(other) {}
            Element& operator=(const Element& other) { dynamic_cast<PyObj<Inherited>&>(*this) = other; return *this; }
            virtual ~Element() {}
        };

        PyIter<Element> begin() const {
            typename GilControl<Texecution>::Acquire acquire;
            return PyIter<Element>(this->_pyobj->begin());
        }

        PyIter<Element> end() const {
            typename GilControl<Texecution>::Acquire acquire;
            return PyIter<Element>(this->_pyobj->end());
        }
    };

    struct StateBase : public PyObj<StateBase> {
        static constexpr char class_name[] = "state";
        StateBase() : PyObj<StateBase>() {}
        StateBase(std::unique_ptr<py::object>&& s) : PyObj<StateBase>(std::move(s)) {}
        StateBase(const py::object& s) : PyObj<StateBase>(s) {}
        StateBase(const StateBase& other) : PyObj<StateBase>(other) {}
        StateBase& operator=(const StateBase& other) { dynamic_cast<PyObj<StateBase>&>(*this) = other; return *this; }
        virtual ~StateBase() {}
    };
    typedef AgentData<StateBase, Tagent> State;

    struct ObservationBase : public PyObj<ObservationBase> {
        static constexpr char class_name[] = "observation";
        ObservationBase() : PyObj<ObservationBase>() {}
        ObservationBase(std::unique_ptr<py::object>&& s) : PyObj<ObservationBase>(std::move(s)) {}
        ObservationBase(const py::object& s) : PyObj<ObservationBase>(s) {}
        ObservationBase(const ObservationBase& other) : PyObj<ObservationBase>(other) {}
        ObservationBase& operator=(const ObservationBase& other) { dynamic_cast<PyObj<ObservationBase>&>(*this) = other; return *this; }
        virtual ~ObservationBase() {}
    };
    
    typedef typename std::conditional<std::is_same<Tobservability, FullyObservable>::value,
                                        State,
                                        AgentData<ObservationBase, Tagent>
                                     >::type Observation;

    struct EventBase : public PyObj<EventBase> {
        static constexpr char class_name[] = "event";
        EventBase() : PyObj<EventBase>() {}
        EventBase(std::unique_ptr<py::object>&& s) : PyObj<EventBase>(std::move(s)) {}
        EventBase(const py::object& s) : PyObj<EventBase>(s) {}
        EventBase(const EventBase& other) : PyObj<EventBase>(other) {}
        EventBase& operator=(const EventBase& other) { dynamic_cast<PyObj<EventBase>&>(*this) = other; return *this; }
        virtual ~EventBase() {}
    };
    typedef AgentData<EventBase, Tagent> Event;

    struct ActionBase : public PyObj<ActionBase> {
        static constexpr char class_name[] = "action";
        ActionBase() : PyObj<ActionBase>() {}
        ActionBase(std::unique_ptr<py::object>&& s) : PyObj<ActionBase>(std::move(s)) {}
        ActionBase(const py::object& s) : PyObj<ActionBase>(s) {}
        ActionBase(const ActionBase& other) : PyObj<ActionBase>(other) {}
        ActionBase& operator=(const ActionBase& other) { dynamic_cast<PyObj<ActionBase>&>(*this) = other; return *this; }
        virtual ~ActionBase() {}
    };
    typedef AgentData<ActionBase, Tagent> Action;

    struct ApplicableActionSpace : public PyObj<ApplicableActionSpace> { // don't inherit from skdecide::EnumerableSpace since otherwise we would need to copy the applicable action python object into a c++ iterable object
        static constexpr char class_name[] = "applicable action space";

        ApplicableActionSpace() : PyObj<ApplicableActionSpace>() {}

        ApplicableActionSpace(std::unique_ptr<py::object>&& applicable_action_space)
        : PyObj<ApplicableActionSpace>(std::move(applicable_action_space)) {
            construct();
        }
        
        ApplicableActionSpace(const py::object& applicable_action_space)
        : PyObj<ApplicableActionSpace>(applicable_action_space) {
            construct();
        }
        
        void construct() {
            typename GilControl<Texecution>::Acquire acquire;
            if (!py::hasattr(*(this->_pyobj), "get_elements")) {
                throw std::invalid_argument("SKDECIDE exception: python applicable action object must implement get_elements()");
            }
        }

        ApplicableActionSpace(const ApplicableActionSpace& other)
        : PyObj<ApplicableActionSpace>(other) {}

        ApplicableActionSpace& operator=(const ApplicableActionSpace& other) {
            dynamic_cast<PyObj<ApplicableActionSpace>&>(*this) = other;
            return *this;
        }

        virtual ~ApplicableActionSpace() {}

        struct ApplicableActionSpaceElements : public PyObj<ApplicableActionSpaceElements> {
            static constexpr char class_name[] = "applicable action space elements";

            ApplicableActionSpaceElements() : PyObj<ApplicableActionSpaceElements>() {}

            ApplicableActionSpaceElements(std::unique_ptr<py::object>&& applicable_action_space_elements)
                : PyObj<ApplicableActionSpaceElements>(std::move(applicable_action_space_elements)) {}
            
            ApplicableActionSpaceElements(const py::object& applicable_action_space_elements)
                : PyObj<ApplicableActionSpaceElements>(applicable_action_space_elements) { }
            
            ApplicableActionSpaceElements(const ApplicableActionSpaceElements& other)
            : PyObj<ApplicableActionSpaceElements>(other) {}

            ApplicableActionSpaceElements& operator=(const ApplicableActionSpaceElements& other) {
                dynamic_cast<PyObj<ApplicableActionSpaceElements>&>(*this) = other;
                return *this;
            }

            virtual ~ApplicableActionSpaceElements() {}

            PyIter<Action> begin() const {
                typename GilControl<Texecution>::Acquire acquire;
                return PyIter<Action>(this->_pyobj->begin());
            }

            PyIter<Action> end() const {
                typename GilControl<Texecution>::Acquire acquire;
                return PyIter<Action>(this->_pyobj->end());
            }
        };

        ApplicableActionSpaceElements get_elements() const {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return ApplicableActionSpaceElements(this->_pyobj->attr("get_elements")());
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting applicable action space's elements: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        Action sample() const {
            typename GilControl<Texecution>::Acquire acquire;
            if (!py::hasattr(*(this->_pyobj), "sample")) {
                throw std::invalid_argument("SKDECIDE exception: python applicable action object must implement sample()");
            } else {
                return Action(this->_pyobj->attr("sample")());
            }
        }
    };

    struct TransitionValueBase : public PyObj<TransitionValueBase> {
        static constexpr char class_name[] = "transition value";
        TransitionValueBase() : PyObj<TransitionValueBase>() {}
        TransitionValueBase(std::unique_ptr<py::object>&& e) : PyObj<TransitionValueBase>(std::move(e)) { construct(); }
        TransitionValueBase(const py::object& e) : PyObj<TransitionValueBase>(e) { construct(); }
        TransitionValueBase(const TransitionValueBase& other) : PyObj<TransitionValueBase>(other) {}
        TransitionValueBase& operator=(const TransitionValueBase& other) { dynamic_cast<PyObj<TransitionValueBase>&>(*this) = other; return *this; }
        virtual ~TransitionValueBase() {}

        void construct() {
            typename GilControl<Texecution>::Acquire acquire;
            if (!py::hasattr(*(this->_pyobj), "cost")) {
                throw std::invalid_argument("SKDECIDE exception: python transition value object must implement cost()");
            }
            if (!py::hasattr(*(this->_pyobj), "reward")) {
                throw std::invalid_argument("SKDECIDE exception: python transition value object must implement reward()");
            }
        }

        double cost() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return py::cast<double>(this->_pyobj->attr("cost"));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting transition value's cost: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        double reward() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return py::cast<double>(this->_pyobj->attr("reward"));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting transition value's reward: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }
    };
    typedef AgentData<TransitionValueBase, Tagent> TransitionValue;

    template <typename Derived, typename Situation>
    struct Outcome : public PyObj<Derived> {
        struct InfoBase : PyObj<InfoBase> {
            static constexpr char class_name[] = "info";
            InfoBase() : PyObj<InfoBase>() {}
            InfoBase(std::unique_ptr<py::object>&& s) : PyObj<InfoBase>(std::move(s)) {}
            InfoBase(const py::object& s) : PyObj<InfoBase>(s) {}
            InfoBase(const InfoBase& other) : PyObj<InfoBase>(other) {}
            InfoBase& operator=(const InfoBase& other) { dynamic_cast<PyObj<InfoBase>&>(*this) = other; return *this; }
            virtual ~InfoBase() {}

            std::size_t get_depth() {
                typename GilControl<Texecution>::Acquire acquire;
                if (py::hasattr(*(this->_pyobj), "depth")) {
                    return py::cast<std::size_t>(this->_pyobj->attr("depth")());
                } else {
                    return 0;
                }
            }
        };
        typedef AgentData<InfoBase, Tagent> Info;

        Outcome() : PyObj<Derived>() {}

        Outcome(std::unique_ptr<py::object>&& outcome)
        : PyObj<Derived>(std::move(outcome)) {
            construct();
        }

        Outcome(const py::object& outcome)
        : PyObj<Derived>(outcome) {
            construct();
        }

        void construct() {
            typename GilControl<Texecution>::Acquire acquire;
            if (!py::hasattr(*(this->_pyobj), Situation::class_name)) {
                throw std::invalid_argument(std::string("SKDECIDE exception: python transition outcome object must provide '") +
                                            Situation::class_name + "'");
            }
            if (!py::hasattr(*(this->_pyobj), "value")) {
                throw std::invalid_argument("SKDECIDE exception: python transition outcome object must provide 'value'");
            }
            if (!py::hasattr(*(this->_pyobj), "termination")) {
                throw std::invalid_argument("SKDECIDE exception: python transition outcome object must provide 'termination'");
            }
            if (!py::hasattr(*(this->_pyobj), "info")) {
                throw std::invalid_argument("SKDECIDE exception: python transition outcome object must provide 'info'");
            }
        }

        Outcome(const Outcome& other)
        : PyObj<Derived>(other) {}

        Outcome& operator=(const Outcome& other) {
            dynamic_cast<PyObj<Derived>&>(*this) = other;
            return *this;
        }

        virtual ~Outcome() {}

        Situation situation() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return Situation(this->_pyobj->attr(Situation::class_name));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting outcome's ") +
                              Situation::class_name + ": " + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void situation(const Situation& s) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                this->_pyobj->attr(Situation::class_name) = s.pyobj();
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when setting outcome's ") +
                              Situation::class_name + ": " + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        TransitionValue transition_value() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return TransitionValue(this->_pyobj->attr("value"));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting outcome's transition value: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void transition_value(const TransitionValue& tv) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                this->_pyobj->attr("value") = tv.pyobj();
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when setting outcome's cost: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        bool termination() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return py::cast<bool>(this->_pyobj->attr("termination"));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting outcome's state: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void termination(bool t) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                this->_pyobj->attr("termination") = py::bool_(t);
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when setting outcome's observation: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        Info info() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return Info(this->_pyobj.attr("info"));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting outcome's info: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void info(const Info& i) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                this->_pyobj->attr("info") = i.pyobj();
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when setting outcome's info: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }
    };

    struct TransitionOutcome : public Outcome<TransitionOutcome, State> {
        static constexpr char class_name[] = "transition outcome";

        TransitionOutcome() : Outcome<TransitionOutcome, State>() {}

        TransitionOutcome(std::unique_ptr<py::object>&& outcome)
        : Outcome<TransitionOutcome, State>(std::move(outcome)) {}

        TransitionOutcome(const py::object& outcome)
        : Outcome<TransitionOutcome, State>(outcome) {}

        TransitionOutcome(const Outcome<TransitionOutcome, State>& other)
        : Outcome<TransitionOutcome, State>(other) {}

        TransitionOutcome& operator=(const TransitionOutcome& other) {
            dynamic_cast<Outcome<TransitionOutcome, State>&>(*this) = other;
            return *this;
        }

        virtual ~TransitionOutcome() {}

        State state() { return this->situation(); }
        void state(const State& s) { this->situation(s); }
    };

    struct EnvironmentOutcome : public Outcome<EnvironmentOutcome, Observation> {
        static constexpr char class_name[] = "environment outcome";

        EnvironmentOutcome() : Outcome<EnvironmentOutcome, Observation>() {}

        EnvironmentOutcome(std::unique_ptr<py::object>&& outcome)
        : Outcome<EnvironmentOutcome, Observation>(std::move(outcome)) {}

        EnvironmentOutcome(const py::object& outcome)
        : Outcome<EnvironmentOutcome, Observation>(outcome) {}

        EnvironmentOutcome(const Outcome<EnvironmentOutcome, Observation>& other)
        : Outcome<EnvironmentOutcome, Observation>(other) {}

        EnvironmentOutcome& operator=(const EnvironmentOutcome& other) {
            dynamic_cast<Outcome<EnvironmentOutcome, Observation>&>(*this) = other;
            return *this;
        }

        virtual ~EnvironmentOutcome() {}

        Observation observation() { return this->situation(); }
        void observation(const Observation& s) { this->situation(s); }
    };

    struct NextStateDistribution : public PyObj<NextStateDistribution> {
        static constexpr char class_name[] = "next state distribution";

        NextStateDistribution() : PyObj<NextStateDistribution>() {}

        NextStateDistribution(std::unique_ptr<py::object>&& next_state_distribution)
        : PyObj<NextStateDistribution>(std::move(next_state_distribution)) {
            construct();
        }

        NextStateDistribution(const py::object& next_state_distribution)
        : PyObj<NextStateDistribution>(next_state_distribution) {
            construct();
        }

        void construct() {
            typename GilControl<Texecution>::Acquire acquire;
            if (!py::hasattr(*(this->_pyobj), "get_values")) {
                throw std::invalid_argument("SKDECIDE exception: python next state distribution object must implement get_values()");
            }
        }

        NextStateDistribution(const NextStateDistribution& other)
        : PyObj<NextStateDistribution>(other) {}

        NextStateDistribution& operator=(const NextStateDistribution& other) {
            dynamic_cast<PyObj<NextStateDistribution>&>(*this) = other;
            return *this;
        }

        virtual ~NextStateDistribution() {}

        struct DistributionValue {
            static constexpr char class_name[] = "distribution value";
            State _state;
            double _probability;

            DistributionValue() {}

            DistributionValue(const py::object& o) {
                typename GilControl<Texecution>::Acquire acquire;
                if (!py::isinstance<py::tuple>(o)) {
                    throw std::invalid_argument("SKDECIDE exception: python next state distribution returned value should be an iterable over tuple objects");
                }
                py::tuple t = o.cast<py::tuple>();
                _state = State(t[0]);
                _probability = t[1].cast<double>();
            }

            DistributionValue(const DistributionValue& other) {
                this->_state = other._state;
                this->_probability = other._probability;
            }

            DistributionValue& operator=(const DistributionValue& other) {
                this->_state = other._state;
                this->_probability = other._probability;
                return *this;
            }

            const State& state() const { return _state; }
            const double& probability() const { return _probability; }
        };

        struct NextStateDistributionValues : public PyObj<NextStateDistributionValues> {
            static constexpr char class_name[] = "next state distribution values";

            NextStateDistributionValues() : PyObj<NextStateDistributionValues>() {}

            NextStateDistributionValues(std::unique_ptr<py::object>&& next_state_distribution)
                : PyObj<NextStateDistributionValues>(std::move(next_state_distribution)) {}
            
            NextStateDistributionValues(const py::object& next_state_distribution)
                : PyObj<NextStateDistributionValues>(next_state_distribution) {}
            
            NextStateDistributionValues(const NextStateDistributionValues& other)
            : PyObj<NextStateDistributionValues>(other) {}

            NextStateDistributionValues& operator=(const NextStateDistributionValues& other) {
                dynamic_cast<PyObj<NextStateDistributionValues>&>(*this) = other;
                return *this;
            }

            virtual ~NextStateDistributionValues() {}

            PyIter<DistributionValue> begin() const {
                typename GilControl<Texecution>::Acquire acquire;
                return PyIter<DistributionValue>(this->_pyobj->begin());
            }

            PyIter<DistributionValue> end() const {
                typename GilControl<Texecution>::Acquire acquire;
                return PyIter<DistributionValue>(this->_pyobj->end());
            }
        };

        NextStateDistributionValues get_values() const {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return NextStateDistributionValues(this->_pyobj->attr("get_values")());
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting next state's distribution values: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }
    };

    PythonDomainAdapter(const py::object& domain) {
        _implementation = std::make_unique<Implementation<Texecution>>(domain);
    }

    std::size_t get_parallel_capacity() {
        return _implementation->get_parallel_capacity();
    }

    ApplicableActionSpace get_applicable_actions(const State& s, const std::size_t* thread_id = nullptr) {
        return _implementation->get_applicable_actions(s, thread_id);
    }

    State reset(const std::size_t* thread_id = nullptr) {
        return _implementation->reset(thread_id);
    }

    EnvironmentOutcome step(const Action& a, const std::size_t* thread_id = nullptr) {
        return _implementation->step(a, thread_id);
    }

    EnvironmentOutcome sample(const State& s, const Action& a, const std::size_t* thread_id = nullptr) {
        return _implementation->sample(s, a, thread_id);
    }

    State get_next_state(const State& s, const Action& a, const std::size_t* thread_id = nullptr) {
        return _implementation->get_next_state(s, a, thread_id);
    }

    NextStateDistribution get_next_state_distribution(const State& s, const Action& a, const std::size_t* thread_id = nullptr) {
        return _implementation->get_next_state_distribution(s, a, thread_id);
    }

    TransitionValue get_transition_value(const State& s, const Action& a, const State& sp, const std::size_t* thread_id = nullptr) {
        return _implementation->get_transition_value(s, a, sp, thread_id);
    }

    bool is_goal(const State& s, const std::size_t* thread_id = nullptr) {
        return _implementation->is_goal(s, thread_id);
    }

    bool is_terminal(const State& s, const std::size_t* thread_id = nullptr) {
        return _implementation->is_terminal(s, thread_id);
    }

    template <typename Tfunction, typename ... Types>
    std::unique_ptr<py::object> call(const std::size_t* thread_id, const Tfunction& func, const Types& ... args) {
        return _implementation->call(thread_id, func, args...);
    }

protected :

    template <typename TexecutionPolicy, typename Enable = void>
    struct Implementation {};

    template <typename TexecutionPolicy>
    struct Implementation<TexecutionPolicy,
                          typename std::enable_if<std::is_same<TexecutionPolicy, SequentialExecution>::value>::type> {
        Implementation(const py::object& domain) : _domain(domain) {}

        static std::size_t get_parallel_capacity() {
            return 1;
        }

        ApplicableActionSpace get_applicable_actions(const State& s, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return ApplicableActionSpace(_domain.attr("get_applicable_actions")(s.pyobj()));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting applicable actions in state ") + s.print() + ": " + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        State reset([[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return State(_domain.attr("reset")());
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when resetting the domain: ") + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        EnvironmentOutcome step(const Action& a, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return EnvironmentOutcome(_domain.attr("step")(a.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when stepping with action ") +
                            a.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        EnvironmentOutcome sample(const State& s, const Action& a, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return EnvironmentOutcome(_domain.attr("sample")(s.pyobj(), a.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when sampling from state ") + s.print() +
                              " with action " + a.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        State get_next_state(const State& s, const Action&a, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return State(_domain.attr("get_next_state")(s.pyobj(), a.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when getting next state from state ") +
                              s.print() + " and applying action " + a.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        NextStateDistribution get_next_state_distribution(const State& s, const Action& a, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return NextStateDistribution(_domain.attr("get_next_state_distribution")(s.pyobj(), a.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when getting next state distribution from state ") +
                              s.print() + " and applying action " + a.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        TransitionValue get_transition_value(const State& s, const Action& a, const State& sp, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return TransitionValue(_domain.attr("get_transition_value")(s.pyobj(), a.pyobj(), sp.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when getting value of transition (") +
                            s.print() + ", " + a.print() + ") -> " + sp.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        bool is_goal(const State& s, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return py::cast<bool>(_domain.attr("is_goal")(s.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when testing goal condition of state ") +
                            s.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        bool is_terminal(const State& s, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return py::cast<bool>(_domain.attr("is_terminal")(s.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when testing terminal condition of state ") +
                            s.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        template <typename Tfunction, typename ... Types>
        std::unique_ptr<py::object> call([[maybe_unused]] const std::size_t* thread_id, const Tfunction& func, const Types& ... args) {
            try {
                return std::make_unique<py::object>(func(_domain, args..., py::none()));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when calling anonymous domain method: " + std::string(e->what())));
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        py::object _domain;
    };

    template <typename TexecutionPolicy>
    struct Implementation<TexecutionPolicy,
                          typename std::enable_if<std::is_same<TexecutionPolicy, ParallelExecution>::value>::type> {
        std::vector<std::unique_ptr<nng::socket>> _connections;
        
        Implementation(const py::object& domain) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_domain = domain;

            if (!py::hasattr(domain, "get_ipc_connections")) {
                std::string err_msg = "SKDECIDE exception: the python domain object must provide the get_shm_files() method in parallel mode.";
                spdlog::error(err_msg);
                throw std::runtime_error(err_msg);
            } else {
                try {
                    py::list ipc_connections = domain.attr("get_ipc_connections")();
                    for (auto f : ipc_connections) {
                        _connections.push_back(std::make_unique<nng::socket>(nng::pull::open()));
                        _connections.back()->listen(std::string(py::str(f)).c_str());
                    }
                } catch (const nng::exception& e) {
                    std::string err_msg("SKDECIDE exception when trying to make pipeline connections with the python parallel domain: ");
                    err_msg += e.who() + std::string(": ") + e.what();
                    spdlog::error(err_msg);
                    throw std::runtime_error(err_msg);
                }
            }
        }

        std::size_t get_parallel_capacity() {
            typename GilControl<Texecution>::Acquire acquire;
            return py::cast<std::size_t>(_domain.attr("get_parallel_capacity")());
        }

        template <typename Tfunction, typename ... Types>
        std::unique_ptr<py::object> do_launch(const std::size_t* thread_id, const Tfunction& func, const Types& ... args) {
            std::unique_ptr<py::object> id;
            nng::socket* conn = nullptr;
            {
                typename GilControl<Texecution>::Acquire acquire;
                try {
                    if (thread_id) {
                        id = std::make_unique<py::object>(func(_domain, args..., py::int_(*thread_id)));
                    } else {
                        id = std::make_unique<py::object>(func(_domain, args..., py::none()));
                    }
                    int did = py::cast<int>(*id);
                    if (did >= 0) {
                        conn = _connections[(std::size_t) did].get();
                    }
                } catch(const py::error_already_set* e) {
                    spdlog::error("SKDECIDE exception when asynchronously calling anonymous domain method: " + std::string(e->what()));
                    std::runtime_error err(e->what());
                    id.reset();
                    delete e;
                    throw err;
                }
            }
            if (conn) { // positive id returned (parallel execution, waiting for python process to return)
                try {
                    nng::msg msg = conn->recv_msg();
                    if (msg.body().size() != 1 || msg.body().data<char>()[0] != '0') { // error
                        typename GilControl<Texecution>::Acquire acquire;
                        id.reset();
                        std::string pyerr(msg.body().data<char>(), msg.body().size());
                        throw std::runtime_error("SKDECIDE exception: C++ parallel domain received an exception from Python parallel domain: " + pyerr);
                    }
                } catch (const nng::exception& e) {
                    std::string err_msg("SKDECIDE exception when waiting for a response from the python parallel domain: ");
                    err_msg += e.who() + std::string(": ") + e.what();
                    spdlog::error(err_msg);
                    typename GilControl<Texecution>::Acquire acquire;
                    id.reset();
                    throw std::runtime_error(err_msg);
                }
            } else {
                std::string err_msg("Unable to establish a connection with the Python parallel domain");
                spdlog::error(err_msg);
                throw std::runtime_error(std::string("SKDECIDE exception: ") + err_msg);
            }
            typename GilControl<Texecution>::Acquire acquire;
            try {
                std::unique_ptr<py::object> r = std::make_unique<py::object>(_domain.attr("get_result")(*id));
                id.reset();
                return r;
            } catch(const py::error_already_set* e) {
                spdlog::error("SKDECIDE exception when asynchronously calling the domain's get_result() method: " + std::string(e->what()));
                std::runtime_error err(e->what());
                id.reset();
                delete e;
                throw err;
            }
            id.reset();
            return std::make_unique<py::object>(py::none());
        }

        template <typename ... Types>
        std::unique_ptr<py::object> launch(const std::size_t* thread_id, const char* name, const Types& ... args) {
            return do_launch(thread_id, [&name](py::object& d, auto ... aargs){
                return d.attr(name)(aargs...);
            }, args...);
        }

        ApplicableActionSpace get_applicable_actions(const State& s, const std::size_t* thread_id = nullptr) {
            try {
                return ApplicableActionSpace(launch(thread_id, "get_applicable_actions", s.pyobj()));
            } catch(const std::exception& e) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when getting applicable actions in state ") + s.print() + ": " + e.what());
                throw;
            }
        }

        State reset(const std::size_t* thread_id = nullptr) {
            try {
                return State(launch(thread_id, "reset"));
            } catch(const std::exception& e) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when resetting the domain: ") + e.what());
                throw;
            }
        }

        EnvironmentOutcome step(const Action& a, const std::size_t* thread_id = nullptr) {
            try {
                return EnvironmentOutcome(launch(thread_id, "step", a.pyobj()));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when stepping with action ") +
                              a.print() + ": " + ex.what());
                throw;
            }
        }

        EnvironmentOutcome sample(const State& s, const Action& a, const std::size_t* thread_id = nullptr) {
            try {
                return EnvironmentOutcome(launch(thread_id, "sample", s.pyobj(), a.pyobj()));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when sampling from state ") + s.print() +
                            " with action " + a.print() + ": " + ex.what());
                throw;
            }
        }

        State get_next_state(const State& s, const Action& a, const std::size_t* thread_id = nullptr) {
            try {
                return State(launch(thread_id, "get_next_state", s.pyobj(), a.pyobj()));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when getting next state from state ") +
                            s.print() + " and applying action " + a.print() + ": " + ex.what());
                throw;
            }
        }

        NextStateDistribution get_next_state_distribution(const State& s, const Action& a, const std::size_t* thread_id = nullptr) {
            try {
                return NextStateDistribution(launch(thread_id, "get_next_state_distribution", s.pyobj(), a.pyobj()));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when getting next state distribution from state ") +
                            s.print() + " and applying action " + a.print() + ": " + ex.what());
                throw;
            }
        }

        TransitionValue get_transition_value(const State& s, const Action& a, const State& sp, const std::size_t* thread_id = nullptr) {
            try {
                return TransitionValue(launch(thread_id, "get_transition_value", s.pyobj(), a.pyobj(), sp.pyobj()));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when getting value of transition (") +
                              s.print() + ", " + a.print() + ") -> " + sp.print() + ": " + ex.what());
                throw;
            }
        }

        bool is_goal(const State& s, const std::size_t* thread_id = nullptr) {
            try {
                std::unique_ptr<py::object> r = launch(thread_id, "is_goal", s.pyobj());
                typename GilControl<Texecution>::Acquire acquire;
                bool rr = py::cast<bool>(*r);
                r.reset();
                return rr;
            } catch(const std::exception& e) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when testing goal condition of state ") +
                              s.print() + ": " + e.what());
                throw;
            }
        }

        bool is_terminal(const State& s, const std::size_t* thread_id = nullptr) {
            try {
                std::unique_ptr<py::object> r = launch(thread_id, "is_terminal", s.pyobj());
                typename GilControl<Texecution>::Acquire acquire;
                bool rr = py::cast<bool>(*r);
                r.reset();
                return rr;
            } catch(const std::exception& e) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when testing terminal condition of state ") +
                              s.print() + ": " + e.what());
                throw;
            }
        }

        template <typename Tfunction, typename ... Types>
        std::unique_ptr<py::object> call(const std::size_t* thread_id, const Tfunction& func, const Types& ... args) {
            try {
                return do_launch(thread_id, func, args...);
            } catch(const std::exception& e) {
                spdlog::error(std::string("SKDECIDE exception when calling anonymous domain method: ") + e.what());
                throw;
            }
        }

        py::object _domain;
    };

    std::unique_ptr<Implementation<Texecution>> _implementation;
};

} // namespace skdecide

#endif // SKDECIDE_PYTHON_DOMAIN_ADAPTER_HH
