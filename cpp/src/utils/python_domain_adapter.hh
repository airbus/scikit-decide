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

template <typename Texecution>
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

    struct State : public PyObj<State> {
        static constexpr char class_name[] = "state";
        State() : PyObj<State>() {}
        State(std::unique_ptr<py::object>&& s) : PyObj<State>(std::move(s)) {}
        State(const py::object& s) : PyObj<State>(s) {}
        State(const State& other) : PyObj<State>(other) {}
        State& operator=(const State& other) { dynamic_cast<PyObj<State>&>(*this) = other; return *this; }
        virtual ~State() {}
    };

   typedef State Observation;

    struct Event : public PyObj<Event> {
        static constexpr char class_name[] = "event";
        Event() : PyObj<Event>() {}
        Event(std::unique_ptr<py::object>&& e) : PyObj<Event>(std::move(e)) {}
        Event(const py::object& e) : PyObj<Event>(e) {}
        Event(const Event& other) : PyObj<Event>(other) {}
        Event& operator=(const Event& other) { dynamic_cast<PyObj<Event>&>(*this) = other; return *this; }
        virtual ~Event() {}
    };

    typedef Event Action;

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

        Event sample() const {
            typename GilControl<Texecution>::Acquire acquire;
            if (!py::hasattr(*(this->_pyobj), "sample")) {
                throw std::invalid_argument("SKDECIDE exception: python applicable action object must implement sample()");
            } else {
                return Event(this->_pyobj->attr("sample")());
            }
        }
    };

    struct TransitionOutcome : public PyObj<TransitionOutcome> {
        static constexpr char class_name[] = "transition outcome";
        std::unique_ptr<py::object> _state;

        TransitionOutcome() : PyObj<TransitionOutcome>() {
            typename GilControl<Texecution>::Acquire acquire;
            _state = std::make_unique<py::object>();
        }

        TransitionOutcome(std::unique_ptr<py::object>&& outcome)
        : PyObj<TransitionOutcome>(std::move(outcome)) {
            construct();
        }

        TransitionOutcome(const py::object& outcome)
        : PyObj<TransitionOutcome>(outcome) {
            construct();
        }

        void construct() {
            typename GilControl<Texecution>::Acquire acquire;
            if (py::hasattr(*(this->_pyobj), "state")) {
                _state = std::make_unique<py::object>(this->_pyobj->attr("state"));
            } else if (py::hasattr(*(this->_pyobj), "observation")) {
                _state = std::make_unique<py::object>(this->_pyobj->attr("observation"));
            } else {
                throw std::invalid_argument("SKDECIDE exception: python transition outcome object must provide 'state' or 'observation'");
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

        TransitionOutcome(const TransitionOutcome& other)
        : PyObj<TransitionOutcome>(other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_state = std::make_unique<py::object>(*other._state);
        }

        TransitionOutcome& operator=(const TransitionOutcome& other) {
            dynamic_cast<PyObj<TransitionOutcome>&>(*this) = other;
            this->_state = std::make_unique<py::object>(*other._state);
            return *this;
        }

        virtual ~TransitionOutcome() {
            typename GilControl<Texecution>::Acquire acquire;
            _state.reset();
        }

        const py::object& state() {
            return *_state;
        }

        const py::object& observation() {
            return state();
        }

        void state(const py::object& s) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                if (py::hasattr(*(this->_pyobj), "state")) {
                    this->_pyobj->attr("state") = s;
                } else {
                    this->_pyobj->attr("observation") = s;
                }
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when setting outcome's state: ") + e->what());
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
                return py::cast<double>(this->_pyobj->attr("value").attr("cost"));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting outcome's cost: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void cost(double c) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                this->_pyobj->attr("value").attr("cost") = py::float_(c);
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when setting outcome's cost: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        double reward() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return py::cast<double>(this->_pyobj->attr("value").attr("reward"));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting outcome's reward: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void reward(double r) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                this->_pyobj->attr("value").attr("reward") = py::float_(r);
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when setting outcome's reward: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        bool terminal() {
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

        std::unique_ptr<py::object> info() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return std::make_unique<py::object>(this->_pyobj.attr("info"));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting outcome's info: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void info(const py::object& i) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                this->_pyobj->attr("info") = i;
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when setting outcome's info: ") + e->what());
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

    TransitionOutcome step(const Event& e, const std::size_t* thread_id = nullptr) {
        return _implementation->step(e, thread_id);
    }

    TransitionOutcome sample(const State& s, const Event& e, const std::size_t* thread_id = nullptr) {
        return _implementation->sample(s, e, thread_id);
    }

    State get_next_state(const State& s, const Event& e, const std::size_t* thread_id = nullptr) {
        return _implementation->get_next_state(s, e, thread_id);
    }

    NextStateDistribution get_next_state_distribution(const State& s, const Event& e, const std::size_t* thread_id = nullptr) {
        return _implementation->get_next_state_distribution(s, e, thread_id);
    }

    double get_transition_cost(const State& s, const Event& e, const State& sp, const std::size_t* thread_id = nullptr) {
        return _implementation->get_transition_cost(s, e, sp, thread_id);
    }

    double get_transition_reward(const State& s, const Event& e, const State& sp, const std::size_t* thread_id = nullptr) {
        return _implementation->get_transition_reward(s, e, sp, thread_id);
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
    struct Implementation;

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

        TransitionOutcome step(const Event& e, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return TransitionOutcome(_domain.attr("step")(e.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when stepping with action ") +
                            e.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        TransitionOutcome sample(const State& s, const Event& e, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return TransitionOutcome(_domain.attr("sample")(s.pyobj(), e.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when sampling from state ") + s.print() +
                              " with action " + e.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        State get_next_state(const State& s, const Event& e, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return State(_domain.attr("get_next_state")(s.pyobj(), e.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when getting next state from state ") +
                              s.print() + " and applying action " + e.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        NextStateDistribution get_next_state_distribution(const State& s, const Event& e, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return NextStateDistribution(_domain.attr("get_next_state_distribution")(s.pyobj(), e.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when getting next state distribution from state ") +
                              s.print() + " and applying action " + e.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        double get_transition_cost(const State& s, const Event& e, const State& sp, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return py::cast<double>(_domain.attr("get_transition_value")(s.pyobj(), e.pyobj(), sp.pyobj()).attr("cost"));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when getting value of transition (") +
                            s.print() + ", " + e.print() + ") -> " + sp.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        double get_transition_reward(const State& s, const Event& e, const State& sp, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return py::cast<double>(_domain.attr("get_transition_value")(s.pyobj(), e.pyobj(), sp.pyobj()).attr("reward"));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when getting value of transition (") +
                            s.print() + ", " + e.print() + ") -> " + sp.print() + ": " + ex->what());
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

        TransitionOutcome step(const Event& e, const std::size_t* thread_id = nullptr) {
            try {
                return TransitionOutcome(launch(thread_id, "step", e.pyobj()));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when stepping with action ") +
                              e.print() + ": " + ex.what());
                throw;
            }
        }

        TransitionOutcome sample(const State& s, const Event& e, const std::size_t* thread_id = nullptr) {
            try {
                return TransitionOutcome(launch(thread_id, "sample", s.pyobj(), e.pyobj()));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when sampling from state ") + s.print() +
                            " with action " + e.print() + ": " + ex.what());
                throw;
            }
        }

        State get_next_state(const State& s, const Event& e, const std::size_t* thread_id = nullptr) {
            try {
                return State(launch(thread_id, "get_next_state", s.pyobj(), e.pyobj()));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when getting next state from state ") +
                            s.print() + " and applying action " + e.print() + ": " + ex.what());
                throw;
            }
        }

        NextStateDistribution get_next_state_distribution(const State& s, const Event& e, const std::size_t* thread_id = nullptr) {
            try {
                return NextStateDistribution(launch(thread_id, "get_next_state_distribution", s.pyobj(), e.pyobj()));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when getting next state distribution from state ") +
                            s.print() + " and applying action " + e.print() + ": " + ex.what());
                throw;
            }
        }

        double get_transition_cost(const State& s, const Event& e, const State& sp, const std::size_t* thread_id = nullptr) {
            try {
                std::unique_ptr<py::object> r = launch(thread_id, "get_transition_value", s.pyobj(), e.pyobj(), sp.pyobj());
                typename GilControl<Texecution>::Acquire acquire;
                double rr = py::cast<double>(r->attr("cost"));
                r.reset();
                return rr;
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when getting value of transition (") +
                              s.print() + ", " + e.print() + ") -> " + sp.print() + ": " + ex.what());
                throw;
            }
        }

        double get_transition_reward(const State& s, const Event& e, const State& sp, const std::size_t* thread_id = nullptr) {
            try {
                std::unique_ptr<py::object> r = launch(thread_id, "get_transition_value", s.pyobj(), e.pyobj(), sp.pyobj());
                typename GilControl<Texecution>::Acquire acquire;
                double rr = py::cast<double>(r->attr("reward"));
                r.reset();
                return rr;
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when getting value of transition (") +
                              s.print() + ", " + e.print() + ") -> " + sp.print() + ": " + ex.what());
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
