/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_DOMAIN_PROXY_HH
#define SKDECIDE_PYTHON_DOMAIN_PROXY_HH

#include <pybind11/pybind11.h>

#include <nngpp/nngpp.h>
#include <nngpp/protocol/pull0.h>

#include "utils/python_gil_control.hh"
#include "utils/python_globals.hh"
#include "utils/python_hash_eq.hh"
#include "utils/execution.hh"

namespace py = pybind11;

namespace skdecide {

struct SingleAgent { SingleAgent() {} };
struct MultiAgent { MultiAgent() {} };

struct PartiallyObservable { PartiallyObservable() {} };
struct FullyObservable { FullyObservable() {} };

struct PartiallyControllable { PartiallyControllable() {} };
struct FullyControllable { FullyControllable() {} };

struct Markovian { Markovian() {} };
struct History { History() {} };

template <typename Texecution,
          typename Tagent = SingleAgent,
          typename Tobservability = FullyObservable,
          typename Tcontrollability = FullyControllable,
          typename Tmemory = Markovian>
class PythonDomainProxy {
public :
    template <typename Derived, typename Tpyobj = py::object>
    struct PyObj {
        std::unique_ptr<Tpyobj> _pyobj;

        template <typename TTpyobj = Tpyobj,
                  std::enable_if_t<std::is_same<TTpyobj, py::object>::value, int> = 0>
        PyObj() {
            typename GilControl<Texecution>::Acquire acquire;
            _pyobj = std::make_unique<Tpyobj>(py::none());
        }

        template <typename TTpyobj = Tpyobj,
                  std::enable_if_t<!std::is_same<TTpyobj, py::object>::value &&
                                   std::is_base_of<py::object, TTpyobj>::value, int> = 0>
        PyObj() {
            typename GilControl<Texecution>::Acquire acquire;
            _pyobj = std::make_unique<Tpyobj>(Tpyobj());
        }

        template <typename TTpyobj = Tpyobj,
                  std::enable_if_t<std::is_same<TTpyobj, py::object>::value, int> = 0>
        PyObj(std::unique_ptr<py::object>&& o, bool check = true) : _pyobj(std::move(o)) {
            if (check && (!_pyobj || !(*_pyobj))) {
                throw std::runtime_error(std::string("Unitialized python ") +
                                         Derived::class_name + " object!");
            }
        }

        template <typename TTpyobj = Tpyobj,
                  std::enable_if_t<!std::is_same<TTpyobj, py::object>::value &&
                                   std::is_base_of<py::object, TTpyobj>::value, int> = 0>
        PyObj(std::unique_ptr<py::object>&& o, bool check = true) {
            typename GilControl<Texecution>::Acquire acquire;
            if (check && (!o || !(*o) || !py::isinstance<Tpyobj>(*o))) {
                throw std::runtime_error(std::string("Python ") + Derived::class_name + " object not initialized as a " +
                                         std::string(py::str(Tpyobj().attr("__class__").attr("__name__"))));
            }
            _pyobj = std::make_unique<Tpyobj>(o->template cast<Tpyobj>());
            o.reset();
        }

        template <typename TTpyobj = Tpyobj,
                  std::enable_if_t<!std::is_same<TTpyobj, py::object>::value &&
                                   std::is_base_of<py::object, TTpyobj>::value, int> = 0>
        PyObj(std::unique_ptr<TTpyobj>&& o, bool check = true) {
            if (check && (!_pyobj || !(*_pyobj))) {
                throw std::runtime_error(std::string("Unitialized python ") +
                                         Derived::class_name + " object!");
            }
        }

        template <typename TTpyobj = Tpyobj,
                  std::enable_if_t<std::is_same<TTpyobj, py::object>::value, int> = 0>
        PyObj(const py::object& o, bool check = true) {
            typename GilControl<Texecution>::Acquire acquire;
            _pyobj = std::make_unique<py::object>(o);
            if (check && !(*_pyobj)) {
                throw std::runtime_error(std::string("Unitialized python ") +
                                         Derived::class_name + " object!");
            }
        }

        template <typename TTpyobj = Tpyobj,
                  std::enable_if_t<!std::is_same<TTpyobj, py::object>::value &&
                                   std::is_base_of<py::object, TTpyobj>::value, int> = 0>
        PyObj(const py::object& o, bool check = true) {
            if (check && (!o || !py::isinstance<Tpyobj>(o))) {
                throw std::runtime_error(std::string("Python ") + Derived::class_name + " object not initialized as a " +
                                         std::string(py::str(Tpyobj().attr("__class__").attr("__name__"))));
            }
            _pyobj = std::make_unique<Tpyobj>(o.template cast<Tpyobj>());
        }

        template <typename TTpyobj = Tpyobj,
                  std::enable_if_t<!std::is_same<TTpyobj, py::object>::value &&
                                   std::is_base_of<py::object, TTpyobj>::value, int> = 0>
        PyObj(const TTpyobj& o, bool check = true) {
            typename GilControl<Texecution>::Acquire acquire;
            _pyobj = std::make_unique<Tpyobj>(o);
            if (check && !(*_pyobj)) {
                throw std::runtime_error(std::string("Unitialized python ") +
                                         Derived::class_name + " object!");
            }
        }

        PyObj(const PyObj& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_pyobj = std::make_unique<Tpyobj>(*other._pyobj);
        }

        PyObj& operator=(const PyObj& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_pyobj = std::make_unique<Tpyobj>(*other._pyobj);
            return *this;
        }

        virtual ~PyObj() {
            typename GilControl<Texecution>::Acquire acquire;
            _pyobj.reset();
        }

        const Tpyobj& pyobj() const { return *_pyobj; }

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
    struct PyIter : PyObj<PyIter<T>, py::iterator> {
        static constexpr char class_name[] = "iterator";

       PyIter(const py::iterator& iterator)
        : PyObj<PyIter<T>, py::iterator>(iterator, false) {}

        PyIter(const PyIter& other)
        : PyObj<PyIter<T>, py::iterator>(other) {}

        PyIter& operator=(const PyIter& other) {
            dynamic_cast<PyObj<PyIter<T>, py::iterator>&>(*this) = other;
            return *this;
        }

        virtual ~PyIter() {}

        PyIter<T>& operator++() {
            typename GilControl<Texecution>::Acquire acquire;
            ++(*(this->_pyobj));
            return *this;
        }

        PyIter<T> operator++(int) {
            typename GilControl<Texecution>::Acquire acquire;
            py::iterator rv = (*(this->_pyobj))++;
            return PyIter<T>(rv);
        }

        T operator*() const {
            typename GilControl<Texecution>::Acquire acquire;
            return T(py::reinterpret_borrow<py::object>(**(this->_pyobj)));
        }

        std::unique_ptr<T> operator->() const {
            typename GilControl<Texecution>::Acquire acquire;
            return std::make_unique<T>(py::reinterpret_borrow<py::object>(**(this->_pyobj)));
        }

        bool operator==(const PyIter<T>& other) const {
            typename GilControl<Texecution>::Acquire acquire;
            return *(this->_pyobj) == *(other._pyobj);
        }

        bool operator!=(const PyIter<T>& other) const {
            typename GilControl<Texecution>::Acquire acquire;
            return *(this->_pyobj) != *(other._pyobj);
        }
    };

    template <typename Inherited, typename TTagent, typename Enable = void>
    struct AgentData {};

    template <typename Inherited, typename TTagent>
    struct AgentData<Inherited, TTagent,
                     typename std::enable_if<std::is_same<TTagent, SingleAgent>::value>::type> : public Inherited {
        AgentData() : Inherited() {}
        AgentData(std::unique_ptr<py::object>&& ad) : Inherited(std::move(ad)) {}
        AgentData(const py::object& ad) : Inherited(ad) {}
        AgentData(const AgentData& other) : Inherited(other) {}
        AgentData& operator=(const AgentData& other) { dynamic_cast<Inherited&>(*this) = other; return *this; }
        virtual ~AgentData() {}
    };

    template <typename Inherited, typename TTagent>
    struct AgentData<Inherited, TTagent,
                     typename std::enable_if<std::is_same<TTagent, MultiAgent>::value>::type> : PyObj<Inherited, py::dict> {
        // AgentData inherits from pyObj<> to manage its python object but Inherited is passed to
        // PyObj as template parameter to print Inherited::class_name when managing AgentData objects

        AgentData() : PyObj<Inherited, py::dict>() {}

        AgentData(std::unique_ptr<py::object>&& ad)
        : PyObj<Inherited, py::dict>(std::move(ad)) {}

        AgentData(const py::object& ad)
        : PyObj<Inherited, py::dict>(ad) {}
        
        AgentData(const AgentData& other)
        : PyObj<Inherited, py::dict>(other) {}

        AgentData& operator=(const AgentData& other) {
            dynamic_cast<PyObj<Inherited, py::dict>&>(*this) = other;
            return *this;
        }

        virtual ~AgentData() {}

        std::size_t size() {
            typename GilControl<Texecution>::Acquire acquire;
            return this->_pyobj->size();
        }

        // Agents are dict keys
        struct Agent : public PyObj<Agent> {
            static constexpr char class_name[] = "agent";
            Agent() : PyObj<Agent>() {}
            Agent(std::unique_ptr<py::object>&& a) : PyObj<Agent>(std::move(a)) {}
            Agent(const py::object& a) : PyObj<Agent>(a) {}
            Agent(const Agent& other) : PyObj<Agent>(other) {}
            Agent& operator=(const Agent& other) { dynamic_cast<PyObj<PyObj<Agent>>&>(*this) = other; return *this; }
            virtual ~Agent() {}
        };

        // Elements are dict values
        struct Element : public Inherited {
            Element() : Inherited() {}
            Element(std::unique_ptr<py::object>&& e) : Inherited(std::move(e)) {}
            Element(const py::object& e) : Inherited(e) {}
            Element(const Element& other) : Inherited(other) {}
            Element& operator=(const Element& other) { dynamic_cast<PyObj<Inherited>&>(*this) = other; return *this; }
            virtual ~Element() {}
        };

        // Dict items
        struct Item : public PyObj<Item, py::tuple> {
            static constexpr char class_name[] = "dictionary item";

            Item() : PyObj<Item>() {}
            Item(std::unique_ptr<py::object>&& a) : PyObj<Item>(std::move(a)) {}
            Item(const py::object& a) : PyObj<Item>(a) {}
            Item(const Item& other) : PyObj<Item>(other) {}
            Item& operator=(const Item& other) { dynamic_cast<PyObj<PyObj<Item>>&>(*this) = other; return *this; }
            virtual ~Item() {}

            Agent agent() {
                typename GilControl<Texecution>::Acquire acquire;
                return Agent((*(this->_pyobj))[0]);
            }

            Element element() {
                typename GilControl<Texecution>::Acquire acquire;
                return Element((*(this->_pyobj))[1]);
            }
        };

        Element operator[](const Agent& a) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return Element((*(this->_pyobj))[a.pyobj()]);
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting ") +
                              Inherited::class_name : " of agent " + a.print() +
                              ": " + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        PyIter<Item> begin() const {
            typename GilControl<Texecution>::Acquire acquire;
            return PyIter<Item>(this->_pyobj->begin());
        }

        PyIter<Item> end() const {
            typename GilControl<Texecution>::Acquire acquire;
            return PyIter<Item>(this->_pyobj->end());
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
        ObservationBase(std::unique_ptr<py::object>&& o) : PyObj<ObservationBase>(std::move(o)) {}
        ObservationBase(const py::object& o) : PyObj<ObservationBase>(o) {}
        ObservationBase(const ObservationBase& other) : PyObj<ObservationBase>(other) {}
        ObservationBase& operator=(const ObservationBase& other) { dynamic_cast<PyObj<ObservationBase>&>(*this) = other; return *this; }
        virtual ~ObservationBase() {}
    };
    
    typedef typename std::conditional<std::is_same<Tobservability, FullyObservable>::value,
                                        State,
                                        typename std::conditional<std::is_same<Tobservability, PartiallyObservable>::value,
                                                                    AgentData<ObservationBase, Tagent>,
                                                                    void
                                                                 >::type
                                     >::type Observation;
    
    struct MemoryState : PyObj<MemoryState, py::list> {
        static constexpr char class_name[] = "memory";

        MemoryState() : PyObj<MemoryState, py::list>() {}

        MemoryState(std::unique_ptr<py::object>&& m)
        : PyObj<MemoryState, py::list>(std::move(m)) {}

        MemoryState(const py::object& m) : PyObj<MemoryState, py::list>(m) {}

        MemoryState(const MemoryState& other) : PyObj<MemoryState>(other) {}

        MemoryState& operator=(const MemoryState& other) {
            dynamic_cast<PyObj<MemoryState>&>(*this) = other;
            return *this;
        }

        virtual ~MemoryState() {}

        void push_state(const State& s) {
            typename GilControl<Texecution>::Acquire acquire;
            static_cast<py::list&>(*(this->_pyobj)).append(s.pyobj());
        }

        State last_state() {
            typename GilControl<Texecution>::Acquire acquire;
            py::list& l = static_cast<py::list&>(*(this->_pyobj));
            if (l.empty()) {
                throw std::runtime_error("Cannot get last state of empty memory state " + this->print());
            } else {
                return State(l[l.size() - 1]);
            }
        }
    };

    typedef typename std::conditional<std::is_same<Tmemory, Markovian>::value,
                                        State,
                                        typename std::conditional<std::is_same<Tmemory, History>::value,
                                                                    MemoryState,
                                                                    void
                                                                 >::type
                                     >::type Memory;

    struct EventBase : public PyObj<EventBase> {
        static constexpr char class_name[] = "event";
        EventBase() : PyObj<EventBase>() {}
        EventBase(std::unique_ptr<py::object>&& e) : PyObj<EventBase>(std::move(e)) {}
        EventBase(const py::object& e) : PyObj<EventBase>(e) {}
        EventBase(const EventBase& other) : PyObj<EventBase>(other) {}
        EventBase& operator=(const EventBase& other) { dynamic_cast<PyObj<EventBase>&>(*this) = other; return *this; }
        virtual ~EventBase() {}
    };

    struct ActionBase : public PyObj<ActionBase> {
        static constexpr char class_name[] = "action";
        ActionBase() : PyObj<ActionBase>() {}
        ActionBase(std::unique_ptr<py::object>&& a) : PyObj<ActionBase>(std::move(a)) {}
        ActionBase(const py::object& a) : PyObj<ActionBase>(a) {}
        ActionBase(const ActionBase& other) : PyObj<ActionBase>(other) {}
        ActionBase& operator=(const ActionBase& other) { dynamic_cast<PyObj<ActionBase>&>(*this) = other; return *this; }
        virtual ~ActionBase() {}
    };

    typedef typename std::conditional<std::is_same<Tcontrollability, FullyControllable>::value,
                                        AgentData<ActionBase, Tagent>,
                                        void
                                     >::type Action;

    typedef typename std::conditional<std::is_same<Tcontrollability, FullyControllable>::value,
                                        Action,
                                        typename std::conditional<std::is_same<Tcontrollability, PartiallyControllable>::value,
                                                                    AgentData<EventBase, Tagent>,
                                                                    void
                                                                 >::type
                                     >::type Event;

    struct ApplicableActionSpaceBase : public PyObj<ApplicableActionSpaceBase> {
        static constexpr char class_name[] = "applicable action space";

        ApplicableActionSpaceBase() : PyObj<ApplicableActionSpaceBase>() {
            construct();
        }

        ApplicableActionSpaceBase(std::unique_ptr<py::object>&& applicable_action_space)
        : PyObj<ApplicableActionSpaceBase>(std::move(applicable_action_space)) {
            construct();
        }
        
        ApplicableActionSpaceBase(const py::object& applicable_action_space)
        : PyObj<ApplicableActionSpaceBase>(applicable_action_space) {
            construct();
        }
        
        void construct() {
            typename GilControl<Texecution>::Acquire acquire;
            if (this->_pyobj->is_none()) {
                this->_pyobj = std::make_unique<py::object>(skdecide::Globals::skdecide().attr("EmptySpace")());
            } else if (!py::hasattr(*(this->_pyobj), "get_elements")) {
                throw std::invalid_argument("SKDECIDE exception: python applicable action object must implement get_elements()");
            }
        }

        ApplicableActionSpaceBase(const ApplicableActionSpaceBase& other)
        : PyObj<ApplicableActionSpaceBase>(other) {}

        ApplicableActionSpaceBase& operator=(const ApplicableActionSpaceBase& other) {
            dynamic_cast<PyObj<ApplicableActionSpaceBase>&>(*this) = other;
            return *this;
        }

        virtual ~ApplicableActionSpaceBase() {}

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
    typedef AgentData<ApplicableActionSpaceBase, Tagent> ApplicableActionSpace;

    struct ValueBase : public PyObj<ValueBase> {
        static constexpr char class_name[] = "value";

        ValueBase() : PyObj<ValueBase>() { construct(); }
        ValueBase(std::unique_ptr<py::object>&& v) : PyObj<ValueBase>(std::move(v)) { construct(); }
        ValueBase(const py::object& v) : PyObj<ValueBase>(v) { construct(); }
        ValueBase(const ValueBase& other) : PyObj<ValueBase>(other) {}
        ValueBase& operator=(const ValueBase& other) { dynamic_cast<PyObj<ValueBase>&>(*this) = other; return *this; }
        virtual ~ValueBase() {}

        void construct() {
            typename GilControl<Texecution>::Acquire acquire;
            if (this->_pyobj->is_none()) {
                this->_pyobj = std::make_unique<py::object>(skdecide::Globals::skdecide().attr("Value")());
            } else {
                if (!py::hasattr(*(this->_pyobj), "cost")) {
                    throw std::invalid_argument("SKDECIDE exception: python value object must provide the 'cost' attribute");
                }
                if (!py::hasattr(*(this->_pyobj), "reward")) {
                    throw std::invalid_argument("SKDECIDE exception: python value object must provide the 'reward' attribute");
                }
            }
        }

        double cost() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return py::cast<double>(this->_pyobj->attr("cost"));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting value's cost: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void cost(const double& c) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                this->_pyobj->attr("cost") = py::float_(c);
                this->_pyobj->attr("reward") = py::float_(-c);
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when setting value's cost: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        double reward() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return this->_pyobj->attr("reward").template cast<double>();
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting value's reward: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void reward(const double& r) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                this->_pyobj->attr("reward") = py::float_(r);
                this->_pyobj->attr("cost") = py::float_(-r);
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when setting value's reward: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }
    };
    typedef AgentData<ValueBase, Tagent> Value;

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

        Outcome() : PyObj<Derived>() { construct(); }

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
            if (this->_pyobj->is_none()) {
                this->_pyobj = std::make_unique<py::object>(skdecide::Globals::skdecide().attr(Derived::pyclass)(py::none()));
            } else {
                if (!py::hasattr(*(this->_pyobj), Derived::situation_name)) {
                    throw std::invalid_argument(std::string("SKDECIDE exception: python transition outcome object must provide '") +
                                                Derived::situation_name + "'");
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
                return Situation(this->_pyobj->attr(Derived::situation_name));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting outcome's ") +
                              Derived::situation_name + ": " + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void situation(const Situation& s) {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                this->_pyobj->attr(Derived::situation_name) = s.pyobj();
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when setting outcome's ") +
                              Derived::situation_name + ": " + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        Value transition_value() {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return Value(this->_pyobj->attr("value"));
            } catch(const py::error_already_set* e) {
                spdlog::error(std::string("SKDECIDE exception when getting outcome's transition value: ") + e->what());
                std::runtime_error err(e->what());
                delete e;
                throw err;
            }
        }

        void transition_value(const Value& tv) {
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
        static constexpr char pyclass[] = "TransitionOutcome";
        static constexpr char class_name[] = "transition outcome";
        static constexpr char situation_name[] = "state"; // mandatory since State == Observation in fully observable domains

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
        static constexpr char pyclass[] = "EnvironmentOutcome";
        static constexpr char class_name[] = "environment outcome";
        static constexpr char situation_name[] = "observation"; // mandatory since State == Observation in fully observable domains

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

        NextStateDistribution() : PyObj<NextStateDistribution>() { construct(); }

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
            if (this->_pyobj->is_none()) {
                this->_pyobj = std::make_unique<py::object>(skdecide::Globals::skdecide().attr("DiscreteDistribution")(py::list()));
            } else if (!py::hasattr(*(this->_pyobj), "get_values")) {
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

    PythonDomainProxy(const py::object& domain) {
        _implementation = std::make_unique<Implementation<Texecution>>(domain);
    }

    std::size_t get_parallel_capacity() {
        return _implementation->get_parallel_capacity();
    }

    ApplicableActionSpace get_applicable_actions(const Memory& m, const std::size_t* thread_id = nullptr) {
        return _implementation->get_applicable_actions(m, thread_id);
    }

    Observation reset(const std::size_t* thread_id = nullptr) {
        return _implementation->reset(thread_id);
    }

    EnvironmentOutcome step(const Event& e, const std::size_t* thread_id = nullptr) {
        return _implementation->step(e, thread_id);
    }

    EnvironmentOutcome sample(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr) {
        return _implementation->sample(m, e, thread_id);
    }

    State get_next_state(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr) {
        return _implementation->get_next_state(m, e, thread_id);
    }

    NextStateDistribution get_next_state_distribution(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr) {
        return _implementation->get_next_state_distribution(m, e, thread_id);
    }

    Value get_transition_value(const Memory& m, const Event& e, const State& sp, const std::size_t* thread_id = nullptr) {
        return _implementation->get_transition_value(m, e, sp, thread_id);
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

        Observation reset([[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return Observation(_domain.attr("reset")());
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when resetting the domain: ") + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        EnvironmentOutcome step(const Event& e, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return EnvironmentOutcome(_domain.attr("step")(e.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when stepping with action ") +
                            e.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        EnvironmentOutcome sample(const Memory& m, const Event& e, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return EnvironmentOutcome(_domain.attr("sample")(m.pyobj(), e.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when sampling from ") + Memory::class_name + " " +
                              m.print() + " with action " + e.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        State get_next_state(const Memory& m, const Event&e, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return State(_domain.attr("get_next_state")(m.pyobj(), e.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when getting next state from ") + Memory::class_name + " " +
                              m.print() + " and applying action " + e.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        NextStateDistribution get_next_state_distribution(const Memory& m, const Event& e, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return NextStateDistribution(_domain.attr("get_next_state_distribution")(m.pyobj(), e.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when getting next state distribution from ") + Memory::class_name + " " +
                              m.print() + " and applying action " + e.print() + ": " + ex->what());
                std::runtime_error err(ex->what());
                delete ex;
                throw err;
            }
        }

        Value get_transition_value(const Memory& m, const Event& e, const State& sp, [[maybe_unused]] const std::size_t* thread_id = nullptr) {
            try {
                return Value(_domain.attr("get_transition_value")(m.pyobj(), e.pyobj(), sp.pyobj()));
            } catch(const py::error_already_set* ex) {
                spdlog::error(std::string("SKDECIDE exception when getting value of transition (") +
                            m.print() + ", " + e.print() + ") -> " + sp.print() + ": " + ex->what());
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

        ApplicableActionSpace get_applicable_actions(const Memory& m, const std::size_t* thread_id = nullptr) {
            try {
                return ApplicableActionSpace(launch(thread_id, "get_applicable_actions", m.pyobj()));
            } catch(const std::exception& e) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when getting applicable actions in ") + Memory::class_name + " " +
                              m.print() + ": " + e.what());
                throw;
            }
        }

        Observation reset(const std::size_t* thread_id = nullptr) {
            try {
                return Observation(launch(thread_id, "reset"));
            } catch(const std::exception& e) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when resetting the domain: ") + e.what());
                throw;
            }
        }

        EnvironmentOutcome step(const Event& e, const std::size_t* thread_id = nullptr) {
            try {
                return EnvironmentOutcome(launch(thread_id, "step", e.pyobj()));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when stepping with action ") +
                              e.print() + ": " + ex.what());
                throw;
            }
        }

        EnvironmentOutcome sample(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr) {
            try {
                return EnvironmentOutcome(launch(thread_id, "sample", m.pyobj(), e.pyobj()));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when sampling from ") + Memory::class_name +
                              m.print() + " " + " with action " + e.print() + ": " + ex.what());
                throw;
            }
        }

        State get_next_state(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr) {
            try {
                return State(launch(thread_id, "get_next_state", m.pyobj(), e.pyobj()));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when getting next state from ") + Memory::class_name + " " +
                              m.print() + " and applying action " + e.print() + ": " + ex.what());
                throw;
            }
        }

        NextStateDistribution get_next_state_distribution(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr) {
            try {
                return NextStateDistribution(launch(thread_id, "get_next_state_distribution", m.pyobj(), e.pyobj()));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when getting next state distribution from ") + Memory::class_name + " " +
                              m.print() + " and applying action " + e.print() + ": " + ex.what());
                throw;
            }
        }

        Value get_transition_value(const Memory& m, const Event& e, const State& sp, const std::size_t* thread_id = nullptr) {
            try {
                return Value(launch(thread_id, "get_transition_value", m.pyobj(), e.pyobj(), sp.pyobj()));
            } catch(const std::exception& ex) {
                typename GilControl<Texecution>::Acquire acquire;
                spdlog::error(std::string("SKDECIDE exception when getting value of transition (") +
                              m.print() + ", " + e.print() + ") -> " + sp.print() + ": " + ex.what());
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

#endif // SKDECIDE_PYTHON_DOMAIN_PROXY_HH
