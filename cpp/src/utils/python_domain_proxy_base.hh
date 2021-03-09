/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_DOMAIN_PROXY_BASE_HH
#define SKDECIDE_PYTHON_DOMAIN_PROXY_BASE_HH

#include <type_traits>
#include <memory>
#include <string>

namespace pybind11 {
    class object;
    class iterator;
    class bool_;
}

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
class PythonDomainProxyBase {
public :
    template <typename Derived, typename Tpyobj = py::object>
    class PyObj {
    public :

        PyObj();

        template <typename TTpyobj,
                  std::enable_if_t<std::is_convertible<TTpyobj, py::object>::value, int> = 0>
        PyObj(std::unique_ptr<TTpyobj>&& o, bool check = true) { move_construct(std::move(o), check); }

        template <typename TTpyobj,
                  std::enable_if_t<!std::is_base_of<typename PythonDomainProxyBase<Texecution>::template PyObj<Derived, Tpyobj>,
                                                    TTpyobj>::value, int> = 0>
        PyObj(const TTpyobj& o, bool check = true) { conv_construct(o, check); }

        PyObj(const PyObj& other);
        PyObj& operator=(const PyObj& other);
        virtual ~PyObj();

        const Tpyobj& pyobj() const;
        std::string print() const;

        struct Hash {
            std::size_t operator()(const PyObj<Derived, Tpyobj>& o) const;
        };

        struct Equal {
            bool operator()(const PyObj<Derived, Tpyobj>& o1, const PyObj<Derived, Tpyobj>& o2) const;
        };
    
    protected :

        std::unique_ptr<Tpyobj> _pyobj;

    private :
    
        struct Implementation;

        template <typename TTpyobj>
        void move_construct(std::unique_ptr<TTpyobj>&& o, bool check = true);

        template <typename TTpyobj>
        void conv_construct(const TTpyobj& o, bool check = true);
    };

    template <typename T, typename Titerator = py::iterator>
    class PyIter : PyObj<PyIter<T, Titerator>, Titerator> {
    public :

        static constexpr char class_name[] = "iterator";

        PyIter(const Titerator& iterator);
        PyIter(const PyIter<T, Titerator>& other);
        PyIter<T, Titerator>& operator=(const PyIter<T, Titerator>& other);
        virtual ~PyIter();

        PyIter<T, Titerator>& operator++();
        PyIter<T, Titerator> operator++(int);
        T operator*() const;
        std::unique_ptr<T> operator->() const;

        bool operator==(const PyIter<T, Titerator>& other) const;
        bool operator!=(const PyIter<T, Titerator>& other) const;
    
    private :
        struct Implementation;
    };

    class State : public PyObj<State> {
    public :
        typedef State AgentData;
        static constexpr char class_name[] = "state";
        State();
        State(std::unique_ptr<py::object>&& s);
        State(const py::object& s);
        State(const State& other);
        State& operator=(const State& other);
        virtual ~State();
    };

    class Observation : public PyObj<Observation> {
    public :
        typedef Observation AgentData;
        static constexpr char class_name[] = "observation";
        Observation();
        Observation(std::unique_ptr<py::object>&& o);
        Observation(const py::object& o);
        Observation(const Observation& other);
        Observation& operator=(const Observation& other);
        virtual ~Observation();
    };

    class Event : public PyObj<Event> {
    public :
        typedef Event AgentData;
        static constexpr char class_name[] = "event";
        Event();
        Event(std::unique_ptr<py::object>&& e);
        Event(const py::object& e);
        Event(const Event& other);
        Event& operator=(const Event& other);
        virtual ~Event();
    };

    class Action : public PyObj<Action> {
    public :
        typedef Action AgentData;
        static constexpr char class_name[] = "action";
        Action();
        Action(std::unique_ptr<py::object>&& a);
        Action(const py::object& a);
        Action(const Action& other);
        Action& operator=(const Action& other);
        virtual ~Action();
    };

    class ApplicableActionSpace : public PyObj<ApplicableActionSpace> {
    public :
        typedef ApplicableActionSpace AgentData;
        static constexpr char class_name[] = "applicable action space";

        ApplicableActionSpace();
        ApplicableActionSpace(std::unique_ptr<py::object>&& applicable_action_space);
        ApplicableActionSpace(const py::object& applicable_action_space);
        ApplicableActionSpace(const ApplicableActionSpace& other);
        ApplicableActionSpace& operator=(const ApplicableActionSpace& other);
        virtual ~ApplicableActionSpace();

        class Elements : public PyObj<Elements> {
        public :
            static constexpr char class_name[] = "applicable action space elements";
            typedef typename PythonDomainProxyBase<Texecution>::template PyIter<Action, py::iterator> PyIter;

            Elements();
            Elements(std::unique_ptr<py::object>&& applicable_action_space_elements);   
            Elements(const py::object& applicable_action_space_elements);
            Elements(const Elements& other);
            Elements& operator=(const Elements& other);
            virtual ~Elements();

            PyIter begin() const;
            PyIter end() const;
            bool empty() const;
        };

        Elements get_elements() const;
        bool empty() const;

        Action sample() const;
        bool contains(const Action& action);
    
    private :
        void construct();
    };

    class Value : public PyObj<Value> {
    public :
        typedef Value AgentData;
        static constexpr char class_name[] = "value";

        Value();
        Value(std::unique_ptr<py::object>&& v);
        Value(const py::object& v);
        Value(const double& value, const bool& reward_or_cost);
        Value(const Value& other);
        Value& operator=(const Value& other);
        virtual ~Value();

        double cost() const;
        void cost(const double& c);
        double reward() const;
        void reward(const double& r);
    
    private :
        void construct(const double& value = 0.0, const bool& reward_or_cost = true);
    };

    class Predicate : public PyObj<Predicate, py::bool_> {
    public :
        typedef Predicate AgentData;
        static constexpr char class_name[] = "predicate";

        Predicate();
        Predicate(std::unique_ptr<py::object>&& v);
        Predicate(const py::object& v);
        Predicate(const bool& predicate);
        Predicate(const Predicate& other);
        Predicate& operator=(const Predicate& other);
        virtual ~Predicate();

        void predicate(const bool& p);
        void operator=(const bool& p);
        bool predicate() const;
        operator bool() const;
    
    private :
        void construct();
    };

    class OutcomeInfo : public PyObj<OutcomeInfo> {
    public :
        static constexpr char class_name[] = "info";

        OutcomeInfo();
        OutcomeInfo(std::unique_ptr<py::object>&& s);
        OutcomeInfo(const py::object& s);
        OutcomeInfo(const OutcomeInfo& other);
        OutcomeInfo& operator=(const OutcomeInfo& other);
        virtual ~OutcomeInfo();

        std::size_t get_depth() const;
    };
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/python_domain_proxy_common_impl.hh"
#include "impl/python_domain_proxy_base_impl.hh"
#endif

#endif // SKDECIDE_PYTHON_DOMAIN_PROXY_BASE_HH
