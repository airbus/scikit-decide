/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_DOMAIN_PROXY_HH
#define SKDECIDE_PYTHON_DOMAIN_PROXY_HH

#include "python_domain_proxy_base.hh"

namespace pybind11 {
    class dict;
    class list;
    class tuple;
    namespace detail {
        template <typename Policy> class accessor;
        namespace accessor_policies { struct generic_item; }
        using item_accessor = accessor<accessor_policies::generic_item>;
        template <typename Policy> class generic_iterator;
        namespace iterator_policies{ class dict_readonly; }
        using dict_iterator = generic_iterator<iterator_policies::dict_readonly>;
    }
}

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
    using PyObj = typename PythonDomainProxyBase<Texecution>::template PyObj<Derived, Tpyobj>;

    template<typename T, typename Titerator = py::iterator>
    using PyIter = typename PythonDomainProxyBase<Texecution>::template PyIter<T, Titerator>;

    class Agent : public PyObj<Agent> {
    public :
        static constexpr char class_name[] = "agent";
        
        Agent();
        Agent(std::unique_ptr<py::object>&& a);
        Agent(const py::object& a);
        Agent(const Agent& other);
        Agent& operator=(const Agent& other);
        virtual ~Agent();
    };

    template <typename DData, typename TTagent = Tagent, typename Enable = void>
    class AgentDataAccess {};

    template <typename DData, typename TTagent>
    class AgentDataAccess<DData, TTagent,
                          typename std::enable_if<std::is_same<TTagent, SingleAgent>::value>::type> {
    public :
        typedef DData Data;
        typedef void Agent;
        typedef Data AgentData;
    };

    template <typename DData, typename TTagent>
    class AgentDataAccess<DData, TTagent,
                          typename std::enable_if<std::is_same<TTagent, MultiAgent>::value>::type> : public PyObj<DData, py::dict> {
    public :
        // AgentDataAccess inherits from pyObj<> to manage its python object but DData is passed to
        // PyObj as template parameter to print DData::class_name when managing AgentDataAccess objects

        typedef AgentDataAccess<DData, TTagent> Data;

        // Agents are dict keys
        typedef typename PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability, Tmemory>::Agent Agent;

        // AgentData are dict values
        typedef DData AgentData;

        AgentDataAccess();
        AgentDataAccess(std::unique_ptr<py::object>&& ad);
        AgentDataAccess(const py::object& ad);
        AgentDataAccess(const AgentDataAccess& other);
        AgentDataAccess& operator=(const AgentDataAccess& other);
        virtual ~AgentDataAccess();

        std::size_t size() const;

        // Dict items
        class Item : public PyObj<Item, py::tuple> {
        public :
            static constexpr char class_name[] = "dictionary item";

            Item();
            Item(std::unique_ptr<py::object>&& a);
            Item(const py::object& a);
            Item(const Item& other);
            Item& operator=(const Item& other);
            virtual ~Item();

            Agent agent();
            AgentData data();
        };

        // To access elements with dict operator []
        // Objective #1: access TagentData's methods
        // Objective #2: modify internal Python object with operator =
        class AgentDataAccessor : public PyObj<AgentDataAccessor, py::detail::item_accessor>,
                                  public AgentData {
        public :
            AgentDataAccessor(const py::detail::item_accessor& a);
            
            // We shall not assign data accessor lvalues because statements like
            // 'auto d = my_data[my_key]; d = other_data;' would assign 'other_data'
            // to 'my_data[my_key]', which is generally not the expected behaviour
            // (since one thinks to reason about the actual data but not the
            // dictionary accessor...).
            AgentDataAccessor& operator=(AgentDataAccessor&& other) & = delete;
            void operator=(const AgentData& other) & = delete;

            AgentDataAccessor& operator=(AgentDataAccessor&& other) &&;
            void operator=(const AgentData& other) &&;
            virtual ~AgentDataAccessor();
        };

        typedef typename PythonDomainProxyBase<Texecution>::template PyIter<Item, py::detail::dict_iterator> PyIter;

        AgentDataAccessor operator[](const Agent& a);
        const AgentData operator[](const Agent& a) const;
        PyIter begin() const;
        PyIter end() const;
    };

    typedef typename AgentDataAccess<typename PythonDomainProxyBase<Texecution>::State>::Data State;
    typedef typename AgentDataAccess<typename PythonDomainProxyBase<Texecution>::Observation>::Data _Observation;

    typedef typename std::conditional<std::is_same<Tobservability, FullyObservable>::value,
                                        State,
                                        typename std::conditional<std::is_same<Tobservability, PartiallyObservable>::value,
                                                                    _Observation,
                                                                    void
                                                                 >::type
                                     >::type Observation;
    
    class MemoryState : public PyObj<MemoryState, py::list> {
    public :
        typedef typename PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability, Tmemory>::State State;
        typedef typename State::AgentData AgentData;
        static constexpr char class_name[] = "memory";

        MemoryState();
        MemoryState(std::unique_ptr<py::object>&& m);
        MemoryState(const py::object& m);
        MemoryState(const MemoryState& other);
        MemoryState& operator=(const MemoryState& other);
        virtual ~MemoryState();

        void push_state(const State& s);
        State last_state();
    };

    typedef typename std::conditional<std::is_same<Tmemory, Markovian>::value,
                                        State,
                                        typename std::conditional<std::is_same<Tmemory, History>::value,
                                                                    MemoryState,
                                                                    void
                                                                 >::type
                                     >::type Memory;

    typedef typename AgentDataAccess<typename PythonDomainProxyBase<Texecution>::Action, Tagent>::Data Action;
    typedef typename AgentDataAccess<typename PythonDomainProxyBase<Texecution>::Event>::Data _Event;

    typedef typename std::conditional<std::is_same<Tcontrollability, FullyControllable>::value,
                                        Action,
                                        typename std::conditional<std::is_same<Tcontrollability, PartiallyControllable>::value,
                                                                    _Event,
                                                                    void
                                                                 >::type
                                     >::type Event;

    typedef typename AgentDataAccess<typename PythonDomainProxyBase<Texecution>::ApplicableActionSpace>::Data ApplicableActionSpace;
    typedef typename AgentDataAccess<typename PythonDomainProxyBase<Texecution>::Value>::Data Value;
    typedef typename AgentDataAccess<typename PythonDomainProxyBase<Texecution>::Predicate>::Data Predicate;

    template <typename Derived, typename SSituation>
    class Outcome : public PyObj<Derived> {
    public :
        typedef SSituation Situation;
        typedef typename PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability, Tmemory>::Value Value;
        typedef typename PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability, Tmemory>::Predicate Predicate;
        typedef typename AgentDataAccess<typename PythonDomainProxyBase<Texecution>::OutcomeInfo>::Data Info;

        Outcome();
        Outcome(std::unique_ptr<py::object>&& outcome);
        Outcome(const py::object& outcome);
        Outcome(const Situation& situation, const Value& transition_value,
                const Predicate& termination, const Info& info);
        Outcome(const Outcome& other);
        Outcome& operator=(const Outcome& other);
        virtual ~Outcome();

        Situation situation() const;
        void situation(const Situation& s);
        Value transition_value() const;
        void transition_value(const Value& tv);
        Predicate termination() const;
        void termination(const Predicate& t);
        Info info() const;
        void info(const Info& i);
    
    private :
        void construct(const Situation& situation = Situation(),
                       const Value& transition_value = Value(),
                       const Predicate& termination = Predicate(),
                       const Info& info = Info());
    };

    class TransitionOutcome : public Outcome<TransitionOutcome, State> {
    public :
        typedef typename PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability, Tmemory>::State State;

        static constexpr char pyclass[] = "TransitionOutcome";
        static constexpr char class_name[] = "transition outcome";
        static constexpr char situation_name[] = "state"; // mandatory since State == Observation in fully observable domains

        TransitionOutcome();
        TransitionOutcome(std::unique_ptr<py::object>&& outcome);
        TransitionOutcome(const py::object& outcome);
        TransitionOutcome(const State& state,
                          const Value& transition_value,
                          const Predicate& termination,
                          const typename Outcome<TransitionOutcome, State>::Info& info);
        TransitionOutcome(const Outcome<TransitionOutcome, State>& other);
        TransitionOutcome& operator=(const TransitionOutcome& other);
        virtual ~TransitionOutcome();

        State state();
        void state(const State& s);
    };

    class EnvironmentOutcome : public Outcome<EnvironmentOutcome, Observation> {
    public :
        typedef typename PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability, Tmemory>::Observation Observation;

        static constexpr char pyclass[] = "EnvironmentOutcome";
        static constexpr char class_name[] = "environment outcome";
        static constexpr char situation_name[] = "observation"; // mandatory since State == Observation in fully observable domains

        EnvironmentOutcome();
        EnvironmentOutcome(std::unique_ptr<py::object>&& outcome);
        EnvironmentOutcome(const py::object& outcome);
        EnvironmentOutcome(const Observation& observation,
                           const Value& transition_value,
                           const Predicate& termination,
                           const typename Outcome<EnvironmentOutcome, Observation>::Info& info);
        EnvironmentOutcome(const Outcome<EnvironmentOutcome, Observation>& other);
        EnvironmentOutcome& operator=(const EnvironmentOutcome& other);

        virtual ~EnvironmentOutcome();
        Observation observation();
        void observation(const Observation& o);
    };

    class NextStateDistribution : public PyObj<NextStateDistribution> {
    public :
        static constexpr char class_name[] = "next state distribution";

        NextStateDistribution();
        NextStateDistribution(std::unique_ptr<py::object>&& next_state_distribution);
        NextStateDistribution(const py::object& next_state_distribution);
        NextStateDistribution(const NextStateDistribution& other);
        NextStateDistribution& operator=(const NextStateDistribution& other);
        virtual ~NextStateDistribution();

        class DistributionValue {
        public :
            typedef typename PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability, Tmemory>::State State;
            static constexpr char class_name[] = "distribution value";
            State _state;
            double _probability;

            DistributionValue();
            DistributionValue(const py::object& o);
            DistributionValue(const DistributionValue& other);
            DistributionValue& operator=(const DistributionValue& other);

            const State& state() const;
            const double& probability() const;
        };

        class NextStateDistributionValues : public PyObj<NextStateDistributionValues> {
        public :
            typedef typename PythonDomainProxyBase<Texecution>::template PyIter<DistributionValue> PyIter;
            static constexpr char class_name[] = "next state distribution values";

            NextStateDistributionValues();
            NextStateDistributionValues(std::unique_ptr<py::object>&& next_state_distribution);
            NextStateDistributionValues(const py::object& next_state_distribution);
            NextStateDistributionValues(const NextStateDistributionValues& other);
            NextStateDistributionValues& operator=(const NextStateDistributionValues& other);
            virtual ~NextStateDistributionValues();

            PyIter begin() const;
            PyIter end() const;
        };

        NextStateDistributionValues get_values() const;
    
    private :
        void construct();
    };

    PythonDomainProxy(const py::object& domain);
    ~PythonDomainProxy();

    void close();
    std::size_t get_parallel_capacity();
    ApplicableActionSpace get_applicable_actions(const Memory& m, const std::size_t* thread_id = nullptr);

    template <typename TTagent = Tagent,
              typename TTaction = Action,
              typename TagentApplicableActions = typename PythonDomainProxyBase<Texecution>::ApplicableActionSpace>
    std::enable_if_t<std::is_same<TTagent, MultiAgent>::value, TagentApplicableActions>
    get_agent_applicable_actions(const Memory& m,
                                 const TTaction& other_agents_actions,
                                 const Agent& agent,
                                 const std::size_t* thread_id = nullptr);

    Observation reset(const std::size_t* thread_id = nullptr);
    EnvironmentOutcome step(const Event& e, const std::size_t* thread_id = nullptr);
    EnvironmentOutcome sample(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr);
    State get_next_state(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr);
    NextStateDistribution get_next_state_distribution(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr);
    Value get_transition_value(const Memory& m, const Event& e, const State& sp, const std::size_t* thread_id = nullptr);
    bool is_goal(const State& s, const std::size_t* thread_id = nullptr);
    bool is_terminal(const State& s, const std::size_t* thread_id = nullptr);

    template <typename Tfunction, typename ... Types>
    std::unique_ptr<py::object> call(const std::size_t* thread_id, const Tfunction& func, const Types& ... args);

protected :

    template <typename TexecutionPolicy = Texecution, typename Enable = void>
    struct Implementation {};

    std::unique_ptr<Implementation<Texecution>> _implementation;
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/python_domain_proxy_common_impl.hh"
#include "impl/python_domain_proxy_call_impl.hh"
#include "impl/python_domain_proxy_impl.hh"
#endif

#endif // SKDECIDE_PYTHON_DOMAIN_PROXY_HH
