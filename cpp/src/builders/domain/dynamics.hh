/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_DYNAMICS_HH
#define SKDECIDE_DYNAMICS_HH

#include "core.hh"
#include "observability.hh"
#include "memory.hh"

namespace skdecide {

template <typename Tstate, typename Tobservation, typename Tevent,
          TransitionType TT, typename Tvalue, typename Tinfo,
          typename TstateSpace = Space<Tstate>,
          typename TobservationSpace = Space<Tobservation>,
          typename TobservationDistribution = Distribution<Tobservation>,
          template <typename...> class TsmartPointer = std::unique_ptr>
class EnvironmentDomain : public virtual PartiallyObservableDomain<Tstate, Tobservation, Tevent,
                                                                   TstateSpace, TobservationSpace,
                                                                   TobservationDistribution, TsmartPointer>,
                          public virtual HistoryDomain<Tstate> {
public :
    typedef Tstate State;
    typedef Tobservation Observation;
    typedef Tvalue Value;
    typedef Tinfo Info;
    typedef EnvironmentOutcome<Observation, TT, Value, Info> EnvironmentOutcomeReturn;
    typedef TransitionOutcome<State, TT, Value, Info> TransitionOutcomeReturn;
    typedef TsmartPointer<TransitionOutcomeReturn> TransitionOutcomePtr;
    typedef Tevent Event;

    EnvironmentOutcomeReturn step(const Event& event) {
        const TransitionOutcomeReturn& transition_outcome = *(make_step(event));
        const State& next_state = transition_outcome.state;
        Observation observation = get_observation_distribution(next_state, event).sample();
        if (this->get_memory_maxlen() > 0) {
            this->_memory.push_back(next_state);
        }
        return EnvironmentOutcomeReturn(observation, transition_outcome.value,
                                        transition_outcome.termination, transition_outcome.info);
    }

protected :
    virtual TransitionOutcomePtr make_step(const Event& event) = 0;
};


template <typename Tstate, typename Tobservation, typename Tevent,
          TransitionType TT, typename Tvalue, typename Tinfo,
          typename TstateSpace = Space<Tstate>,
          typename TobservationSpace = Space<Tobservation>,
          typename TobservationDistribution = Distribution<Tobservation>,
          template <typename...> class TsmartPointer = std::unique_ptr>
class SimulationDomain : public EnvironmentDomain<Tstate, Tobservation, Tevent,
                                                  TT, Tvalue, Tinfo,
                                                  TstateSpace, TobservationSpace,
                                                  TobservationDistribution, TsmartPointer> {
public :
    typedef Tstate State;
    typedef Tobservation Observation;
    typedef Tvalue Value;
    typedef Tinfo Info;
    typedef EnvironmentOutcome<Observation, TT, Value, Info> EnvironmentOutcomeReturn;
    typedef TransitionOutcome<State, TT, Value, Info> TransitionOutcomeReturn;
    typedef TsmartPointer<TransitionOutcomeReturn> TransitionOutcomePtr;
    typedef Tevent Event;
    typedef Memory<Tstate> StateMemory;

    EnvironmentOutcomeReturn sample(const StateMemory& memory, const Event& event) {
        const TransitionOutcomeReturn& transition_outcome = *(make_sample(memory, event));
        const State& next_state = transition_outcome.state;
        Observation observation = get_observation_distribution(next_state, event).sample();
        return EnvironmentOutcomeReturn(observation, transition_outcome.value,
                                        transition_outcome.termination, transition_outcome.info);
    }

    inline void set_memory(const StateMemory& m) {
        this->_memory = m;
    }

protected :
    inline virtual TransitionOutcomePtr make_step(const Event& event) {
        return make_sample(this->_memory, event);
    }

    virtual TransitionOutcomePtr make_sample(const StateMemory& memory, const Event& event) = 0;
};


template <typename Tstate, typename Tobservation, typename Tevent,
          TransitionType TT, typename Tvalue, typename Tinfo,
          typename TstateDistribution = Distribution<Tstate>,
          typename TstateSpace = Space<Tstate>,
          typename TobservationSpace = Space<Tobservation>,
          typename TobservationDistribution = Distribution<Tobservation>,
          template <typename...> class TsmartPointer = std::unique_ptr>
class UncertainTransitionDomain : public SimulationDomain<Tstate, Tobservation, Tevent,
                                                          TT, Tvalue, Tinfo,
                                                          TstateSpace, TobservationSpace,
                                                          TobservationDistribution, TsmartPointer> {
public :
    typedef Tevent Event;
    typedef Tstate State;
    typedef Memory<State> StateMemory;
    typedef TstateDistribution NextStateDistribution;
    typedef TsmartPointer<NextStateDistribution> NextStateDistributionPtr;
    typedef Tvalue Value;
    typedef Tinfo Info;
    typedef TransitionValue<TT, Value> TransitionValueReturn;
    typedef TransitionOutcome<State, TT, Value, Info> TransitionOutcomeReturn;
    typedef TsmartPointer<TransitionOutcomeReturn> TransitionOutcomePtr;

    virtual NextStateDistributionPtr get_next_state_distribution(const StateMemory& memory, const Event& event) = 0;
    virtual TransitionValueReturn get_transition_value(const StateMemory& memory, const Event& event, const State& next_state) = 0;
    virtual bool is_terminal(const State& state) = 0;

protected :
    virtual TransitionOutcomePtr make_sample(const StateMemory& memory, const Event& event) {
        State next_state = get_next_state_distribution(memory, event).sample();
        TransitionValueReturn value = get_transition_value(memory, event, next_state);
        bool termination = is_terminal(next_state);
        return std::make_unique<TransitionOutcomeReturn>(next_state, value, termination);
    }
};


template <typename Tstate, typename Tobservation, typename Tevent,
          TransitionType TT, typename Tvalue, typename Tinfo,
          typename TstateDistribution = DiscreteDistribution<Tstate>,
          typename TstateSpace = Space<Tstate>,
          typename TobservationSpace = Space<Tobservation>,
          typename TobservationDistribution = Distribution<Tobservation>,
          template <typename...> class TsmartPointer = std::unique_ptr>
class EnumerableTransitionDomain : public UncertainTransitionDomain<Tstate, Tobservation, Tevent,
                                                                    TT, Tvalue, Tinfo,
                                                                    Distribution<Tstate>, // Not the specialized DiscreteDistribution<Tstate> to allow for common base class recognition with multiple inheritance
                                                                    TstateSpace, TobservationSpace,
                                                                    TobservationDistribution, TsmartPointer> {
    static_assert(std::is_base_of<DiscreteDistribution<Tstate>, TstateDistribution>::value, "State distribution type must be derived from skdecide::DiscreteDistribution<Tstate>");
    
public :
    typedef Tevent Event;
    typedef Tstate State;
    typedef Memory<State> StateMemory;
    typedef Distribution<State> NextStateDistribution;
    typedef TsmartPointer<NextStateDistribution> NextStateDistributionPtr;

    virtual NextStateDistributionPtr get_next_state_distribution(const StateMemory& memory, const Event& event) = 0;
};


template <typename Tstate, typename Tobservation, typename Tevent,
          TransitionType TT, typename Tvalue, typename Tinfo,
          typename TstateSpace = Space<Tstate>,
          typename TobservationSpace = Space<Tobservation>,
          typename TobservationDistribution = Distribution<Tobservation>,
          template <typename...> class TsmartPointer = std::unique_ptr>
class DeterministicTransitionDomain : public EnumerableTransitionDomain<Tstate, Tobservation, Tevent,
                                                                        TT, Tvalue, Tinfo,
                                                                        Distribution<Tstate>, // Not the specialized SingleValueDistribution<Tstate> to allow for common base class recognition with multiple inheritance
                                                                        TstateSpace, TobservationSpace,
                                                                        TobservationDistribution, TsmartPointer> {
public :
    typedef Tevent Event;
    typedef Tstate State;
    typedef TsmartPointer<State> StatePtr;
    typedef Memory<State> StateMemory;
    typedef Distribution<State> NextStateDistribution;
    typedef TsmartPointer<NextStateDistribution> NextStateDistributionPtr;

    inline virtual NextStateDistributionPtr get_next_state_distribution(const StateMemory& memory, const Event& event) {
        return std::make_unique<SingleValueDistribution<State>>(get_next_state(memory, event));
    }

    virtual StatePtr get_next_state(const StateMemory& memory, const Event& event) = 0;
};

}

#endif // SKDECIDE_DYNAMICS_HH
