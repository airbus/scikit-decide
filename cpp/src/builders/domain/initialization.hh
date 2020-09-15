/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_INITIALIZATION_HH
#define SKDECIDE_INITIALIZATION_HH

#include "observability.hh"
#include "memory.hh"
#include <memory>

namespace skdecide {

template <typename Tstate, typename Tobservation, typename Tevent,
          typename TstateSpace = Space<Tstate>,
          typename TobservationSpace = Space<Tobservation>,
          typename TobservationDistribution = Distribution<Tobservation>,
          template <typename...> class TsmartPointer = std::unique_ptr>
class InitializableDomain : public virtual PartiallyObservableDomain<Tstate, Tobservation, Tevent,
                                                                     TstateSpace, TobservationSpace,
                                                                     TobservationDistribution, TsmartPointer>,
                            public virtual HistoryDomain<Tstate> {
public :
    typedef Tobservation Observation;
    typedef TobservationSpace ObservationSpace;
    typedef TobservationDistribution ObservationDistribution;
    typedef TsmartPointer<ObservationDistribution> ObservationDistributionPtr;
    typedef Tstate State;
    typedef TstateSpace StateSpace;
    typedef Tevent Event;

    Observation reset() {
        State initial_state = _reset();
        Observation initial_observation = this->get_observation_distribution(initial_state, nullptr)->sample();
        auto _memory = this->_init_memory({initial_state});
        return initial_observation;
    }

protected :
    virtual State _reset() = 0;
};


template <typename Tstate, typename Tobservation, typename Tevent,
          typename TinitialStateDistribution = Distribution<Tstate>,
          typename TstateSpace = Space<Tstate>,
          typename TobservationSpace = Space<Tobservation>,
          typename TobservationDistribution = Distribution<Tobservation>,
          template <typename...> class TsmartPointer = std::unique_ptr>
class UncertainInitializedDomain : public InitializableDomain<Tstate, Tobservation, Tevent,
                                                              TstateSpace, TobservationSpace,
                                                              TobservationDistribution, TsmartPointer> {
public :
    typedef Tobservation Observation;
    typedef TobservationSpace ObservationSpace;
    typedef TobservationDistribution ObservationDistribution;
    typedef TsmartPointer<ObservationDistribution> ObservationDistributionPtr;
    typedef Tstate State;
    typedef TstateSpace StateSpace;
    typedef Tevent Event;
    typedef TinitialStateDistribution InitialStateDistribution;
    typedef TsmartPointer<InitialStateDistribution> InitialStateDistributionPtr;

    InitialStateDistribution& get_initial_state_distribution() {
        if (!_initial_state_distribution) {
            _initial_state_distribution = make_initial_state_distribution();
        }
        return *_initial_state_distribution;
    }

protected :
    virtual InitialStateDistributionPtr make_initial_state_distribution() =0;

    virtual State _reset() {
        return get_initial_state_distribution().sample();
    }

private :
    InitialStateDistributionPtr _initial_state_distribution;
};


template <typename Tstate, typename Tobservation, typename Tevent,
          typename TinitialStateDistribution = Distribution<Tobservation>,
          typename TstateSpace = Space<Tstate>,
          typename TobservationSpace = Space<Tobservation>,
          typename TobservationDistribution = Distribution<Tobservation>,
          template <typename...> class TsmartPointer = std::unique_ptr>
class DeterministicInitializedDomain : public UncertainInitializedDomain<Tstate, Tobservation, Tevent,
                                                                         TinitialStateDistribution,
                                                                         TstateSpace, TobservationSpace,
                                                                         TobservationDistribution,
                                                                         TsmartPointer> {
public :
    typedef Tobservation Observation;
    typedef TobservationSpace ObservationSpace;
    typedef TobservationDistribution ObservationDistribution;
    typedef TsmartPointer<ObservationDistribution> ObservationDistributionPtr;
    typedef Tstate State;
    typedef TsmartPointer<State> StatePtr;
    typedef TstateSpace StateSpace;
    typedef Tevent Event;
    typedef TinitialStateDistribution InitialStateDistribution;
    typedef TsmartPointer<InitialStateDistribution> InitialStateDistributionPtr;

    const State& get_initial_state() {
        if (!_initial_state) {
            _initial_state = make_initial_state();
        }
        return *_initial_state;
    }

protected :
    virtual StatePtr make_initial_state() =0;

private :
    StatePtr _initial_state;

    virtual InitialStateDistributionPtr make_initial_state_distribution() {
        return std::make_unique<SingleValueDistribution<State>>(get_initial_state());
    }
};

} // namespace skdecide

#endif // SKDECIDE_INITIALIZATION_HH
