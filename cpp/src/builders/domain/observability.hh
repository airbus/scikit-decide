/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_OBSERVABILITY_HH
#define SKDECIDE_OBSERVABILITY_HH

#include <memory>
#include <type_traits>
#include "core.hh"

namespace skdecide {

template <typename Tstate, typename Tobservation, typename Tevent,
          typename TstateSpace = Space<Tstate>,
          typename TobservationSpace = Space<Tobservation>,
          typename TobservationDistribution = Distribution<Tobservation>,
          template <typename...> class TsmartPointer = std::unique_ptr>
class PartiallyObservableDomain {
    static_assert(std::is_same<typename TstateSpace::element_type, Tstate>::value, "State space elements must be of type Tstate");
    static_assert(std::is_base_of<Space<Tstate>, TstateSpace>::value, "State space type must be derived from skdecide::Space<Tstate>");
    static_assert(std::is_same<typename TobservationSpace::element_type, Tobservation>::value, "Observation space elements must be of type Tobservation");
    static_assert(std::is_base_of<Space<Tobservation>, TobservationSpace>::value, "Observation space type must be derived from skdecide::Space<Tobservation>");
    static_assert(std::is_same<typename TobservationDistribution::element_type, Tobservation>::value, "Observation distribution elements must be of type Tobservation");
    static_assert(std::is_base_of<Distribution<Tobservation>, TobservationDistribution>::value, "State space type must be derived from skdecide::Space<Tstate>");

public :
    typedef Tobservation Observation;
    typedef TobservationSpace ObservationSpace;
    typedef TsmartPointer<ObservationSpace> ObservationSpacePtr;
    typedef TobservationDistribution ObservationDistribution;
    typedef TsmartPointer<ObservationDistribution> ObservationDistributionPtr;
    typedef Tstate State;
    typedef TstateSpace StateSpace;
    typedef TsmartPointer<StateSpace> StateSpacePtr;
    typedef Tevent Event;

    const ObservationSpace& get_observation_space() {
        if (!_observation_space) {
            _observation_space = make_observation_space();
        }
        return *_observation_space;
    }

    const StateSpace& get_state_space() {
        if (!_state_space) {
            _state_space = make_state_space();
        }
        return *_state_space;
    }

    inline bool is_observation(const Observation& observation) {
        return get_observation_space().contains(observation);
    }

    inline bool is_state(const State& state) {
        return get_state_space().contains(state);
    }

    virtual ObservationDistributionPtr get_observation_distribution(const State& state, const Event& event) = 0;

protected :
    virtual ObservationSpacePtr make_observation_space() =0;
    virtual StateSpacePtr make_state_space() =0;

private :
    ObservationSpacePtr _observation_space;
    StateSpacePtr _state_space;
};


template <typename Tstate, typename Tevent,
          typename TstateSpace = Space<Tstate>,
          template <typename...> class TsmartPointer = std::unique_ptr>
class FullyObservableDomain : public virtual PartiallyObservableDomain<Tstate, Tstate, Tevent,
                                                                       TstateSpace, TstateSpace,
                                                                       Distribution<Tstate>, // Not the specialized SingleValueDistribution<Tstate> to allow for common base class recognition with multiple inheritance
                                                                       TsmartPointer> {
public :
    typedef Tstate State;
    typedef TstateSpace StateSpace;
    typedef TsmartPointer<StateSpace> StateSpacePtr;
    typedef Tevent Event;
    typedef TsmartPointer<Distribution<State>> ObservationDistributionPtr;

    inline virtual ObservationDistributionPtr get_observation_distribution(const State& state, const Event& event) {
        return std::make_unique<SingleValueDistribution<State>>(state);
    }

protected :
    inline virtual StateSpacePtr make_observation_space() {
        return this->make_state_space();
    }
};

} // namespace skdecide

#endif // SKDECIDE_OBSERVABILITY_HH
