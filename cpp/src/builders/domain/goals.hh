#ifndef AIRLAPS_GOALS_HH
#define AIRLAPS_GOALS_HH

#include "core.hh"

namespace airlaps {

template <typename Tobservation,
          typename TobservationSpace = Space<Tobservation>,
          template <typename...> class TsmartPointer = std::unique_ptr>
class GoalDomain {
    static_assert(std::is_same<typename TobservationSpace::element_type, Tobservation>::value, "Observation space elements must be of type Tobservation");
    static_assert(std::is_base_of<Space<Tobservation>, TobservationSpace>::value, "Observation space type must be derived from airlaps::Space<Tobservation>");
    
public :
    typedef Tobservation Observation;
    typedef TobservationSpace ObservationSpace;
    typedef TsmartPointer<ObservationSpace> ObservationSpacePtr;

    const ObservationSpace& get_goals() {
        if (!_goals) {
            _goals = make_goals();
        }
        return *_goals;
    }

    inline bool is_goal(const Observation& observation) {
        return get_goals().contains(observation);
    }

protected :
    virtual ObservationSpacePtr make_goals() =0;

private :
    ObservationSpacePtr _goals;
};

} // namespace airlaps

#endif // AIRLAPS_GOALS_HH
