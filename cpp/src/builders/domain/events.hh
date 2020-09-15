/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_EVENTS_HH
#define SKDECIDE_EVENTS_HH

#include <memory>
#include <type_traits>
#include "core.hh"
#include "memory.hh"

namespace skdecide {

template <typename Tstate, typename Tevent,
          typename TeventSpace = Space<Tevent>,
          typename TactionSpace = TeventSpace,
          typename TenabledEventSpace = TeventSpace,
          typename TapplicableActionSpace = TactionSpace,
          template <typename...> class TsmartPointer = std::unique_ptr>
class EventDomain : public virtual HistoryDomain<Tstate> {
    static_assert(std::is_same<typename TeventSpace::element_type, Tevent>::value, "Event space elements must be of type Tevent");
    static_assert(std::is_base_of<Space<Tevent>, TeventSpace>::value, "Event space type must be derived from skdecide::Space<Tevent>");
    static_assert(std::is_same<typename TactionSpace::element_type, Tevent>::value, "Action space elements must be of type Tevent");
    static_assert(std::is_base_of<Space<Tevent>, TactionSpace>::value, "Action space type must be derived from skdecide::Space<Tevent>");
    static_assert(std::is_same<typename TenabledEventSpace::element_type, Tevent>::value, "Enabled event space elements must be of type Tevent");
    static_assert(std::is_base_of<Space<Tevent>, TenabledEventSpace>::value, "Enabled event space type must be derived from skdecide::Space<Tevent>");
    static_assert(std::is_same<typename TapplicableActionSpace::element_type, Tevent>::value, "Applicable action space elements must be of type Tevent");
    static_assert(std::is_base_of<Space<Tevent>, TapplicableActionSpace>::value, "Applicable action space type must be derived from skdecide::Space<Tevent>");

public :
    typedef Tstate State;
    typedef Memory<State> StateMemory;
    typedef Tevent Event;
    typedef TeventSpace EventSpace;
    typedef TsmartPointer<EventSpace> EventSpacePtr;
    typedef TenabledEventSpace EnabledEventSpace;
    typedef TsmartPointer<EnabledEventSpace> EnabledEventSpacePtr;
    typedef Tevent Action;
    typedef TactionSpace ActionSpace;
    typedef TsmartPointer<TactionSpace> ActionSpacePtr;
    typedef TapplicableActionSpace ApplicableActionSpace;
    typedef TsmartPointer<ApplicableActionSpace> ApplicableActionSpacePtr;

    const EventSpace& get_event_space() {
        if (!_event_space) {
            _event_space = make_event_space();
        }
        return *_event_space;
    }

    virtual EnabledEventSpacePtr get_enabled_events(const StateMemory& memory) = 0;

    inline EnabledEventSpacePtr get_enabled_events() {
        return get_enabled_events(this->_memory);
    }

    inline bool is_enabled_event(const Event& event, const StateMemory& memory) {
        return get_enabled_events(memory).contains(event);
    }

    inline bool is_enabled_event(const Event& event) {
        return is_enabled_event(event, this->_memory);
    }

    const ActionSpace& get_action_space() {
        if (!_action_space) {
            _action_space = make_action_space();
        }
        return *_action_space;
    }

    inline bool is_action(const Event& event) {
        return get_action_space().contains(event);
    }

    virtual ApplicableActionSpacePtr get_applicable_actions(const StateMemory& memory) = 0;

    inline ApplicableActionSpacePtr get_applicable_actions() {
        return get_applicable_actions(this->_memory);
    }

    inline bool is_applicable_action(const Event& event, const StateMemory& memory) {
        return get_applicable_actions(memory).contains(event);
    }

    inline bool is_applicable_action(const Event& event) {
        return is_applicable_action(event, this->_memory);
    }

protected :
    virtual EventSpacePtr make_event_space() =0;
    virtual ActionSpacePtr make_action_space() =0;

private :
    EventSpacePtr _event_space;
    ActionSpacePtr _action_space;
};


template <typename Tstate, typename Taction,
          typename TactionSpace = Space<Taction>,
          typename TapplicableActionSpace = TactionSpace,
          template <typename...> class TsmartPointer = std::unique_ptr>
class ActionDomain : public EventDomain<Tstate, Taction,
                                        TactionSpace, TactionSpace,
                                        TapplicableActionSpace, TapplicableActionSpace,
                                        TsmartPointer> {
public :
    typedef Tstate State;
    typedef Memory<State> MemoryState;
    typedef Taction Action;
    typedef TactionSpace ActionSpace;
    typedef TapplicableActionSpace ApplicableActionSpace;
    typedef TsmartPointer<ApplicableActionSpace> ApplicableActionSpacePtr;

    inline virtual const ActionSpace& get_event_space() {
        return this->get_action_space();
    }
    
    inline virtual ApplicableActionSpacePtr get_enabled_events(const MemoryState& memory) {
        return this->get_enabled_actions(memory);
    }
};


template <typename Tstate, typename Taction,
          typename TactionSpace = Space<Taction>,
          typename TapplicableActionSpace = TactionSpace,
          template <typename...> class TsmartPointer = std::unique_ptr>
class UnrestrictedActionDomain : public ActionDomain<Tstate, Taction,
                                                     TactionSpace, TapplicableActionSpace,
                                                     TsmartPointer> {
public :
    typedef Tstate State;
    typedef Memory<State> StateMemory;
    typedef Taction Action;
    typedef TactionSpace ActionSpace;
    typedef TapplicableActionSpace ApplicableActionSpace;
    typedef TsmartPointer<ApplicableActionSpace> ApplicableActionSpacePtr;

    inline virtual ApplicableActionSpacePtr get_applicable_actions(const StateMemory& memory) {
        return this->get_action_space();
    }
};

} // namespace skdecide

#endif // SKDECIDE_EVENTS_HH
