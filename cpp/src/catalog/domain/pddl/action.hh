#ifndef AIRLAPS_PDDL_ACTION_HH
#define AIRLAPS_PDDL_ACTION_HH

#include "identifier.hh"
#include "variable_container.hh"

namespace airlaps {

    namespace pddl {

        class Action : public Identifier,
                       public VariableContainer<Action> {
        public :
            static constexpr char class_name[] = "action";

            Action(const std::string& name)
                : Identifier(name) {}

            Action(const Action& other)
                : Identifier(other), VariableContainer<Action>(other) {}

            Action& operator=(const Action& other) {
                dynamic_cast<Identifier&>(*this) = other;
                dynamic_cast<VariableContainer<Action>&>(*this) = other;
                return *this;
            }

            typedef VariableContainer<Action>::VariablePtr VariablePtr;
            typedef VariableContainer<Action>::VariableVector VariableVector;
        };

    } // namespace pddl

} // namespace airlaps

// Object printing operator
inline std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Action& p) {
    return p.print(o);
}

#endif // AIRLAPS_PDDL_ACTION_HH
