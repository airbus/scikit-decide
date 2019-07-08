#ifndef AIRLAPS_PDDL_PREDICATE_HH
#define AIRLAPS_PDDL_PREDICATE_HH

#include "identifier.hh"
#include "variable_container.hh"

namespace airlaps {

    namespace pddl {

        class Predicate : public Identifier,
                          public VariableContainer<Predicate> {
        public :
            static constexpr char class_name[] = "predicate";

            Predicate(const std::string& name)
                : Identifier(name) {}

            Predicate(const Predicate& other)
                : Identifier(other), VariableContainer<Predicate>(other) {}

            Predicate& operator=(const Predicate& other) {
                dynamic_cast<Identifier&>(*this) = other;
                dynamic_cast<VariableContainer<Predicate>&>(*this) = other;
                return *this;
            }

            typedef VariableContainer<Predicate>::VariablePtr VariablePtr;
            typedef VariableContainer<Predicate>::VariableVector VariableVector;
        };

    } // namespace pddl

} // namespace airlaps

// Object printing operator
inline std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Predicate& p) {
    return p.print(o);
}

#endif // AIRLAPS_PDDL_PREDICATE_HH
