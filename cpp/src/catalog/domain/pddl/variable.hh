#ifndef AIRLAPS_PDDL_VARIABLE_HH
#define AIRLAPS_PDDL_VARIABLE_HH

#include "type.hh"
#include "term.hh"

namespace airlaps {

    namespace pddl {

        class Variable : public Term, public TypeContainer<Variable> {
        public :
            static constexpr char cls_name[] = "variable";

            Variable(const std::string& name)
                : TypeContainer<Variable>(name) {}

            Variable(const Variable& other)
                : TypeContainer<Variable>(other) {}

            Variable& operator=(const Variable& other) {
                dynamic_cast<TypeContainer<Variable>&>(*this) = other;
                return *this;
            }
        };

    } // namespace pddl

} // namespace airlaps

// Object printing operator
inline std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Variable& v) {
    return v.print(o);
}

#endif // AIRLAPS_PDDL_VARIABLE_HH
