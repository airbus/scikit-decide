#ifndef AIRLAPS_PDDL_OBJECT_HH
#define AIRLAPS_PDDL_OBJECT_HH

#include "type.hh"
#include "term.hh"

namespace airlaps {

    namespace pddl {

        class Object : public Term,
                       public Identifier,
                       public TypeContainer<Object> {
        public :
            static constexpr char class_name[] = "object";

            Object(const std::string& name)
                : Identifier(name) {}

            Object(const Object& other)
                : Identifier(other), TypeContainer<Object>(other) {}

            Object& operator=(const Object& other) {
                dynamic_cast<Identifier&>(*this) = other;
                dynamic_cast<TypeContainer<Object>&>(*this) = other;
                return *this;
            }
        };

    } // namespace pddl

} // namespace airlaps

// Object printing operator
inline std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Object& ob) {
    return ob.print(o);
}

#endif // AIRLAPS_PDDL_OBJECT_HH
