/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_OBJECT_HH
#define SKDECIDE_PDDL_OBJECT_HH

#include "type.hh"
#include "term.hh"

namespace skdecide {

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

            virtual const std::string& get_name() const {
                return Identifier::get_name();
            }

            virtual std::ostream& print(std::ostream& o) const {
                return TypeContainer<Object>::print(o);
            }
        };

    } // namespace pddl

} // namespace skdecide

// Object printing operator
inline std::ostream& operator<<(std::ostream& o, const skdecide::pddl::Object& ob) {
    return ob.print(o);
}

#endif // SKDECIDE_PDDL_OBJECT_HH
