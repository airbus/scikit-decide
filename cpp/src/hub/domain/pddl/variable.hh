/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_VARIABLE_HH
#define AIRLAPS_PDDL_VARIABLE_HH

#include "type.hh"
#include "term.hh"

namespace airlaps {

    namespace pddl {

        class Variable : public Term,
                         public Identifier,
                         public TypeContainer<Variable> {
        public :
            static constexpr char class_name[] = "variable";

            Variable(const std::string& name)
                : Identifier("?" + name) {}

            Variable(const Variable& other)
                : Identifier(other), TypeContainer<Variable>(other) {}

            Variable& operator=(const Variable& other) {
                dynamic_cast<Identifier&>(*this) = other;
                dynamic_cast<TypeContainer<Variable>&>(*this) = other;
                return *this;
            }

            virtual const std::string& get_name() const {
                return Identifier::get_name();
            }

            virtual std::ostream& print(std::ostream& o) const {
                return TypeContainer<Variable>::print(o);
            }
        };

    } // namespace pddl

} // namespace airlaps

// Object printing operator
inline std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Variable& v) {
    return v.print(o);
}

#endif // AIRLAPS_PDDL_VARIABLE_HH
