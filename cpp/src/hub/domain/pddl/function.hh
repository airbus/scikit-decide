/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_FUNCTION_HH
#define AIRLAPS_PDDL_FUNCTION_HH

#include "identifier.hh"
#include "variable_container.hh"

namespace airlaps {

    namespace pddl {

        class Function : public Identifier,
                         public VariableContainer<Function> {
        public :
            static constexpr char class_name[] = "function";

            typedef std::shared_ptr<Function> Ptr;

            Function(const std::string& name)
                : Identifier(name) {}

            Function(const Function& other)
                : Identifier(other), VariableContainer<Function>(other) {}

            Function& operator=(const Function& other) {
                dynamic_cast<Identifier&>(*this) = other;
                dynamic_cast<VariableContainer<Function>&>(*this) = other;
                return *this;
            }

            typedef VariableContainer<Function>::VariablePtr VariablePtr;
            typedef VariableContainer<Function>::VariableVector VariableVector;
        };

    } // namespace pddl

} // namespace airlaps

// Object printing operator
inline std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Function& f) {
    return f.print(o);
}

#endif // AIRLAPS_PDDL_FUNCTION_HH
