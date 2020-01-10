/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_CLASS_HH
#define SKDECIDE_PDDL_CLASS_HH

#include "identifier.hh"
#include "function_container.hh"

namespace skdecide {

    namespace pddl {
        
        class Class : public Identifier,
                      public FunctionContainer<Class> {
        public :
            static constexpr char class_name[] = "class";

            typedef std::shared_ptr<Class> Ptr;
            typedef FunctionContainer<Class>::FunctionPtr FunctionPtr;
            typedef FunctionContainer<Class>::FunctionSet FunctionSet;
            
            Class(const std::string& name)
                : Identifier(name) {}

            Class(const Class& other)
                : Identifier(other), FunctionContainer<Class>(other) {}

            Class& operator= (const Class& other) {
                dynamic_cast<Identifier&>(*this) = other;
                dynamic_cast<FunctionContainer<Class>&>(*this) = other;
                return *this;
            }

            virtual ~Class() {}

            std::ostream& print(std::ostream& o) const {
                o << "(:class" << this->_name;
                for (const auto& f : this->_container) {
                    o << " " << *f;
                }
                o << ")";
                return o;
            }
        };

    } // namespace pddl

} // namespace skdecide

// Class printing operator
inline std::ostream& operator<<(std::ostream& o, const skdecide::pddl::Class& c) {
    return c.print(o);
}

#endif // SKDECIDE_PDDL_CLASS_HH
