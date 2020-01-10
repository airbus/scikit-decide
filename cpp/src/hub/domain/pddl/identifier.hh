/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_IDENTIFIER_HH
#define SKDECIDE_PDDL_IDENTIFIER_HH

#include <string>

namespace skdecide {
    
    namespace pddl {

        class Identifier {
        public :
            Identifier(const Identifier& other)
                : _name(other._name) {}
            
            Identifier& operator=(const Identifier& other) {
                this->_name = other._name;
                return *this;
            }

            const std::string& get_name() const {
                return _name;
            }

        protected :
            std::string _name;

            Identifier() {}

            Identifier(const std::string& name)
                : _name(name) {}
            
            void set_name(const std::string& name) {
                _name = name;
            }
        };
        
    } // namespace pddl
    
} // namespace skdecide

#endif // SKDECIDE_PDDL_IDENTIFIER_HH
