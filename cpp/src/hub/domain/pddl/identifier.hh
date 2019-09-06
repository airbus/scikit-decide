#ifndef AIRLAPS_PDDL_IDENTIFIER_HH
#define AIRLAPS_PDDL_IDENTIFIER_HH

#include <string>

namespace airlaps {
    
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
    
} // namespace airlaps

#endif // AIRLAPS_PDDL_IDENTIFIER_HH
