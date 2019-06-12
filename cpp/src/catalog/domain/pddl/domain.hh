#ifndef AIRLAPS_PDDL_DOMAIN_HH
#define AIRLAPS_PDDL_DOMAIN_HH

#include <string>

#include "requirements.hh"

namespace airlaps {

    namespace pddl {
        
        class Domain {
        public :
            void set_name(const std::string& name);
            const std::string& get_name() const;
            
            void set_requirements(const Requirements& requirements);
            const Requirements& get_requirements() const;

        private :
            std::string _name;
            Requirements _requirements;
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_DOMAIN_HH
