#ifndef AIRLAPS_PDDL_DOMAIN_HH
#define AIRLAPS_PDDL_DOMAIN_HH

#include <string>
#include <memory>
#include <unordered_set>

#include "identifier.hh"
#include "requirements.hh"
#include "type_container.hh"
#include "object_container.hh"

namespace airlaps {

    namespace pddl {
        
        class Domain : public Identifier,
                       public TypeContainer<Domain>,
                       public ObjectContainer<Domain> {
        public :
            static constexpr char class_name[] = "domain";
            
            Domain();
            
            void set_name(const std::string& name);
            
            void set_requirements(const Requirements& requirements);
            const Requirements& get_requirements() const;

            std::string print() const;

            typedef TypeContainer<Domain>::TypePtr TypePtr;
            typedef TypeContainer<Domain>::TypeSet TypeSet;
            typedef ObjectContainer<Domain>::ObjectPtr ObjectPtr;
            typedef ObjectContainer<Domain>::ObjectSet ObjectSet;

        private :
            Requirements _requirements;
        };

    } // namespace pddl

} // namespace airlaps

// Requirements printing operator
std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Domain& d);

#endif // AIRLAPS_PDDL_DOMAIN_HH
