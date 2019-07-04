#ifndef AIRLAPS_PDDL_DOMAIN_HH
#define AIRLAPS_PDDL_DOMAIN_HH

#include <string>
#include <memory>
#include <unordered_set>

#include "requirements.hh"
#include "type_container.hh"
#include "object_container.hh"

namespace airlaps {

    namespace pddl {
        
        class Domain : public TypeContainer<Domain>,
                       public ObjectContainer<Domain> {
        public :
            static constexpr char cls_name[] = "domain";
            
            Domain();
            
            void set_name(const std::string& name);
            const std::string& get_name() const;
            
            void set_requirements(const Requirements& requirements);
            const Requirements& get_requirements() const;

            std::string print() const;

            typedef TypeContainer<Domain>::TypePtr TypePtr;
            typedef TypeContainer<Domain>::TypeSet TypeSet;
            typedef ObjectContainer<Domain>::ObjectPtr ObjectPtr;
            typedef ObjectContainer<Domain>::ObjectSet ObjectSet;

        private :
            std::string _name;
            Requirements _requirements;
        };

    } // namespace pddl

} // namespace airlaps

// Requirements printing operator
std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Domain& d);

#endif // AIRLAPS_PDDL_DOMAIN_HH
