#ifndef AIRLAPS_PDDL_DOMAIN_HH
#define AIRLAPS_PDDL_DOMAIN_HH

#include <string>
#include <unordered_set>

#include "requirements.hh"
#include "type.hh"

namespace airlaps {

    namespace pddl {
        
        class Domain {
        public :
            void set_name(const std::string& name);
            const std::string& get_name() const;
            
            void set_requirements(const Requirements& requirements);
            const Requirements& get_requirements() const;

            typedef std::unordered_set<Type, Type::Hash, Type::Equal> TypeSet;
            /**
             * Adds a type.
             * Throws an exception if the given type is already in the set of
             * types of this domain
             */
            void add_type(const Type& type);
            /**
             * Adds a type.
             * Throws an exception if the given type is already in the set of
             * types of this domain
             */
            Type& add_type(const std::string& type);
            /**
             * Removes a type.
             * Throws an exception if the given type is not in the set of
             * types of this domain
             */
            void remove_type(const Type& type);
            /**
             * Removes a type.
             * Throws an exception if the given type is not in the set of
             * types of this domain
             */
            void remove_type(const std::string& type);
            const TypeSet& get_types() const;

            std::string print() const;

        private :
            std::string _name;
            Requirements _requirements;
            TypeSet _types;
        };

    } // namespace pddl

} // namespace airlaps

// Requirements printing operator
std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Domain& d);

#endif // AIRLAPS_PDDL_DOMAIN_HH
