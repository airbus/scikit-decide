#ifndef AIRLAPS_PDDL_DOMAIN_HH
#define AIRLAPS_PDDL_DOMAIN_HH

#include <string>
#include <memory>
#include <unordered_set>

#include "requirements.hh"
#include "type.hh"
#include "object.hh"

namespace airlaps {

    namespace pddl {
        
        class Domain {
        public :
            Domain();
            
            void set_name(const std::string& name);
            const std::string& get_name() const;
            
            void set_requirements(const Requirements& requirements);
            const Requirements& get_requirements() const;

            /**
             * Adds a type.
             * Throws an exception if the given type is already in the set of
             * types of this domain
             */
            const Type::Ptr& add_type(const Type::Ptr& type);
            /**
             * Adds a type.
             * Throws an exception if the given type is already in the set of
             * types of this domain
             */
            const Type::Ptr& add_type(const std::string& type);
            /**
             * Removes a type.
             * Throws an exception if the given type is not in the set of
             * types of this domain
             */
            void remove_type(const Type::Ptr& type);
            /**
             * Removes a type.
             * Throws an exception if the given type is not in the set of
             * types of this domain
             */
            void remove_type(const std::string& type);
            /**
             * Gets a type.
             * Throws an exception if the given type is not in the set of
             * types of this domain
             */
            const Type::Ptr& get_type(const std::string& type) const;
            const Type::Set& get_types() const;

            /**
             * Adds a constant.
             * Throws an exception if the given constant is already in the set of
             * constants of this domain
             */
            const Object::Ptr& add_constant(const Object::Ptr& constant);
            /**
             * Adds a constant.
             * Throws an exception if the given constant is already in the set of
             * constants of this domain
             */
            const Object::Ptr& add_constant(const std::string& constant);
            /**
             * Removes a constant.
             * Throws an exception if the given constant is not in the set of
             * constants of this domain
             */
            void remove_constant(const Object::Ptr& constant);
            /**
             * Removes a constant.
             * Throws an exception if the given constant is not in the set of
             * constants of this domain
             */
            void remove_constant(const std::string& constant);
            /**
             * Gets a constant.
             * Throws an exception if the given constant is not in the set of
             * constants of this domain
             */
            const Object::Ptr& get_constant(const std::string& constant) const;
            const Object::Set& get_constants() const;

            std::string print() const;

        private :
            std::string _name;
            Requirements _requirements;
            Type::Set _types;
            Object::Set _constants;
        };

    } // namespace pddl

} // namespace airlaps

// Requirements printing operator
std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Domain& d);

#endif // AIRLAPS_PDDL_DOMAIN_HH
