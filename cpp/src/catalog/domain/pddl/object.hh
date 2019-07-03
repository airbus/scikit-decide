#ifndef AIRLAPS_PDDL_OBJECT_HH
#define AIRLAPS_PDDL_OBJECT_HH

#include <string>
#include <unordered_set>
#include <memory>
#include <ostream>

#include "type.hh"

namespace airlaps {

    namespace pddl {

        class Object {
        public :
            typedef std::shared_ptr<Object> Ptr;

            struct Hash {
                inline std::size_t operator()(const Ptr& o) const {
                    return std::hash<std::string>()(o->get_name());
                }
            };

            struct Equal {
                inline bool operator()(const Ptr& o1, const Ptr& o2) const {
                    return std::equal_to<std::string>()(o1->get_name(), o2->get_name());
                }
            };

            typedef std::unordered_set<Ptr, Hash, Equal> Set;
            

            Object(const std::string& name);
            Object(const Object& other);
            Object& operator=(const Object& other);

            const std::string& get_name() const;

            /**
             * Adds a type.
             * Throws an exception if the given type is already in the set of
             * types of this object
             */
            void add_type(const Type::Ptr& type);
            /**
             * Removes a type.
             * Throws an exception if the given type is not in the set of
             * types of this object
             */
            void remove_type(const Type::Ptr& type);
            /**
             * Gets a type.
             * Throws an exception if the given type is not in the set of
             * types of this object
             */
            const Type::Ptr& get_type(const std::string& t) const;
            const Type::Set& get_types() const;

            std::string print() const;

        private :
            std::string _name;
            Type::Set _types;
        };

    } // namespace pddl

} // namespace airlaps

// Type printing operator
std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Object& ob);

#endif // AIRLAPS_PDDL_OBJECT_HH
