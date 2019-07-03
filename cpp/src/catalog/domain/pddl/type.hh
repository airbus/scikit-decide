#ifndef AIRLAPS_PDDL_TYPE_HH
#define AIRLAPS_PDDL_TYPE_HH

#include <string>
#include <unordered_set>
#include <memory>
#include <ostream>

namespace airlaps {

    namespace pddl {

        class Type {
        public :
            typedef std::shared_ptr<Type> Ptr;

            struct Hash {
                inline std::size_t operator()(const Ptr& t) const {
                    return std::hash<std::string>()(t->get_name());
                }
            };

            struct Equal {
                inline bool operator()(const Ptr& t1, const Ptr& t2) const {
                    return std::equal_to<std::string>()(t1->get_name(), t2->get_name());
                }
            };

            typedef std::unordered_set<Ptr, Hash, Equal> Set;


            Type(const std::string& name);
            Type(const Type& other);
            Type& operator=(const Type& other);

            const std::string& get_name() const;

            /**
             * Adds a parent type.
             * Throws an exception if the given type is already in the set of
             * parent types
             */
            void add_parent(const Ptr& type);
            /**
             * Remove a parent type.
             * Throws an exception if the given type is not in the set of parent
             * types
             */
            void remove_parent(const Ptr& type);
            /**
             * Gets a parent type.
             * Throws an exception if the given type is not in the set of parent
             * types
             */
            const Ptr& get_parent(const std::string& t) const;
            const Set& get_parents() const;

            std::string print() const;

            static const Ptr& object() { return _object; }
            static const Ptr& number() { return _number; }

        private :
            std::string _name;
            Set _parents;

            static const Ptr _object;
            static const Ptr _number;
        };

    } // namespace pddl

} // namespace airlaps

// Type printing operator
std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Type& t);

#endif // AIRLAPS_PDDL_TYPE_HH
