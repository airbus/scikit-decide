#ifndef AIRLAPS_PDDL_TYPE_HH
#define AIRLAPS_PDDL_TYPE_HH

#include <string>
#include <unordered_set>
#include <ostream>

namespace airlaps {

    namespace pddl {

        class Type {
        public :
            struct Hash {
                inline std::size_t operator()(const Type& t) const {
                    return std::hash<std::string>()(t.get_name());
                }
                inline std::size_t operator()(const Type* t) const {
                    return std::hash<std::string>()(t->get_name());
                }
            };

            struct Equal {
                inline bool operator()(const Type& t1, const Type& t2) const {
                    return std::equal_to<std::string>()(t1.get_name(), t2.get_name());
                }
                inline bool operator()(const Type* t1, const Type* t2) const {
                    return std::equal_to<std::string>()(t1->get_name(), t2->get_name());
                }
            };

            Type(const std::string& name);
            Type(const Type& other);
            Type& operator=(const Type& other);


            const std::string& get_name() const;

            typedef std::unordered_set<const Type*, Hash, Equal> ParentSet;
            /**
             * Adds a parent type.
             * Throws an exception if the given type is already in the set of
             * parent types
             */
            void add_parent(const Type& type);
            /**
             * Remove a parent type.
             * Throws an exception if the given type is not in the set of parent
             * types
             */
            void remove_parent(const Type& type);
            const ParentSet& get_parents() const;

            std::string print() const;

        private :
            std::string _name;
            ParentSet _parents;
        };

    } // namespace pddl

} // namespace airlaps

// Type printing operator
std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Type& t);

#endif // AIRLAPS_PDDL_TYPE_HH
