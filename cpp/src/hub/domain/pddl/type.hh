#ifndef AIRLAPS_PDDL_TYPE_HH
#define AIRLAPS_PDDL_TYPE_HH

#include <ostream>

#include "identifier.hh"
#include "type_container.hh"

namespace airlaps {

    namespace pddl {

        class Type : public Identifier,
                     public TypeContainer<Type> {
        public :
            static constexpr char class_name[] = "type";

            typedef AssociativeContainer<Type, Type>::SymbolPtr Ptr;
            typedef AssociativeContainer<Type, Type>::SymbolSet Set;

            Type(const std::string& name);
            Type(const Type& other);
            Type& operator=(const Type& other);

            static const Ptr& object() { return _object; }
            static const Ptr& number() { return _number; }

        private :
            static const Ptr _object;
            static const Ptr _number;
        };

    } // namespace pddl

} // namespace airlaps

// Type printing operator
std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Type& t);

#endif // AIRLAPS_PDDL_TYPE_HH
