#ifndef AIRLAPS_PDDL_TYPE_CONTAINER_HH
#define AIRLAPS_PDDL_TYPE_CONTAINER_HH

#include "named_container.hh"

namespace airlaps {

    namespace pddl{

        class Type;

        template <typename Derived>
        class TypeContainer : public NamedContainer<Derived, Type> {
        public :
            typedef typename NamedContainer<Derived, Type>::SymbolPtr TypePtr;
            typedef typename NamedContainer<Derived, Type>::SymbolSet TypeSet;

            TypeContainer(const TypeContainer& other)
                : NamedContainer<Derived, Type>(other) {}
            
            TypeContainer& operator=(const TypeContainer& other) {
                dynamic_cast<NamedContainer<Derived, Type>&>(*this) = other;
                return *this;
            }

            template <typename T>
            inline const TypePtr& add_type(const T& type) {
                return NamedContainer<Derived, Type>::add(type);
            }

            template <typename T>
            inline void remove_type(const T& type) {
                NamedContainer<Derived, Type>::remove(type);
            }

            template <typename T>
            inline const TypePtr& get_type(const T& type) const {
                return NamedContainer<Derived, Type>::get(type);
            }

            inline const TypeSet& get_types() const {
                return NamedContainer<Derived, Type>::get_container();
            }

            std::ostream& print(std::ostream& o) const {
                o << this->get_name();
                if (!get_types().empty()) {
                    o << " - ";
                    if (get_types().size() > 1) {
                        o << "(either";
                        for (const auto& t : get_types()) {
                            o << " " << t->get_name();
                        }
                        o << ")";
                    } else {
                        o << (*get_types().begin())->get_name();
                    }
                }
                return o;
            }

            std::string print() const {
                std::ostringstream o;
                print(o);
                return o.str();
            }
        
        protected :
            TypeContainer() {}

            TypeContainer(const std::string& name)
            : NamedContainer<Derived, Type>(name) {}
        };     

    } // namespace pddl
    
} // namespace airlap 

#endif // AIRLAPS_PDDL_TYPE_CONTAINER_HH
