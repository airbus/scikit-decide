#ifndef AIRLAPS_PDDL_OBJECT_CONTAINER_HH
#define AIRLAPS_PDDL_OBJECT_CONTAINER_HH

#include "named_container.hh"

namespace airlaps {

    namespace pddl{

        class Object;

        template <typename Derived>
        class ObjectContainer : public NamedContainer<Derived, Object> {
        public :
            typedef typename NamedContainer<Derived, Object>::SymbolPtr ObjectPtr;
            typedef typename NamedContainer<Derived, Object>::SymbolSet ObjectSet;

            ObjectContainer(const ObjectContainer& other)
                : NamedContainer<Derived, Object>(other) {}
            
            ObjectContainer& operator=(const ObjectContainer& other) {
                dynamic_cast<NamedContainer<Derived, Object>&>(*this) = other;
                return *this;
            }

            template <typename T>
            inline const ObjectPtr& add_object(const T& object) {
                return NamedContainer<Derived, Object>::add(object);
            }

            template <typename T>
            inline void remove_object(const T& object) {
                NamedContainer<Derived, Object>::remove(object);
            }

            template <typename T>
            inline const ObjectPtr& get_object(const T& object) const {
                return NamedContainer<Derived, Object>::get(object);
            }

            inline const ObjectSet& get_objects() const {
                return NamedContainer<Derived, Object>::get_container();
            }
        
        protected :
            ObjectContainer() {}

            ObjectContainer(const std::string& name)
            : NamedContainer<Derived, Object>(name) {}
        };     

    } // namespace pddl
    
} // namespace airlap 

#endif // AIRLAPS_PDDL_OBJECT_CONTAINER_HH
