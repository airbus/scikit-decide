/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_OBJECT_CONTAINER_HH
#define AIRLAPS_PDDL_OBJECT_CONTAINER_HH

#include "associative_container.hh"

namespace airlaps {

    namespace pddl {

        class Object;

        template <typename Derived>
        class ObjectContainer : public AssociativeContainer<Derived, Object> {
        public :
            typedef typename AssociativeContainer<Derived, Object>::SymbolPtr ObjectPtr;
            typedef typename AssociativeContainer<Derived, Object>::SymbolSet ObjectSet;

            ObjectContainer(const ObjectContainer& other)
                : AssociativeContainer<Derived, Object>(other) {}
            
            ObjectContainer& operator=(const ObjectContainer& other) {
                dynamic_cast<AssociativeContainer<Derived, Object>&>(*this) = other;
                return *this;
            }

            template <typename T>
            inline const ObjectPtr& add_object(const T& object) {
                return AssociativeContainer<Derived, Object>::add(object);
            }

            template <typename T>
            inline void remove_object(const T& object) {
                AssociativeContainer<Derived, Object>::remove(object);
            }

            template <typename T>
            inline const ObjectPtr& get_object(const T& object) const {
                return AssociativeContainer<Derived, Object>::get(object);
            }

            inline const ObjectSet& get_objects() const {
                return AssociativeContainer<Derived, Object>::get_container();
            }
        
        protected :
            ObjectContainer() {}
        };     

    } // namespace pddl
    
} // namespace airlaps

#endif // AIRLAPS_PDDL_OBJECT_CONTAINER_HH
