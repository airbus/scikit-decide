/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_CLASS_CONTAINER_HH
#define AIRLAPS_PDDL_CLASS_CONTAINER_HH

#include "associative_container.hh"

namespace airlaps {

    namespace pddl {

        class Class;

        template <typename Derived>
        class ClassContainer : public AssociativeContainer<Derived, Class> {
        public :
            typedef typename AssociativeContainer<Derived, Class>::SymbolPtr ClassPtr;
            typedef typename AssociativeContainer<Derived, Class>::SymbolSet ClassSet;

            ClassContainer(const ClassContainer& other)
                : AssociativeContainer<Derived, Class>(other) {}
            
            ClassContainer& operator=(const ClassContainer& other) {
                dynamic_cast<AssociativeContainer<Derived, Class>&>(*this) = other;
                return *this;
            }

            template <typename T>
            inline const ClassPtr& add_class(const T& cls) {
                return AssociativeContainer<Derived, Class>::add(cls);
            }

            template <typename T>
            inline void remove_class(const T& cls) {
                AssociativeContainer<Derived, Class>::remove(cls);
            }

            template <typename T>
            inline const ClassPtr& get_class(const T& cls) const {
                return AssociativeContainer<Derived, Class>::get(cls);
            }

            inline const ClassSet& get_classes() const {
                return AssociativeContainer<Derived, Class>::get_container();
            }
        
        protected :
            ClassContainer() {}
        };     

    } // namespace pddl
    
} // namespace airlaps

#endif // AIRLAPS_PDDL_CLASS_CONTAINER_HH
