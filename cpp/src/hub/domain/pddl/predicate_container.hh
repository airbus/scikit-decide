/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_PREDICATE_CONTAINER_HH
#define AIRLAPS_PDDL_PREDICATE_CONTAINER_HH

#include "associative_container.hh"

namespace airlaps {

    namespace pddl {

        class Predicate;

        template <typename Derived>
        class PredicateContainer : public AssociativeContainer<Derived, Predicate> {
        public :
            typedef typename AssociativeContainer<Derived, Predicate>::SymbolPtr PredicatePtr;
            typedef typename AssociativeContainer<Derived, Predicate>::SymbolSet PredicateSet;

            PredicateContainer(const PredicateContainer& other)
                : AssociativeContainer<Derived, Predicate>(other) {}
            
            PredicateContainer& operator=(const PredicateContainer& other) {
                dynamic_cast<AssociativeContainer<Derived, Predicate>&>(*this) = other;
                return *this;
            }

            template <typename T>
            inline const PredicatePtr& add_predicate(const T& predicate) {
                return AssociativeContainer<Derived, Predicate>::add(predicate);
            }

            template <typename T>
            inline void remove_predicate(const T& predicate) {
                AssociativeContainer<Derived, Predicate>::remove(predicate);
            }

            template <typename T>
            inline const PredicatePtr& get_predicate(const T& predicate) const {
                return AssociativeContainer<Derived, Predicate>::get(predicate);
            }

            inline const PredicateSet& get_predicates() const {
                return AssociativeContainer<Derived, Predicate>::get_container();
            }
        
        protected :
            PredicateContainer() {}
        };     

    } // namespace pddl
    
} // namespace airlaps

#endif // AIRLAPS_PDDL_PREDICATE_CONTAINER_HH
