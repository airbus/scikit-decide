/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_DERIVED_PREDICATE_CONTAINER_HH
#define SKDECIDE_PDDL_DERIVED_PREDICATE_CONTAINER_HH

#include "associative_container.hh"

namespace skdecide {

    namespace pddl {

        class DerivedPredicate;

        template <typename Derived>
        class DerivedPredicateContainer : public AssociativeContainer<Derived, DerivedPredicate> {
        public :
            typedef typename AssociativeContainer<Derived, DerivedPredicate>::SymbolPtr DerivedPredicatePtr;
            typedef typename AssociativeContainer<Derived, DerivedPredicate>::SymbolSet DerivedPredicateSet;

            DerivedPredicateContainer(const DerivedPredicateContainer& other)
                : AssociativeContainer<Derived, DerivedPredicate>(other) {}
            
            DerivedPredicateContainer& operator=(const DerivedPredicateContainer& other) {
                dynamic_cast<AssociativeContainer<Derived, DerivedPredicate>&>(*this) = other;
                return *this;
            }

            template <typename T>
            inline const DerivedPredicatePtr& add_derived_predicate(const T& derived_predicate) {
                return AssociativeContainer<Derived, DerivedPredicate>::add(derived_predicate);
            }

            template <typename T>
            inline void remove_derived_predicate(const T& derived_predicate) {
                AssociativeContainer<Derived, DerivedPredicate>::remove(derived_predicate);
            }

            template <typename T>
            inline const DerivedPredicatePtr& get_derived_predicate(const T& derived_predicate) const {
                return AssociativeContainer<Derived, DerivedPredicate>::get(derived_predicate);
            }

            inline const DerivedPredicateSet& get_derived_predicates() const {
                return AssociativeContainer<Derived, DerivedPredicate>::get_container();
            }
        
        protected :
            DerivedPredicateContainer() {}
        };     

    } // namespace pddl
    
} // namespace skdecide

#endif // SKDECIDE_PDDL_DERIVED_PREDICATE_CONTAINER_HH
