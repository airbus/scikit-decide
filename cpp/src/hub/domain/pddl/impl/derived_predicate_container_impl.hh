/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/derived_predicate_container.hh"

namespace skdecide {

namespace pddl {

template <typename Derived>
DerivedPredicateContainer<Derived>::DerivedPredicateContainer() {}

template <typename Derived>
DerivedPredicateContainer<Derived>::DerivedPredicateContainer(
    const DerivedPredicateContainer<Derived> &other)
    : AssociativeContainer<Derived, DerivedPredicate>(other) {}

template <typename Derived>
DerivedPredicateContainer<Derived> &
DerivedPredicateContainer<Derived>::operator=(
    const DerivedPredicateContainer<Derived> &other) {
  dynamic_cast<AssociativeContainer<Derived, DerivedPredicate> &>(*this) =
      other;
  return *this;
}

template <typename Derived>
DerivedPredicateContainer<Derived>::~DerivedPredicateContainer() {}

template <typename Derived>
template <typename T>
const typename DerivedPredicateContainer<Derived>::DerivedPredicatePtr &
DerivedPredicateContainer<Derived>::add_derived_predicate(
    const T &derived_predicate) {
  return AssociativeContainer<Derived, DerivedPredicate>::add(
      derived_predicate);
}

template <typename Derived>
template <typename T>
void DerivedPredicateContainer<Derived>::remove_derived_predicate(
    const T &derived_predicate) {
  AssociativeContainer<Derived, DerivedPredicate>::remove(derived_predicate);
}

template <typename Derived>
template <typename T>
const typename DerivedPredicateContainer<Derived>::DerivedPredicatePtr &
DerivedPredicateContainer<Derived>::get_derived_predicate(
    const T &derived_predicate) const {
  return AssociativeContainer<Derived, DerivedPredicate>::get(
      derived_predicate);
}

template <typename Derived>
const typename DerivedPredicateContainer<Derived>::DerivedPredicateSet &
DerivedPredicateContainer<Derived>::get_derived_predicates() const {
  return AssociativeContainer<Derived, DerivedPredicate>::get_container();
}

} // namespace pddl

} // namespace skdecide
