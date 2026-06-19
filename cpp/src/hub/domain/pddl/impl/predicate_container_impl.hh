/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/predicate_container.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> PredicateContainer<Derived>::PredicateContainer() {}

template <typename Derived>
PredicateContainer<Derived>::PredicateContainer(
    const PredicateContainer<Derived> &other)
    : AssociativeContainer<Derived, Predicate>(other) {}

template <typename Derived>
PredicateContainer<Derived> &PredicateContainer<Derived>::operator=(
    const PredicateContainer<Derived> &other) {
  dynamic_cast<AssociativeContainer<Derived, Predicate> &>(*this) = other;
  return *this;
}

template <typename Derived>
PredicateContainer<Derived>::~PredicateContainer() {}

template <typename Derived>
template <typename T>
const typename PredicateContainer<Derived>::PredicatePtr &
PredicateContainer<Derived>::add_predicate(const T &predicate) {
  return AssociativeContainer<Derived, Predicate>::add(predicate);
}

template <typename Derived>
template <typename T>
void PredicateContainer<Derived>::remove_predicate(const T &predicate) {
  AssociativeContainer<Derived, Predicate>::remove(predicate);
}

template <typename Derived>
template <typename T>
const typename PredicateContainer<Derived>::PredicatePtr &
PredicateContainer<Derived>::get_predicate(const T &predicate) const {
  return AssociativeContainer<Derived, Predicate>::get(predicate);
}

template <typename Derived>
const typename PredicateContainer<Derived>::PredicateSet &
PredicateContainer<Derived>::get_predicates() const {
  return AssociativeContainer<Derived, Predicate>::get_container();
}

} // namespace pddl

} // namespace skdecide
