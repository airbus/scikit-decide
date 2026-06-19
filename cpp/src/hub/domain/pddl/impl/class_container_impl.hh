/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/class_container.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> ClassContainer<Derived>::ClassContainer() {}

template <typename Derived>
ClassContainer<Derived>::ClassContainer(const ClassContainer &other)
    : AssociativeContainer<Derived, Class>(other) {}

template <typename Derived>
ClassContainer<Derived> &
ClassContainer<Derived>::operator=(const ClassContainer<Derived> &other) {
  dynamic_cast<AssociativeContainer<Derived, Class> &>(*this) = other;
  return *this;
}

template <typename Derived> ClassContainer<Derived>::~ClassContainer() {}

template <typename Derived>
template <typename T>
inline const typename ClassContainer<Derived>::ClassPtr &
ClassContainer<Derived>::add_class(const T &cls) {
  return AssociativeContainer<Derived, Class>::add(cls);
}

template <typename Derived>
template <typename T>
void ClassContainer<Derived>::remove_class(const T &cls) {
  AssociativeContainer<Derived, Class>::remove(cls);
}

template <typename Derived>
template <typename T>
const typename ClassContainer<Derived>::ClassPtr &
ClassContainer<Derived>::get_class(const T &cls) const {
  return AssociativeContainer<Derived, Class>::get(cls);
}

template <typename Derived>
const typename ClassContainer<Derived>::ClassSet &
ClassContainer<Derived>::get_classes() const {
  return AssociativeContainer<Derived, Class>::get_container();
}

} // namespace pddl

} // namespace skdecide
