/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/object_container.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> ObjectContainer<Derived>::ObjectContainer() {}

template <typename Derived>
ObjectContainer<Derived>::ObjectContainer(const ObjectContainer &other)
    : AssociativeContainer<Derived, Object>(other) {}

template <typename Derived>
ObjectContainer<Derived> &
ObjectContainer<Derived>::operator=(const ObjectContainer &other) {
  dynamic_cast<AssociativeContainer<Derived, Object> &>(*this) = other;
  return *this;
}

template <typename Derived> ObjectContainer<Derived>::~ObjectContainer() {}

template <typename Derived>
template <typename T>
const typename ObjectContainer<Derived>::ObjectPtr &
ObjectContainer<Derived>::add_object(const T &object) {
  return AssociativeContainer<Derived, Object>::add(object);
}

template <typename Derived>
template <typename T>
void ObjectContainer<Derived>::remove_object(const T &object) {
  AssociativeContainer<Derived, Object>::remove(object);
}

template <typename Derived>
template <typename T>
const typename ObjectContainer<Derived>::ObjectPtr &
ObjectContainer<Derived>::get_object(const T &object) const {
  return AssociativeContainer<Derived, Object>::get(object);
}

template <typename Derived>
const typename ObjectContainer<Derived>::ObjectSet &
ObjectContainer<Derived>::get_objects() const {
  return AssociativeContainer<Derived, Object>::get_container();
}

template <typename Derived>
std::ostream &ObjectContainer<Derived>::print(std::ostream &o) const {
  o << "(" << static_cast<const Derived *>(this)->get_name();
  for (const auto &ob : get_objects()) {
    o << " " << *ob;
  }
  o << ")";
  return o;
}

template <typename Derived>
std::string ObjectContainer<Derived>::print() const {
  std::ostringstream o;
  print(o);
  return o.str();
}

} // namespace pddl

} // namespace skdecide
