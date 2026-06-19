/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/type_container.hh"

#include <sstream>

namespace skdecide {

namespace pddl {

template <typename Derived>
TypeContainer<Derived> &
TypeContainer<Derived>::operator=(const TypeContainer &other) {
  dynamic_cast<AssociativeContainer<Derived, Type> &>(*this) = other;
  return *this;
}

template <typename Derived>
const typename TypeContainer<Derived>::TypeSet &
TypeContainer<Derived>::get_types() const {
  return AssociativeContainer<Derived, Type>::get_container();
}

template <typename Derived>
std::ostream &TypeContainer<Derived>::print(std::ostream &o) const {
  o << static_cast<const Derived *>(this)->get_name();
  if (!get_types().empty()) {
    o << " - ";
    if (get_types().size() > 1) {
      o << "(either";
      for (const auto &t : get_types()) {
        o << " " << t->get_name();
      }
      o << ")";
    } else {
      o << (*get_types().begin())->get_name();
    }
  }
  return o;
}

template <typename Derived> std::string TypeContainer<Derived>::print() const {
  std::ostringstream o;
  print(o);
  return o.str();
}

} // namespace pddl

} // namespace skdecide
