/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/preference_container.hh"

#include <sstream>

namespace skdecide {

namespace pddl {

template <typename Derived>
PreferenceContainer<Derived> &
PreferenceContainer<Derived>::operator=(const PreferenceContainer &other) {
  dynamic_cast<AssociativeContainer<Derived, Preference> &>(*this) = other;
  return *this;
}

template <typename Derived>
const typename PreferenceContainer<Derived>::PreferenceSet &
PreferenceContainer<Derived>::get_preferences() const {
  return AssociativeContainer<Derived, Preference>::get_container();
}

template <typename Derived>
std::ostream &PreferenceContainer<Derived>::print(std::ostream &o) const {
  o << "(" << static_cast<const Derived *>(this)->get_name();
  for (const auto &ob : get_preferences()) {
    o << " " << *ob;
  }
  o << ")";
  return o;
}

template <typename Derived>
std::string PreferenceContainer<Derived>::print() const {
  std::ostringstream o;
  print(o);
  return o.str();
}

} // namespace pddl

} // namespace skdecide
