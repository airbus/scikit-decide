/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/variable_container.hh"

#include <sstream>

namespace skdecide {

namespace pddl {

template <typename Derived>
VariableContainer<Derived> &
VariableContainer<Derived>::operator=(const VariableContainer &other) {
  dynamic_cast<SequenceContainer<Derived, Variable> &>(*this) = other;
  return *this;
}

template <typename Derived>
const typename VariableContainer<Derived>::VariablePtr &
VariableContainer<Derived>::variable_at(const std::size_t &index) const {
  return SequenceContainer<Derived, Variable>::at(index);
}

template <typename Derived>
const typename VariableContainer<Derived>::VariableVector &
VariableContainer<Derived>::get_variables() const {
  return SequenceContainer<Derived, Variable>::get_container();
}

template <typename Derived>
std::ostream &VariableContainer<Derived>::print(std::ostream &o) const {
  o << "(" << static_cast<const Derived *>(this)->get_name();
  for (const auto &v : get_variables()) {
    o << " " << *v;
  }
  o << ")";
  return o;
}

template <typename Derived>
std::string VariableContainer<Derived>::print() const {
  std::ostringstream o;
  print(o);
  return o.str();
}

} // namespace pddl

} // namespace skdecide
