/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/predicate.hh"

namespace skdecide {

namespace pddl {

Predicate &Predicate::operator=(const Predicate &other) {
  dynamic_cast<Identifier &>(*this) = other;
  dynamic_cast<VariableContainer<Predicate> &>(*this) = other;
  return *this;
}

std::ostream &operator<<(std::ostream &o, const Predicate &p) {
  return p.print(o);
}

} // namespace pddl

} // namespace skdecide
