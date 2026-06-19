/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/variable.hh"

namespace skdecide {

namespace pddl {

Variable &Variable::operator=(const Variable &other) {
  dynamic_cast<Identifier &>(*this) = other;
  dynamic_cast<TypeContainer<Variable> &>(*this) = other;
  return *this;
}

std::ostream &Variable::print(std::ostream &o) const {
  return TypeContainer<Variable>::print(o);
}

std::ostream &operator<<(std::ostream &o, const Variable &v) {
  return v.print(o);
}

} // namespace pddl

} // namespace skdecide
