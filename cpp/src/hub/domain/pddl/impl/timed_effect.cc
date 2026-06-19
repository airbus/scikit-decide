/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/timed_effect.hh"

namespace skdecide {

namespace pddl {

AtStartEffect &AtStartEffect::operator=(const AtStartEffect &other) {
  dynamic_cast<UnaryEffect<AtStartEffect> &>(*this) = other;
  return *this;
}

AtEndEffect &AtEndEffect::operator=(const AtEndEffect &other) {
  dynamic_cast<UnaryEffect<AtEndEffect> &>(*this) = other;
  return *this;
}

AtTimeEffect &AtTimeEffect::operator=(const AtTimeEffect &other) {
  dynamic_cast<UnaryEffect<AtTimeEffect> &>(*this) = other;
  this->_time = other._time;
  return *this;
}

std::ostream &AtTimeEffect::print(std::ostream &o) const {
  o << "(at " << *_time << " " << *_effect << ")";
  return o;
}

} // namespace pddl

} // namespace skdecide
