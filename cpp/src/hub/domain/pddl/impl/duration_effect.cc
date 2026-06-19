/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/duration_effect.hh"

namespace skdecide {

namespace pddl {

DurationEffect::DurationEffect() {}

DurationEffect::DurationEffect(
    const DurationEffect::DurativeActionPtr &durative_action)
    : _durative_action(durative_action) {}

DurationEffect::DurationEffect(const DurationEffect &other)
    : _durative_action(other._durative_action) {}

DurationEffect &DurationEffect::operator=(const DurationEffect &other) {
  this->_durative_action = other._durative_action;
  return *this;
}

DurationEffect::~DurationEffect() {}

void DurationEffect::set_durative_action(
    const DurationEffect::DurativeActionPtr &durative_action) {
  _durative_action = durative_action;
}

const DurationEffect::DurativeActionPtr &
DurationEffect::get_durative_action() const {
  return _durative_action;
}

std::ostream &DurationEffect::print(std::ostream &o) const {
  o << "?duration";
  return o;
}

} // namespace pddl

} // namespace skdecide
