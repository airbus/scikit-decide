/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/duration_formula.hh"

namespace skdecide {

namespace pddl {

DurationFormula::DurationFormula() {}

DurationFormula::DurationFormula(
    const DurationFormula::DurativeActionPtr &durative_action)
    : _durative_action(durative_action) {}

DurationFormula::DurationFormula(const DurationFormula &other)
    : _durative_action(other._durative_action) {}

DurationFormula &DurationFormula::operator=(const DurationFormula &other) {
  this->_durative_action = other._durative_action;
  return *this;
}

DurationFormula::~DurationFormula() {}

void DurationFormula::set_durative_action(
    const DurationFormula::DurativeActionPtr &durative_action) {
  _durative_action = durative_action;
}

const DurationFormula::DurativeActionPtr &
DurationFormula::get_durative_action() const {
  return _durative_action;
}

std::ostream &DurationFormula::print(std::ostream &o) const {
  o << "?duration";
  return o;
}

} // namespace pddl

} // namespace skdecide
