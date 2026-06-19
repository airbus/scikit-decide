/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/timed_formula.hh"

namespace skdecide {

namespace pddl {

AtStartFormula &AtStartFormula::operator=(const AtStartFormula &other) {
  dynamic_cast<UnaryFormula<AtStartFormula> &>(*this) = other;
  return *this;
}

AtEndFormula &AtEndFormula::operator=(const AtEndFormula &other) {
  dynamic_cast<UnaryFormula<AtEndFormula> &>(*this) = other;
  return *this;
}

OverAllFormula &OverAllFormula::operator=(const OverAllFormula &other) {
  dynamic_cast<UnaryFormula<OverAllFormula> &>(*this) = other;
  return *this;
}

} // namespace pddl

} // namespace skdecide
