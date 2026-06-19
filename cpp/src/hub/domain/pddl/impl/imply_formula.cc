/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/imply_formula.hh"

namespace skdecide {

namespace pddl {

ImplyFormula::ImplyFormula() {}

ImplyFormula::ImplyFormula(const Formula::Ptr &left_formula,
                           const Formula::Ptr &right_formula)
    : BinaryFormula<ImplyFormula>(left_formula, right_formula) {}

ImplyFormula::ImplyFormula(const ImplyFormula &other)
    : BinaryFormula<ImplyFormula>(other) {}

ImplyFormula &ImplyFormula::operator=(const ImplyFormula &other) {
  dynamic_cast<BinaryFormula<ImplyFormula> &>(*this) = other;
  return *this;
}

ImplyFormula::~ImplyFormula() {}

} // namespace pddl

} // namespace skdecide
