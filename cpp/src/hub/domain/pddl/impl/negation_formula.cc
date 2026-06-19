/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/negation_formula.hh"

namespace skdecide {

namespace pddl {

NegationFormula::NegationFormula() {}

NegationFormula::NegationFormula(const Formula::Ptr &formula)
    : UnaryFormula<NegationFormula>(formula) {}

NegationFormula::NegationFormula(const NegationFormula &other)
    : UnaryFormula<NegationFormula>(other) {}

NegationFormula &NegationFormula::operator=(const NegationFormula &other) {
  dynamic_cast<UnaryFormula<NegationFormula> &>(*this) = other;
  return *this;
}

NegationFormula::~NegationFormula() {}

} // namespace pddl

} // namespace skdecide
