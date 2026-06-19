/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/quantified_formula.hh"

namespace skdecide {

namespace pddl {

// === UniversalFormula implementation ===

UniversalFormula::UniversalFormula() {}

UniversalFormula::UniversalFormula(
    const Formula::Ptr &formula,
    const VariableContainer<UniversalFormula> &variables)
    : QuantifiedFormula<UniversalFormula>(formula, variables) {}

UniversalFormula::UniversalFormula(const UniversalFormula &other)
    : QuantifiedFormula<UniversalFormula>(other) {}

UniversalFormula &UniversalFormula::operator=(const UniversalFormula &other) {
  dynamic_cast<QuantifiedFormula<UniversalFormula> &>(*this) = other;
  return *this;
}

UniversalFormula::~UniversalFormula() {}

// === ExistentialFormula implementation ===

ExistentialFormula::ExistentialFormula() {}

ExistentialFormula::ExistentialFormula(
    const Formula::Ptr &formula,
    const VariableContainer<ExistentialFormula> &variables)
    : QuantifiedFormula<ExistentialFormula>(formula, variables) {}

ExistentialFormula::ExistentialFormula(const ExistentialFormula &other)
    : QuantifiedFormula<ExistentialFormula>(other) {}

ExistentialFormula &
ExistentialFormula::operator=(const ExistentialFormula &other) {
  dynamic_cast<QuantifiedFormula<ExistentialFormula> &>(*this) = other;
  return *this;
}

ExistentialFormula::~ExistentialFormula() {}

} // namespace pddl

} // namespace skdecide
