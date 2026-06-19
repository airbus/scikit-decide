/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/quantified_effect.hh"

namespace skdecide {

namespace pddl {

// === UniversalEffect implementation ===

UniversalEffect::UniversalEffect() {}

UniversalEffect::UniversalEffect(
    const Effect::Ptr &effect,
    const VariableContainer<UniversalEffect> &variables)
    : QuantifiedEffect<UniversalEffect>(effect, variables) {}

UniversalEffect::UniversalEffect(const UniversalEffect &other)
    : QuantifiedEffect<UniversalEffect>(other) {}

UniversalEffect &UniversalEffect::operator=(const UniversalEffect &other) {
  dynamic_cast<QuantifiedEffect<UniversalEffect> &>(*this) = other;
  return *this;
}

UniversalEffect::~UniversalEffect() {}

// === ExistentialEffect implementation ===

ExistentialEffect::ExistentialEffect() {}

ExistentialEffect::ExistentialEffect(
    const Effect::Ptr &effect,
    const VariableContainer<ExistentialEffect> &variables)
    : QuantifiedEffect<ExistentialEffect>(effect, variables) {}

ExistentialEffect::ExistentialEffect(const ExistentialEffect &other)
    : QuantifiedEffect<ExistentialEffect>(other) {}

ExistentialEffect &
ExistentialEffect::operator=(const ExistentialEffect &other) {
  dynamic_cast<QuantifiedEffect<ExistentialEffect> &>(*this) = other;
  return *this;
}

ExistentialEffect::~ExistentialEffect() {}

} // namespace pddl

} // namespace skdecide
