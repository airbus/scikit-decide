/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/negation_effect.hh"

namespace skdecide {

namespace pddl {

NegationEffect::NegationEffect() {}

NegationEffect::NegationEffect(const PredicateEffect::Ptr &effect)
    : UnaryEffect<NegationEffect, PredicateEffect>(effect) {}

NegationEffect::NegationEffect(const NegationEffect &other)
    : UnaryEffect<NegationEffect, PredicateEffect>(other) {}

NegationEffect &NegationEffect::operator=(const NegationEffect &other) {
  dynamic_cast<UnaryEffect<NegationEffect, PredicateEffect> &>(*this) = other;
  return *this;
}

NegationEffect::~NegationEffect() {}

} // namespace pddl

} // namespace skdecide
