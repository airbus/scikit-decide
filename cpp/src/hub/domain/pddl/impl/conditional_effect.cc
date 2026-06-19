/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/conditional_effect.hh"

namespace skdecide {

namespace pddl {

ConditionalEffect::ConditionalEffect() {}

ConditionalEffect::ConditionalEffect(const Formula::Ptr &condition,
                                     const Effect::Ptr &effect)
    : BinaryEffect(condition, effect) {}

ConditionalEffect::ConditionalEffect(const ConditionalEffect &other)
    : BinaryEffect(other) {}

ConditionalEffect &
ConditionalEffect::operator=(const ConditionalEffect &other) {
  dynamic_cast<BinaryEffect &>(*this) = other;
  return *this;
}

ConditionalEffect::~ConditionalEffect() {}

std::ostream &ConditionalEffect::print(std::ostream &o) const {
  o << "(when " << *_condition << " " << *_effect << ")";
  return o;
}

} // namespace pddl

} // namespace skdecide
