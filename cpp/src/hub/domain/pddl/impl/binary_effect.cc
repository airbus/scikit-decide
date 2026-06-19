/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/binary_effect.hh"

namespace skdecide {

namespace pddl {

BinaryEffect::BinaryEffect() {}

BinaryEffect::BinaryEffect(const Formula::Ptr &condition,
                           const Effect::Ptr &effect)
    : _condition(condition), _effect(effect) {}

BinaryEffect::BinaryEffect(const BinaryEffect &other)
    : _condition(other._condition), _effect(other._effect) {}

BinaryEffect &BinaryEffect::operator=(const BinaryEffect &other) {
  this->_condition = other._condition;
  this->_effect = other._effect;
  return *this;
}

BinaryEffect::~BinaryEffect() {}

void BinaryEffect::set_condition(const Formula::Ptr &condition) {
  _condition = condition;
}

const Formula::Ptr &BinaryEffect::get_condition() const { return _condition; }

void BinaryEffect::set_effect(const Effect::Ptr &effect) { _effect = effect; }

const Effect::Ptr &BinaryEffect::get_effect() const { return _effect; }

} // namespace pddl

} // namespace skdecide
