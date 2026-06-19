/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/optimization_expression.hh"

namespace skdecide {

namespace pddl {

// === MinimizeExpression implementation ===

MinimizeExpression::MinimizeExpression() {}

MinimizeExpression::MinimizeExpression(const Expression::Ptr &expression)
    : UnaryExpression<MinimizeExpression>(expression) {}

MinimizeExpression::MinimizeExpression(const MinimizeExpression &other)
    : UnaryExpression<MinimizeExpression>(other) {}

MinimizeExpression &
MinimizeExpression::operator=(const MinimizeExpression &other) {
  dynamic_cast<UnaryExpression<MinimizeExpression> &>(*this) = other;
  return *this;
}

MinimizeExpression::~MinimizeExpression() {}

// === MaximizeExpression implementation ===

MaximizeExpression::MaximizeExpression() {}

MaximizeExpression::MaximizeExpression(const Expression::Ptr &expression)
    : UnaryExpression<MaximizeExpression>(expression) {}

MaximizeExpression::MaximizeExpression(const MaximizeExpression &other)
    : UnaryExpression<MaximizeExpression>(other) {}

MaximizeExpression &
MaximizeExpression::operator=(const MaximizeExpression &other) {
  dynamic_cast<UnaryExpression<MaximizeExpression> &>(*this) = other;
  return *this;
}

MaximizeExpression::~MaximizeExpression() {}

} // namespace pddl

} // namespace skdecide
