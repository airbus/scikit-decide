/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/minus_expression.hh"

namespace skdecide {

namespace pddl {

MinusExpression::MinusExpression() {}

MinusExpression::MinusExpression(const Expression::Ptr &expression)
    : UnaryExpression<MinusExpression>(expression) {}

MinusExpression::MinusExpression(const MinusExpression &other)
    : UnaryExpression<MinusExpression>(other) {}

MinusExpression &MinusExpression::operator=(const MinusExpression &other) {
  dynamic_cast<UnaryExpression<MinusExpression> &>(*this) = other;
  return *this;
}

MinusExpression::~MinusExpression() {}

} // namespace pddl

} // namespace skdecide
