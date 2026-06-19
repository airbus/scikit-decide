/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/operation_expression.hh"

namespace skdecide {

namespace pddl {

// === AddExpression implementation ===

AddExpression::AddExpression() {}

AddExpression::AddExpression(const Expression::Ptr &left_expression,
                             const Expression::Ptr &right_expression)
    : BinaryExpression<AddExpression>(left_expression, right_expression) {}

AddExpression::AddExpression(const AddExpression &other)
    : BinaryExpression<AddExpression>(other) {}

AddExpression &AddExpression::operator=(const AddExpression &other) {
  dynamic_cast<BinaryExpression<AddExpression> &>(*this) = other;
  return *this;
}

AddExpression::~AddExpression() {}

// === SubExpression implementation ===

SubExpression::SubExpression() {}

SubExpression::SubExpression(const Expression::Ptr &left_expression,
                             const Expression::Ptr &right_expression)
    : BinaryExpression<SubExpression>(left_expression, right_expression) {}

SubExpression::SubExpression(const SubExpression &other)
    : BinaryExpression<SubExpression>(other) {}

SubExpression &SubExpression::operator=(const SubExpression &other) {
  dynamic_cast<BinaryExpression<SubExpression> &>(*this) = other;
  return *this;
}

SubExpression::~SubExpression() {}

// === MulExpression implementation ===

MulExpression::MulExpression() {}

MulExpression::MulExpression(const Expression::Ptr &left_expression,
                             const Expression::Ptr &right_expression)
    : BinaryExpression<MulExpression>(left_expression, right_expression) {}

MulExpression::MulExpression(const MulExpression &other)
    : BinaryExpression<MulExpression>(other) {}

MulExpression &MulExpression::operator=(const MulExpression &other) {
  dynamic_cast<BinaryExpression<MulExpression> &>(*this) = other;
  return *this;
}

MulExpression::~MulExpression() {}

// === DivExpression implementation ===

DivExpression::DivExpression() {}

DivExpression::DivExpression(const Expression::Ptr &left_expression,
                             const Expression::Ptr &right_expression)
    : BinaryExpression<DivExpression>(left_expression, right_expression) {}

DivExpression::DivExpression(const DivExpression &other)
    : BinaryExpression<DivExpression>(other) {}

DivExpression &DivExpression::operator=(const DivExpression &other) {
  dynamic_cast<BinaryExpression<DivExpression> &>(*this) = other;
  return *this;
}

DivExpression::~DivExpression() {}

} // namespace pddl

} // namespace skdecide
