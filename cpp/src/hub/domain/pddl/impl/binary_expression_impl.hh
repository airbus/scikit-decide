/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/binary_expression.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> BinaryExpression<Derived>::BinaryExpression() {}

template <typename Derived>
BinaryExpression<Derived>::BinaryExpression(
    const Expression::Ptr &left_expression,
    const Expression::Ptr &right_expression)
    : _left_expression(left_expression), _right_expression(right_expression) {}

template <typename Derived>
BinaryExpression<Derived>::BinaryExpression(
    const BinaryExpression<Derived> &other)
    : _left_expression(other._left_expression),
      _right_expression(other._right_expression) {}

template <typename Derived>
BinaryExpression<Derived> &
BinaryExpression<Derived>::operator=(const BinaryExpression<Derived> &other) {
  this->_left_expression = other._left_expression;
  this->_right_expression = other._right_expression;
  return *this;
}

template <typename Derived> BinaryExpression<Derived>::~BinaryExpression() {}

template <typename Derived>
void BinaryExpression<Derived>::set_left_expression(
    const Expression::Ptr &expression) {
  _left_expression = expression;
}

template <typename Derived>
const Expression::Ptr &BinaryExpression<Derived>::get_left_expression() const {
  return _left_expression;
}

template <typename Derived>
void BinaryExpression<Derived>::set_right_expression(
    const Expression::Ptr &expression) {
  _right_expression = expression;
}

template <typename Derived>
const Expression::Ptr &BinaryExpression<Derived>::get_right_expression() const {
  return _right_expression;
}

template <typename Derived>
std::ostream &BinaryExpression<Derived>::print(std::ostream &o) const {
  o << "(" << Derived::class_name << " " << *_left_expression << " "
    << *_right_expression << ")";
  return o;
}

} // namespace pddl

} // namespace skdecide
