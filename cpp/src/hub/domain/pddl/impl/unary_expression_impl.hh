/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/unary_expression.hh"

namespace skdecide {

namespace pddl {

template <typename Derived>
UnaryExpression<Derived> &
UnaryExpression<Derived>::operator=(const UnaryExpression<Derived> &other) {
  this->_expression = other._expression;
  return *this;
}

template <typename Derived>
void UnaryExpression<Derived>::set_expression(
    const Expression::Ptr &expression) {
  _expression = expression;
}

template <typename Derived>
const Expression::Ptr &UnaryExpression<Derived>::get_expression() const {
  return _expression;
}

template <typename Derived>
std::ostream &UnaryExpression<Derived>::print(std::ostream &o) const {
  o << "(" << Derived::class_name << " " << *_expression << ")";
  return o;
}

} // namespace pddl

} // namespace skdecide
