/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/assignment_effect.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> AssignmentEffect<Derived>::AssignmentEffect() {}

template <typename Derived>
AssignmentEffect<Derived>::AssignmentEffect(
    const FunctionExpression::Ptr &function, const Expression::Ptr &expression)
    : _function(function), _expression(expression) {}

template <typename Derived>
AssignmentEffect<Derived>::AssignmentEffect(
    const AssignmentEffect<Derived> &other)
    : _function(other._function), _expression(other._expression) {}

template <typename Derived>
AssignmentEffect<Derived> &
AssignmentEffect<Derived>::operator=(const AssignmentEffect<Derived> &other) {
  this->_function = other._function;
  this->_expression = other._expression;
  return *this;
}

template <typename Derived> AssignmentEffect<Derived>::~AssignmentEffect() {}

template <typename Derived>
void AssignmentEffect<Derived>::set_function(
    const FunctionExpression::Ptr &function) {
  _function = function;
}

template <typename Derived>
const FunctionExpression::Ptr &AssignmentEffect<Derived>::get_function() const {
  return _function;
}

template <typename Derived>
void AssignmentEffect<Derived>::set_expression(
    const Expression::Ptr &expression) {
  _expression = expression;
}

template <typename Derived>
const Expression::Ptr &AssignmentEffect<Derived>::get_expression() const {
  return _expression;
}

template <typename Derived>
std::ostream &AssignmentEffect<Derived>::print(std::ostream &o) const {
  o << "(" << Derived::class_name << " " << *_function << " " << *_expression
    << ")";
  return o;
}

template <typename Derived>
std::string AssignmentEffect<Derived>::print() const {
  return Effect::print();
}

} // namespace pddl

} // namespace skdecide
