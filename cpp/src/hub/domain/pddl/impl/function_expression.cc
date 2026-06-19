/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/function_expression.hh"

namespace skdecide {

namespace pddl {

FunctionExpression::FunctionExpression() {}

FunctionExpression::FunctionExpression(
    const Function::Ptr &function,
    const TermContainer<FunctionExpression> &terms)
    : TermContainer<FunctionExpression>(terms), _function(function) {}

FunctionExpression::FunctionExpression(const FunctionExpression &other)
    : TermContainer<FunctionExpression>(other), _function(other._function) {}

FunctionExpression &
FunctionExpression::operator=(const FunctionExpression &other) {
  dynamic_cast<TermContainer<FunctionExpression> &>(*this) = other;
  this->_function = other._function;
  return *this;
}

FunctionExpression::~FunctionExpression() {}

void FunctionExpression::set_function(const Function::Ptr &function) {
  _function = function;
}

const Function::Ptr &FunctionExpression::get_function() const {
  return _function;
}

const std::string &FunctionExpression::get_name() const {
  return _function->get_name();
}

std::ostream &FunctionExpression::print(std::ostream &o) const {
  return TermContainer<FunctionExpression>::print(o);
}

} // namespace pddl

} // namespace skdecide
