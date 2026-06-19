/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/numerical_expression.hh"

namespace skdecide {

namespace pddl {

NumericalExpression::NumericalExpression() {}

NumericalExpression::NumericalExpression(const Number::Ptr &number)
    : _number(number) {}

NumericalExpression::NumericalExpression(const NumericalExpression &other)
    : _number(other._number) {}

NumericalExpression &
NumericalExpression::operator=(const NumericalExpression &other) {
  this->_number = other._number;
  return *this;
}

NumericalExpression::~NumericalExpression() {}

void NumericalExpression::set_number(const Number::Ptr &number) {
  _number = number;
}

const Number::Ptr &NumericalExpression::get_number() const { return _number; }

std::ostream &NumericalExpression::print(std::ostream &o) const {
  if (_number) {
    return _number->print(o);
  } else {
    o << 0;
    return o;
  }
}

} // namespace pddl

} // namespace skdecide
