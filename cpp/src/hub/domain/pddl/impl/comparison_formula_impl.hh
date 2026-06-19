/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/comparison_formula.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> ComparisonFormula<Derived>::ComparisonFormula() {}

template <typename Derived>
ComparisonFormula<Derived>::ComparisonFormula(
    const Expression::Ptr &left_expression,
    const Expression::Ptr &right_expression)
    : BinaryExpression<Derived>(left_expression, right_expression) {}

template <typename Derived>
ComparisonFormula<Derived>::ComparisonFormula(
    const ComparisonFormula<Derived> &other)
    : BinaryExpression<Derived>(other) {}

template <typename Derived>
ComparisonFormula<Derived> &
ComparisonFormula<Derived>::operator=(const ComparisonFormula<Derived> &other) {
  dynamic_cast<BinaryExpression<Derived> &>(*this) = other;
  return *this;
}

template <typename Derived>
std::ostream &ComparisonFormula<Derived>::print(std::ostream &o) const {
  return BinaryExpression<Derived>::print(o);
}

template <typename Derived>
std::string ComparisonFormula<Derived>::print() const {
  return Formula::print();
}

} // namespace pddl

} // namespace skdecide
