/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/binary_formula.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> BinaryFormula<Derived>::BinaryFormula() {}

template <typename Derived>
BinaryFormula<Derived>::BinaryFormula(const Formula::Ptr &left_formula,
                                      const Formula::Ptr &right_formula)
    : _left_formula(left_formula), _right_formula(right_formula) {}

template <typename Derived>
BinaryFormula<Derived>::BinaryFormula(const BinaryFormula<Derived> &other)
    : _left_formula(other._left_formula), _right_formula(other._right_formula) {
}

template <typename Derived>
BinaryFormula<Derived> &
BinaryFormula<Derived>::operator=(const BinaryFormula<Derived> &other) {
  this->_left_formula = other._left_formula;
  this->_right_formula = other._right_formula;
  return *this;
}

template <typename Derived> BinaryFormula<Derived>::~BinaryFormula() {}

template <typename Derived>
void BinaryFormula<Derived>::set_left_formula(const Formula::Ptr &formula) {
  _left_formula = formula;
}

template <typename Derived>
const Formula::Ptr &BinaryFormula<Derived>::get_left_formula() const {
  return _left_formula;
}

template <typename Derived>
void BinaryFormula<Derived>::set_right_formula(const Formula::Ptr &formula) {
  _right_formula = formula;
}

template <typename Derived>
const Formula::Ptr &BinaryFormula<Derived>::get_right_formula() const {
  return _right_formula;
}

template <typename Derived>
std::ostream &BinaryFormula<Derived>::print(std::ostream &o) const {
  o << "(" << Derived::class_name << " " << *_left_formula << " "
    << *_right_formula << ")";
  return o;
}

} // namespace pddl

} // namespace skdecide
