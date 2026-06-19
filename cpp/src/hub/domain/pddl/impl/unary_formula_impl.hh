/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/unary_formula.hh"

namespace skdecide {

namespace pddl {

template <typename Derived>
UnaryFormula<Derived> &
UnaryFormula<Derived>::operator=(const UnaryFormula<Derived> &other) {
  this->_formula = other._formula;
  return *this;
}

template <typename Derived>
void UnaryFormula<Derived>::set_formula(const Formula::Ptr &formula) {
  _formula = formula;
}

template <typename Derived>
const Formula::Ptr &UnaryFormula<Derived>::get_formula() const {
  return _formula;
}

template <typename Derived>
std::ostream &UnaryFormula<Derived>::print(std::ostream &o) const {
  o << "(" << Derived::class_name << " " << *_formula << ")";
  return o;
}

} // namespace pddl

} // namespace skdecide
