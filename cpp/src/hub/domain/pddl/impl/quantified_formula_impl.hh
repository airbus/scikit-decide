/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/quantified_formula.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> QuantifiedFormula<Derived>::QuantifiedFormula() {}

template <typename Derived>
QuantifiedFormula<Derived>::QuantifiedFormula(
    const Formula::Ptr &formula, const VariableContainer<Derived> &variables)
    : VariableContainer<Derived>(variables), _formula(formula) {}

template <typename Derived>
QuantifiedFormula<Derived>::QuantifiedFormula(
    const QuantifiedFormula<Derived> &other)
    : VariableContainer<Derived>(other), _formula(other._formula) {}

template <typename Derived>
QuantifiedFormula<Derived> &
QuantifiedFormula<Derived>::operator=(const QuantifiedFormula<Derived> &other) {
  dynamic_cast<VariableContainer<Derived> &>(*this) = other;
  this->_formula = other._formula;
  return *this;
}

template <typename Derived> QuantifiedFormula<Derived>::~QuantifiedFormula() {}

template <typename Derived>
QuantifiedFormula<Derived> &
QuantifiedFormula<Derived>::set_formula(const Formula::Ptr &formula) {
  _formula = formula;
  return *this;
}

template <typename Derived>
const Formula::Ptr &QuantifiedFormula<Derived>::get_formula() const {
  return _formula;
}

template <typename Derived> const char *QuantifiedFormula<Derived>::get_name() {
  return Derived::class_name;
}

template <typename Derived>
std::ostream &QuantifiedFormula<Derived>::print(std::ostream &o) const {
  o << "(" << Derived::class_name << " (";
  for (const auto &v : this->get_variables()) {
    o << " " << *v;
  }
  o << ") " << *_formula << ")";
  return o;
}

} // namespace pddl

} // namespace skdecide
