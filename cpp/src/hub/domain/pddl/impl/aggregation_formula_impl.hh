/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/aggregation_formula.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> AggregationFormula<Derived>::AggregationFormula() {}

template <typename Derived>
AggregationFormula<Derived>::AggregationFormula(
    const AggregationFormula<Derived> &other)
    : _formulas(other._formulas) {}

template <typename Derived>
AggregationFormula<Derived> &AggregationFormula<Derived>::operator=(
    const AggregationFormula<Derived> &other) {
  this->_formulas = other._formulas;
  return *this;
}

template <typename Derived>
AggregationFormula<Derived>::~AggregationFormula() {}

template <typename Derived>
AggregationFormula<Derived> &
AggregationFormula<Derived>::append_formula(const Formula::Ptr &formula) {
  _formulas.push_back(formula);
  return *this;
}

template <typename Derived>
AggregationFormula<Derived> &
AggregationFormula<Derived>::remove_formula(const std::size_t &index) {
  if (index >= _formulas.size()) {
    throw std::out_of_range(
        "SKDECIDE exception: index " + std::to_string(index) +
        " exceeds the size of the '" + Derived::class_name + "' formula");
  } else {
    _formulas.erase(_formulas.begin() + index);
    return *this;
  }
}

template <typename Derived>
const Formula::Ptr &
AggregationFormula<Derived>::formula_at(const std::size_t &index) {
  if (index >= _formulas.size()) {
    throw std::out_of_range(
        "SKDECIDE exception: index " + std::to_string(index) +
        " exceeds the size of the '" + Derived::class_name + "' formula");
  } else {
    return _formulas[index];
  }
}

template <typename Derived>
const typename AggregationFormula<Derived>::FormulaVector &
AggregationFormula<Derived>::get_formulas() const {
  return _formulas;
}

template <typename Derived>
std::ostream &AggregationFormula<Derived>::print(std::ostream &o) const {
  o << "(" << Derived::class_name;
  for (const auto &f : _formulas) {
    o << " " << *f;
  }
  o << ")";
  return o;
}

} // namespace pddl

} // namespace skdecide
