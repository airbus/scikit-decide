/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/predicate_formula.hh"

namespace skdecide {

namespace pddl {

PredicateFormula &PredicateFormula::operator=(const PredicateFormula &other) {
  dynamic_cast<TermContainer<PredicateFormula> &>(*this) = other;
  this->_predicate = other._predicate;
  return *this;
}

void PredicateFormula::set_predicate(const Predicate::Ptr &predicate) {
  _predicate = predicate;
}

const std::string &PredicateFormula::get_name() const {
  return _predicate->get_name();
}

std::ostream &PredicateFormula::print(std::ostream &o) const {
  return TermContainer<PredicateFormula>::print(o);
}

} // namespace pddl

} // namespace skdecide
