/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/derived_predicate.hh"

namespace skdecide {

namespace pddl {

DerivedPredicate::DerivedPredicate(const std::string &name) : Predicate(name) {}

DerivedPredicate::DerivedPredicate(const std::string &name,
                                   const Formula::Ptr &formula)
    : Predicate(name), _formula(formula) {}

DerivedPredicate::DerivedPredicate(const Predicate::Ptr &predicate,
                                   const Formula::Ptr &formula)
    : Predicate(*predicate), _formula(formula) {}

DerivedPredicate::DerivedPredicate(const DerivedPredicate &other)
    : Predicate(other), _formula(other._formula) {}

DerivedPredicate &DerivedPredicate::operator=(const DerivedPredicate &other) {
  dynamic_cast<Predicate &>(*this) = other;
  this->_formula = other._formula;
  return *this;
}

DerivedPredicate::~DerivedPredicate() {}

void DerivedPredicate::set_formula(const Formula::Ptr &formula) {
  _formula = formula;
}

const Formula::Ptr &DerivedPredicate::get_formula() const { return _formula; }

std::ostream &DerivedPredicate::print(std::ostream &o) const {
  o << "(:derived " << dynamic_cast<const Predicate &>(*this) << " "
    << *_formula << ")";
  return o;
}

std::ostream &operator<<(std::ostream &o, const DerivedPredicate &d) {
  return d.print(o);
}

} // namespace pddl

} // namespace skdecide
