/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/predicate_effect.hh"

namespace skdecide {

namespace pddl {

PredicateEffect &PredicateEffect::operator=(const PredicateEffect &other) {
  dynamic_cast<TermContainer<PredicateEffect> &>(*this) = other;
  this->_predicate = other._predicate;
  return *this;
}

void PredicateEffect::set_predicate(const Predicate::Ptr &predicate) {
  _predicate = predicate;
}

const std::string &PredicateEffect::get_name() const {
  return _predicate->get_name();
}

std::ostream &PredicateEffect::print(std::ostream &o) const {
  return TermContainer<PredicateEffect>::print(o);
}

} // namespace pddl

} // namespace skdecide
