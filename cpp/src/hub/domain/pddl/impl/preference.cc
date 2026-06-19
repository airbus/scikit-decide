/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/preference.hh"

namespace skdecide {

namespace pddl {

Preference::Preference() : Identifier("anonymous") {}

Preference::Preference(const std::string &name) : Identifier(name) {}

Preference::Preference(const Formula::Ptr &formula, const std::string &name)
    : Identifier(name), _formula(formula) {}

Preference::Preference(const Preference &other)
    : Identifier(other), _formula(other._formula) {}

Preference &Preference::operator=(const Preference &other) {
  dynamic_cast<Identifier &>(*this) = other;
  this->_formula = other._formula;
  return *this;
}

Preference::~Preference() {}

void Preference::set_name(const std::string &name) {
  Identifier::set_name(name);
}

Preference &Preference::set_formula(const Formula::Ptr &formula) {
  _formula = formula;
  return *this;
}

const Formula::Ptr &Preference::get_formula() const { return _formula; }

std::ostream &Preference::print(std::ostream &o) const {
  o << "(preference " << ((_name != "anonymous ") ? _name : "") << *_formula
    << ")";
  return o;
}

} // namespace pddl

} // namespace skdecide
