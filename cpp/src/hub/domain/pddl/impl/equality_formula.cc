/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/equality_formula.hh"
#include "hub/domain/pddl/term.hh"

namespace skdecide {

namespace pddl {

EqualityFormula::EqualityFormula() {}

EqualityFormula::EqualityFormula(const TermContainer<EqualityFormula> &terms)
    : TermContainer<EqualityFormula>(terms) {}

EqualityFormula::EqualityFormula(const EqualityFormula &other)
    : TermContainer<EqualityFormula>(other) {}

EqualityFormula &EqualityFormula::operator=(const EqualityFormula &other) {
  dynamic_cast<TermContainer<EqualityFormula> &>(*this) = other;
  return *this;
}

EqualityFormula::~EqualityFormula() {}

std::string EqualityFormula::get_name() const { return "="; }

std::ostream &EqualityFormula::print(std::ostream &o) const {
  return TermContainer<EqualityFormula>::print(o);
}

} // namespace pddl

} // namespace skdecide
