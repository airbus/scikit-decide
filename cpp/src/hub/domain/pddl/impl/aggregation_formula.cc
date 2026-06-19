/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/aggregation_formula.hh"

namespace skdecide {

namespace pddl {

// === ConjunctionFormula implementation ===

ConjunctionFormula::ConjunctionFormula() {}

ConjunctionFormula::ConjunctionFormula(const ConjunctionFormula &other)
    : AggregationFormula(other) {}

ConjunctionFormula &
ConjunctionFormula::operator=(const ConjunctionFormula &other) {
  dynamic_cast<AggregationFormula<ConjunctionFormula> &>(*this) = other;
  return *this;
}

ConjunctionFormula::~ConjunctionFormula() {}

// === DisjunctionFormula implementation ===

DisjunctionFormula::DisjunctionFormula() {}

DisjunctionFormula::DisjunctionFormula(const DisjunctionFormula &other)
    : AggregationFormula(other) {}

DisjunctionFormula &
DisjunctionFormula::operator=(const DisjunctionFormula &other) {
  dynamic_cast<AggregationFormula<DisjunctionFormula> &>(*this) = other;
  return *this;
}

DisjunctionFormula::~DisjunctionFormula() {}

} // namespace pddl

} // namespace skdecide
