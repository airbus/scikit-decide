/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/aggregation_effect.hh"

namespace skdecide {

namespace pddl {

// === ConjunctionEffect implementation ===

ConjunctionEffect::ConjunctionEffect() {}

ConjunctionEffect::ConjunctionEffect(const ConjunctionEffect &other)
    : AggregationEffect<ConjunctionEffect>(other) {}

ConjunctionEffect &
ConjunctionEffect::operator=(const ConjunctionEffect &other) {
  dynamic_cast<AggregationEffect<ConjunctionEffect> &>(*this) = other;
  return *this;
}

ConjunctionEffect::~ConjunctionEffect() {}

// === DisjunctionEffect implementation ===

DisjunctionEffect::DisjunctionEffect() {}

DisjunctionEffect::DisjunctionEffect(const DisjunctionEffect &other)
    : AggregationEffect<DisjunctionEffect>(other) {}

DisjunctionEffect &
DisjunctionEffect::operator=(const DisjunctionEffect &other) {
  dynamic_cast<AggregationEffect<DisjunctionEffect> &>(*this) = other;
  return *this;
}

DisjunctionEffect::~DisjunctionEffect() {}

} // namespace pddl

} // namespace skdecide
