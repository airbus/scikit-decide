/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/probabilistic_effect.hh"
#include <stdexcept>

namespace skdecide {

namespace pddl {

ProbabilisticEffect::ProbabilisticEffect() {}

ProbabilisticEffect::ProbabilisticEffect(const ProbabilisticEffect &other)
    : _outcomes(other._outcomes) {}

ProbabilisticEffect &
ProbabilisticEffect::operator=(const ProbabilisticEffect &other) {
  _outcomes = other._outcomes;
  return *this;
}

ProbabilisticEffect::~ProbabilisticEffect() {}

ProbabilisticEffect &
ProbabilisticEffect::append_outcome(double probability,
                                    const Effect::Ptr &effect) {
  _outcomes.emplace_back(probability, effect);
  return *this;
}

ProbabilisticEffect &
ProbabilisticEffect::remove_outcome(const std::size_t &index) {
  if (index >= _outcomes.size()) {
    throw std::out_of_range(
        "Outcome index out of range in ProbabilisticEffect::remove_outcome");
  }
  _outcomes.erase(_outcomes.begin() + index);
  return *this;
}

const ProbabilisticEffect::Outcome &
ProbabilisticEffect::outcome_at(const std::size_t &index) const {
  if (index >= _outcomes.size()) {
    throw std::out_of_range(
        "Outcome index out of range in ProbabilisticEffect::outcome_at");
  }
  return _outcomes[index];
}

const ProbabilisticEffect::OutcomeVector &
ProbabilisticEffect::get_outcomes() const {
  return _outcomes;
}

std::ostream &ProbabilisticEffect::print(std::ostream &o) const {
  o << "(probabilistic";
  for (const auto &outcome : _outcomes) {
    o << " " << outcome.first << " " << *outcome.second;
  }
  o << ")";
  return o;
}

} // namespace pddl

} // namespace skdecide
