/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/effect.hh"

namespace skdecide {

namespace pddl {

Effect::~Effect() {}

std::string Effect::print() const {
  std::ostringstream o;
  print(o);
  return o.str();
}

void Effect::collect_add_atoms(const Task &, const Binding &,
                               const AtomCallback &) const {}

void Effect::collect_cost_increase(const Task &, const Binding &,
                                   const CostCallback &) const {}

bool Effect::is_probabilistic() const { return false; }

Effect::Ptr Effect::determinize(const Effect::Ptr &self, DeterminizationMode,
                                std::mt19937 &) const {
  return self;
}

std::vector<Effect::Ptr> Effect::all_determinizations(const Effect::Ptr &self,
                                                      std::mt19937 &) const {
  return {self};
}

std::ostream &operator<<(std::ostream &o, const Effect &e) {
  return e.print(o);
}

} // namespace pddl

} // namespace skdecide
