/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_CONDITIONAL_EFFECT_HH
#define SKDECIDE_PDDL_CONDITIONAL_EFFECT_HH

#include "binary_effect.hh"

namespace skdecide {

namespace pddl {

class ConditionalEffect : public Effect, public BinaryEffect {
public:
  typedef std::shared_ptr<ConditionalEffect> Ptr;

  ConditionalEffect();
  ConditionalEffect(const Formula::Ptr &condition, const Effect::Ptr &effect);
  ConditionalEffect(const ConditionalEffect &other);
  ConditionalEffect &operator=(const ConditionalEffect &other);
  virtual ~ConditionalEffect();

  virtual std::ostream &print(std::ostream &o) const;

  virtual Outcomes apply(const State &state, const Task &task,
                         const Binding &binding) const override;

  void collect_add_atoms(const Task &task, const Binding &binding,
                         const AtomCallback &callback) const override;

  void collect_cost_increase(const Task &task, const Binding &binding,
                             const CostCallback &callback) const override;

  Effect::Ptr determinize(const Effect::Ptr &self, DeterminizationMode mode,
                          std::mt19937 &rng) const override;

  std::vector<Effect::Ptr>
  all_determinizations(const Effect::Ptr &self,
                       std::mt19937 &rng) const override;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_CONDITIONAL_EFFECT_HH
