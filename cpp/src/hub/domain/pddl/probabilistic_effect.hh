/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_PROBABILISTIC_EFFECT_HH
#define SKDECIDE_PDDL_PROBABILISTIC_EFFECT_HH

#include "effect.hh"
#include <utility>
#include <vector>

namespace skdecide {

namespace pddl {

class ProbabilisticEffect : public Effect {
public:
  static constexpr char class_name[] = "probabilistic";

  typedef std::shared_ptr<ProbabilisticEffect> Ptr;
  typedef std::pair<double, Effect::Ptr> Outcome;
  typedef std::vector<Outcome> OutcomeVector;

  ProbabilisticEffect();
  ProbabilisticEffect(const ProbabilisticEffect &other);
  ProbabilisticEffect &operator=(const ProbabilisticEffect &other);
  virtual ~ProbabilisticEffect();

  ProbabilisticEffect &append_outcome(double probability,
                                      const Effect::Ptr &effect);

  ProbabilisticEffect &remove_outcome(const std::size_t &index);

  const Outcome &outcome_at(const std::size_t &index) const;

  const OutcomeVector &get_outcomes() const;

  virtual std::ostream &print(std::ostream &o) const;

  virtual Outcomes apply(const State &state, const Task &task,
                         const Binding &binding) const override;

  void collect_add_atoms(const Task &task, const Binding &binding,
                         const AtomCallback &callback) const override;

  void collect_cost_increase(const Task &task, const Binding &binding,
                             const CostCallback &callback) const override;

  bool is_probabilistic() const override;

  Effect::Ptr determinize(const Effect::Ptr &self, DeterminizationMode mode,
                          std::mt19937 &rng) const override;

  std::vector<Effect::Ptr>
  all_determinizations(const Effect::Ptr &self,
                       std::mt19937 &rng) const override;

private:
  OutcomeVector _outcomes;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PROBABILISTIC_EFFECT_HH
