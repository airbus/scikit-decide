/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_QUANTIFIED_EFFECT_HH
#define SKDECIDE_PDDL_QUANTIFIED_EFFECT_HH

#include "effect.hh"
#include "variable_container.hh"

namespace skdecide {

namespace pddl {

template <typename Derived>
class QuantifiedEffect : public Effect, public VariableContainer<Derived> {
public:
  typedef std::shared_ptr<QuantifiedEffect<Derived>> Ptr;
  typedef typename VariableContainer<Derived>::VariablePtr VariablePtr;
  typedef typename VariableContainer<Derived>::VariableVector VariableVector;

  QuantifiedEffect();
  QuantifiedEffect(const Effect::Ptr &effect,
                   const VariableContainer<Derived> &variables);
  QuantifiedEffect(const QuantifiedEffect &other);
  QuantifiedEffect &operator=(const QuantifiedEffect &other);
  virtual ~QuantifiedEffect();

  QuantifiedEffect &set_effect(const Effect::Ptr &effect);
  const Effect::Ptr &get_effect() const;

  static const char *get_name();

  virtual std::ostream &print(std::ostream &o) const;

private:
  Effect::Ptr _effect;
};

class UniversalEffect : public QuantifiedEffect<UniversalEffect> {
public:
  static constexpr char class_name[] = "forall";

  typedef std::shared_ptr<UniversalEffect> Ptr;
  typedef QuantifiedEffect<UniversalEffect> VariablePtr;
  typedef QuantifiedEffect<UniversalEffect> VariableVector;

  UniversalEffect();
  UniversalEffect(const Effect::Ptr &effect,
                  const VariableContainer<UniversalEffect> &variables);
  UniversalEffect(const UniversalEffect &other);
  UniversalEffect &operator=(const UniversalEffect &other);
  virtual ~UniversalEffect();

  virtual Outcomes apply(const State &state, const Task &task,
                         const Binding &binding) const override;

  Effect::Ptr determinize(const Effect::Ptr &self, DeterminizationMode mode,
                          std::mt19937 &rng) const override;

  std::vector<Effect::Ptr>
  all_determinizations(const Effect::Ptr &self,
                       std::mt19937 &rng) const override;
};

class ExistentialEffect : public QuantifiedEffect<ExistentialEffect> {
public:
  static constexpr char class_name[] = "exists";

  typedef std::shared_ptr<ExistentialEffect> Ptr;
  typedef QuantifiedEffect<ExistentialEffect> VariablePtr;
  typedef QuantifiedEffect<ExistentialEffect> VariableVector;

  ExistentialEffect();
  ExistentialEffect(const Effect::Ptr &effect,
                    const VariableContainer<ExistentialEffect> &variables);
  ExistentialEffect(const ExistentialEffect &other);
  ExistentialEffect &operator=(const ExistentialEffect &other);
  virtual ~ExistentialEffect();

  virtual Outcomes apply(const State &state, const Task &task,
                         const Binding &binding) const override;

  Effect::Ptr determinize(const Effect::Ptr &self, DeterminizationMode mode,
                          std::mt19937 &rng) const override;

  std::vector<Effect::Ptr>
  all_determinizations(const Effect::Ptr &self,
                       std::mt19937 &rng) const override;
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/quantified_effect_impl.hh"
#endif

#endif // SKDECIDE_PDDL_QUANTIFIED_EFFECT_HH
