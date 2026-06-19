/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_PREDICATE_EFFECT_HH
#define SKDECIDE_PDDL_PREDICATE_EFFECT_HH

#include "effect.hh"
#include "predicate.hh"
#include "term_container.hh"

namespace skdecide {

namespace pddl {

class PredicateEffect : public Effect, public TermContainer<PredicateEffect> {
public:
  static constexpr char class_name[] = "predicate effect";

  typedef std::shared_ptr<PredicateEffect> Ptr;
  typedef TermContainer<PredicateEffect>::TermPtr TermPtr;
  typedef TermContainer<PredicateEffect>::TermVector TermVector;

  PredicateEffect() {}

  PredicateEffect(const Predicate::Ptr &predicate,
                  const TermContainer<PredicateEffect> &terms)
      : TermContainer<PredicateEffect>(terms), _predicate(predicate) {}

  PredicateEffect(const PredicateEffect &other)
      : TermContainer<PredicateEffect>(other), _predicate(other._predicate) {}

  PredicateEffect &operator=(const PredicateEffect &other);

  virtual ~PredicateEffect() {}

  void set_predicate(const Predicate::Ptr &predicate);

  const Predicate::Ptr &get_predicate() const { return _predicate; }

  const std::string &get_name() const;

  std::ostream &print(std::ostream &o) const override;

  virtual Outcomes apply(const State &state, const Task &task,
                         const Binding &binding) const override;

  void collect_add_atoms(const Task &task, const Binding &binding,
                         const AtomCallback &callback) const override;

private:
  Predicate::Ptr _predicate;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PREDICATE_EFFECT_HH
