/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_BINARY_EFFECT_HH
#define SKDECIDE_PDDL_BINARY_EFFECT_HH

#include "formula.hh"
#include "effect.hh"

namespace skdecide {

namespace pddl {

class BinaryEffect { // does not inherit from Effect since Action is not an
                     // effect but needs BinaryEffect's methods
public:
  typedef std::shared_ptr<BinaryEffect> Ptr;

  BinaryEffect();
  BinaryEffect(const Formula::Ptr &condition, const Effect::Ptr &effect);
  BinaryEffect(const BinaryEffect &other);
  BinaryEffect &operator=(const BinaryEffect &other);
  virtual ~BinaryEffect();

  void set_condition(const Formula::Ptr &condition);
  const Formula::Ptr &get_condition() const;

  void set_effect(const Effect::Ptr &effect);
  const Effect::Ptr &get_effect() const;

protected:
  Formula::Ptr _condition;
  Effect::Ptr _effect;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_BINARY_EFFECT_HH
