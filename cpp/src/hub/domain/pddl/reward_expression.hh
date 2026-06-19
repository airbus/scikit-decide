/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_REWARD_EXPRESSION_HH
#define SKDECIDE_PDDL_REWARD_EXPRESSION_HH

#include "expression.hh"

namespace skdecide {

namespace pddl {

class RewardExpression : public Expression {
public:
  static constexpr char class_name[] = "reward";

  typedef std::shared_ptr<RewardExpression> Ptr;

  RewardExpression() {}

  virtual ~RewardExpression() {}

  std::ostream &print(std::ostream &o) const override;

  virtual double evaluate(const State &state, const Task &task,
                          const Binding &binding) const override;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_REWARD_EXPRESSION_HH
