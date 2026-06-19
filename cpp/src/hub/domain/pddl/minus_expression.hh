/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_MINUS_EXPRESSION_HH
#define SKDECIDE_PDDL_MINUS_EXPRESSION_HH

#include "unary_expression.hh"

namespace skdecide {

namespace pddl {

class MinusExpression : public UnaryExpression<MinusExpression> {
public:
  static constexpr char class_name[] = "-";

  typedef std::shared_ptr<MinusExpression> Ptr;

  MinusExpression();
  MinusExpression(const Expression::Ptr &expression);
  MinusExpression(const MinusExpression &other);
  MinusExpression &operator=(const MinusExpression &other);
  virtual ~MinusExpression();

  virtual double evaluate(const State &state, const Task &task,
                          const Binding &binding) const override;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_MINUS_EXPRESSION_HH
