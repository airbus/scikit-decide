/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_OPTIMIZATION_EXPRESSION_HH
#define SKDECIDE_PDDL_OPTIMIZATION_EXPRESSION_HH

#include "unary_expression.hh"

namespace skdecide {

namespace pddl {

class MinimizeExpression : public UnaryExpression<MinimizeExpression> {
public:
  static constexpr char class_name[] = "minimize";

  typedef std::shared_ptr<MinimizeExpression> Ptr;

  MinimizeExpression();
  MinimizeExpression(const Expression::Ptr &expression);
  MinimizeExpression(const MinimizeExpression &other);
  MinimizeExpression &operator=(const MinimizeExpression &other);
  virtual ~MinimizeExpression();

  virtual double evaluate(const State &state, const Task &task,
                          const Binding &binding) const override;
};

class MaximizeExpression : public UnaryExpression<MaximizeExpression> {
public:
  static constexpr char class_name[] = "maximize";

  typedef std::shared_ptr<MaximizeExpression> Ptr;

  MaximizeExpression();
  MaximizeExpression(const Expression::Ptr &expression);
  MaximizeExpression(const MaximizeExpression &other);
  MaximizeExpression &operator=(const MaximizeExpression &other);
  virtual ~MaximizeExpression();

  virtual double evaluate(const State &state, const Task &task,
                          const Binding &binding) const override;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_OPTIMIZATION_EXPRESSION_HH
