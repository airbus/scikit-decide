/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_NUMERICAL_EXPRESSION_HH
#define SKDECIDE_PDDL_NUMERICAL_EXPRESSION_HH

#include "expression.hh"
#include "number.hh"

namespace skdecide {

namespace pddl {

class NumericalExpression : public Expression {
public:
  typedef std::shared_ptr<NumericalExpression> Ptr;

  NumericalExpression();
  NumericalExpression(const Number::Ptr &number);
  NumericalExpression(const NumericalExpression &other);
  NumericalExpression &operator=(const NumericalExpression &other);
  virtual ~NumericalExpression();

  void set_number(const Number::Ptr &number);
  const Number::Ptr &get_number() const;

  virtual std::ostream &print(std::ostream &o) const;

  virtual double evaluate(const State &state, const Task &task,
                          const Binding &binding) const override;

private:
  Number::Ptr _number;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_NUMERICAL_EXPRESSION_HH
