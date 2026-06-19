/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_UNARY_EXPRESSION_HH
#define SKDECIDE_PDDL_UNARY_EXPRESSION_HH

#include "expression.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> class UnaryExpression : public Expression {
public:
  typedef std::shared_ptr<UnaryExpression<Derived>> Ptr;

  UnaryExpression() {}

  UnaryExpression(const Expression::Ptr &expression)
      : _expression(expression) {}

  UnaryExpression(const UnaryExpression<Derived> &other)
      : _expression(other._expression) {}

  UnaryExpression<Derived> &operator=(const UnaryExpression<Derived> &other);

  virtual ~UnaryExpression() {}

  void set_expression(const Expression::Ptr &expression);

  const Expression::Ptr &get_expression() const;

  virtual std::ostream &print(std::ostream &o) const;

protected:
  Expression::Ptr _expression;
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/unary_expression_impl.hh"
#endif

#endif // SKDECIDE_PDDL_UNARY_EXPRESSION_HH
