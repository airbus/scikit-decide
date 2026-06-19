/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_BINARY_EXPRESSION_HH
#define SKDECIDE_PDDL_BINARY_EXPRESSION_HH

#include "expression.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> class BinaryExpression : public Expression {
public:
  typedef std::shared_ptr<BinaryExpression<Derived>> Ptr;

  BinaryExpression();
  BinaryExpression(const Expression::Ptr &left_expression,
                   const Expression::Ptr &right_expression);
  BinaryExpression(const BinaryExpression<Derived> &other);
  BinaryExpression<Derived> &operator=(const BinaryExpression<Derived> &other);
  virtual ~BinaryExpression();

  void set_left_expression(const Expression::Ptr &expression);
  const Expression::Ptr &get_left_expression() const;

  void set_right_expression(const Expression::Ptr &expression);
  const Expression::Ptr &get_right_expression() const;

  virtual std::ostream &print(std::ostream &o) const;

protected:
  Expression::Ptr _left_expression;
  Expression::Ptr _right_expression;
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/binary_expression_impl.hh"
#endif

#endif // SKDECIDE_PDDL_BINARY_EXPRESSION_HH
