/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_OPERATION_EXPRESSION_HH
#define SKDECIDE_PDDL_OPERATION_EXPRESSION_HH

#include "binary_expression.hh"

namespace skdecide {

namespace pddl {

class AddExpression : public BinaryExpression<AddExpression> {
public:
  static constexpr char class_name[] = "+";

  typedef std::shared_ptr<AddExpression> Ptr;

  AddExpression();
  AddExpression(const Expression::Ptr &left_expression,
                const Expression::Ptr &right_expression);
  AddExpression(const AddExpression &other);
  AddExpression &operator=(const AddExpression &other);
  virtual ~AddExpression();

  virtual double evaluate(const State &state, const Task &task,
                          const Binding &binding) const override;
};

class SubExpression : public BinaryExpression<SubExpression> {
public:
  static constexpr char class_name[] = "-";

  typedef std::shared_ptr<SubExpression> Ptr;

  SubExpression();
  SubExpression(const Expression::Ptr &left_expression,
                const Expression::Ptr &right_expression);
  SubExpression(const SubExpression &other);
  SubExpression &operator=(const SubExpression &other);
  virtual ~SubExpression();

  virtual double evaluate(const State &state, const Task &task,
                          const Binding &binding) const override;
};

class MulExpression : public BinaryExpression<MulExpression> {
public:
  static constexpr char class_name[] = "*";

  typedef std::shared_ptr<MulExpression> Ptr;

  MulExpression();
  MulExpression(const Expression::Ptr &left_expression,
                const Expression::Ptr &right_expression);
  MulExpression(const MulExpression &other);
  MulExpression &operator=(const MulExpression &other);
  virtual ~MulExpression();

  virtual double evaluate(const State &state, const Task &task,
                          const Binding &binding) const override;
};

class DivExpression : public BinaryExpression<DivExpression> {
public:
  static constexpr char class_name[] = "/";

  typedef std::shared_ptr<DivExpression> Ptr;

  DivExpression();
  DivExpression(const Expression::Ptr &left_expression,
                const Expression::Ptr &right_expression);
  DivExpression(const DivExpression &other);
  DivExpression &operator=(const DivExpression &other);
  virtual ~DivExpression();

  virtual double evaluate(const State &state, const Task &task,
                          const Binding &binding) const override;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_OPERATION_EXPRESSION_HH
