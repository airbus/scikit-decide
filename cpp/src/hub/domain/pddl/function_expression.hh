/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_FUNCTION_EXPRESSION_HH
#define SKDECIDE_PDDL_FUNCTION_EXPRESSION_HH

#include "expression.hh"
#include "function.hh"
#include "term_container.hh"

namespace skdecide {

namespace pddl {

class FunctionExpression : public Expression,
                           public TermContainer<FunctionExpression> {
public:
  static constexpr char class_name[] = "function expression";

  typedef std::shared_ptr<FunctionExpression> Ptr;
  typedef typename TermContainer<FunctionExpression>::TermPtr TermPtr;
  typedef typename TermContainer<FunctionExpression>::TermVector TermVector;

  FunctionExpression();
  FunctionExpression(const Function::Ptr &function,
                     const TermContainer<FunctionExpression> &terms);
  FunctionExpression(const FunctionExpression &other);
  FunctionExpression &operator=(const FunctionExpression &other);
  virtual ~FunctionExpression();

  void set_function(const Function::Ptr &function);
  const Function::Ptr &get_function() const;

  const std::string &get_name() const;

  virtual std::ostream &print(std::ostream &o) const;

  virtual double evaluate(const State &state, const Task &task,
                          const Binding &binding) const override;

private:
  Function::Ptr _function;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_FUNCTION_EXPRESSION_HH
