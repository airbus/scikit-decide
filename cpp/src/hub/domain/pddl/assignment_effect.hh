/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_ASSIGNMENT_EFFECT_HH
#define SKDECIDE_PDDL_ASSIGNMENT_EFFECT_HH

#include "effect.hh"
#include "function_expression.hh"
#include "expression.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> class AssignmentEffect : public Effect {
public:
  typedef std::shared_ptr<AssignmentEffect<Derived>> Ptr;

  AssignmentEffect();
  AssignmentEffect(const FunctionExpression::Ptr &function,
                   const Expression::Ptr &expression);
  AssignmentEffect(const AssignmentEffect<Derived> &other);
  AssignmentEffect<Derived> &operator=(const AssignmentEffect<Derived> &other);
  virtual ~AssignmentEffect();

  void set_function(const FunctionExpression::Ptr &function);
  const FunctionExpression::Ptr &get_function() const;

  void set_expression(const Expression::Ptr &expression);
  const Expression::Ptr &get_expression() const;

  virtual std::ostream &print(std::ostream &o) const;
  std::string print() const;

private:
  FunctionExpression::Ptr _function;
  Expression::Ptr _expression;
};

class AssignEffect : public AssignmentEffect<AssignEffect> {
public:
  static constexpr char class_name[] = "assign";

  typedef std::shared_ptr<AssignEffect> Ptr;

  AssignEffect();
  AssignEffect(const FunctionExpression::Ptr &function,
               const Expression::Ptr &expression);
  AssignEffect(const AssignEffect &other);
  AssignEffect &operator=(const AssignEffect &other);
  virtual ~AssignEffect();

  virtual Outcomes apply(const State &state, const Task &task,
                         const Binding &binding) const override;
};

class ScaleUpEffect : public AssignmentEffect<ScaleUpEffect> {
public:
  static constexpr char class_name[] = "scale-up";

  typedef std::shared_ptr<ScaleUpEffect> Ptr;

  ScaleUpEffect();
  ScaleUpEffect(const FunctionExpression::Ptr &function,
                const Expression::Ptr &expression);
  ScaleUpEffect(const ScaleUpEffect &other);
  ScaleUpEffect &operator=(const ScaleUpEffect &other);
  virtual ~ScaleUpEffect();

  virtual Outcomes apply(const State &state, const Task &task,
                         const Binding &binding) const override;
};

class ScaleDownEffect : public AssignmentEffect<ScaleDownEffect> {
public:
  static constexpr char class_name[] = "scale-down";

  typedef std::shared_ptr<ScaleDownEffect> Ptr;

  ScaleDownEffect();
  ScaleDownEffect(const FunctionExpression::Ptr &function,
                  const Expression::Ptr &expression);
  ScaleDownEffect(const ScaleDownEffect &other);
  ScaleDownEffect &operator=(const ScaleDownEffect &other);
  virtual ~ScaleDownEffect();

  virtual Outcomes apply(const State &state, const Task &task,
                         const Binding &binding) const override;
};

class IncreaseEffect : public AssignmentEffect<IncreaseEffect> {
public:
  static constexpr char class_name[] = "increase";

  typedef std::shared_ptr<IncreaseEffect> Ptr;

  IncreaseEffect();
  IncreaseEffect(const FunctionExpression::Ptr &function,
                 const Expression::Ptr &expression);
  IncreaseEffect(const IncreaseEffect &other);
  IncreaseEffect &operator=(const IncreaseEffect &other);
  virtual ~IncreaseEffect();

  virtual Outcomes apply(const State &state, const Task &task,
                         const Binding &binding) const override;

  void collect_cost_increase(const Task &task, const Binding &binding,
                             const CostCallback &callback) const override;
};

class DecreaseEffect : public AssignmentEffect<DecreaseEffect> {
public:
  static constexpr char class_name[] = "decrease";

  typedef std::shared_ptr<DecreaseEffect> Ptr;

  DecreaseEffect();
  DecreaseEffect(const FunctionExpression::Ptr &function,
                 const Expression::Ptr &expression);
  DecreaseEffect(const DecreaseEffect &other);
  DecreaseEffect &operator=(const DecreaseEffect &other);
  virtual ~DecreaseEffect();

  virtual Outcomes apply(const State &state, const Task &task,
                         const Binding &binding) const override;
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/assignment_effect_impl.hh"
#endif

#endif // SKDECIDE_PDDL_ASSIGNMENT_EFFECT_HH
