/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/assignment_effect.hh"

namespace skdecide {

namespace pddl {

// === AssignEffect implementation ===

AssignEffect::AssignEffect() {}

AssignEffect::AssignEffect(const FunctionExpression::Ptr &function,
                           const Expression::Ptr &expression)
    : AssignmentEffect<AssignEffect>(function, expression) {}

AssignEffect::AssignEffect(const AssignEffect &other)
    : AssignmentEffect<AssignEffect>(other) {}

AssignEffect &AssignEffect::operator=(const AssignEffect &other) {
  dynamic_cast<AssignmentEffect<AssignEffect> &>(*this) = other;
  return *this;
}

AssignEffect::~AssignEffect() {}

// === ScaleUpEffect implementation ===

ScaleUpEffect::ScaleUpEffect() {}

ScaleUpEffect::ScaleUpEffect(const FunctionExpression::Ptr &function,
                             const Expression::Ptr &expression)
    : AssignmentEffect<ScaleUpEffect>(function, expression) {}

ScaleUpEffect::ScaleUpEffect(const ScaleUpEffect &other)
    : AssignmentEffect<ScaleUpEffect>(other) {}

ScaleUpEffect &ScaleUpEffect::operator=(const ScaleUpEffect &other) {
  dynamic_cast<AssignmentEffect<ScaleUpEffect> &>(*this) = other;
  return *this;
}

ScaleUpEffect::~ScaleUpEffect() {}

// === ScaleDownEffect implementation ===

ScaleDownEffect::ScaleDownEffect() {}

ScaleDownEffect::ScaleDownEffect(const FunctionExpression::Ptr &function,
                                 const Expression::Ptr &expression)
    : AssignmentEffect<ScaleDownEffect>(function, expression) {}

ScaleDownEffect::ScaleDownEffect(const ScaleDownEffect &other)
    : AssignmentEffect<ScaleDownEffect>(other) {}

ScaleDownEffect &ScaleDownEffect::operator=(const ScaleDownEffect &other) {
  dynamic_cast<AssignmentEffect<ScaleDownEffect> &>(*this) = other;
  return *this;
}

ScaleDownEffect::~ScaleDownEffect() {}

// === IncreaseEffect implementation ===

IncreaseEffect::IncreaseEffect() {}

IncreaseEffect::IncreaseEffect(const FunctionExpression::Ptr &function,
                               const Expression::Ptr &expression)
    : AssignmentEffect<IncreaseEffect>(function, expression) {}

IncreaseEffect::IncreaseEffect(const IncreaseEffect &other)
    : AssignmentEffect<IncreaseEffect>(other) {}

IncreaseEffect &IncreaseEffect::operator=(const IncreaseEffect &other) {
  dynamic_cast<AssignmentEffect<IncreaseEffect> &>(*this) = other;
  return *this;
}

IncreaseEffect::~IncreaseEffect() {}

// === DecreaseEffect implementation ===

DecreaseEffect::DecreaseEffect() {}

DecreaseEffect::DecreaseEffect(const FunctionExpression::Ptr &function,
                               const Expression::Ptr &expression)
    : AssignmentEffect<DecreaseEffect>(function, expression) {}

DecreaseEffect::DecreaseEffect(const DecreaseEffect &other)
    : AssignmentEffect<DecreaseEffect>(other) {}

DecreaseEffect &DecreaseEffect::operator=(const DecreaseEffect &other) {
  dynamic_cast<AssignmentEffect<DecreaseEffect> &>(*this) = other;
  return *this;
}

DecreaseEffect::~DecreaseEffect() {}

} // namespace pddl

} // namespace skdecide
