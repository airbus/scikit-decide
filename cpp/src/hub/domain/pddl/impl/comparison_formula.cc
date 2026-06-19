/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/comparison_formula.hh"

namespace skdecide {

namespace pddl {

// === GreaterFormula implementation ===

GreaterFormula::GreaterFormula() {}

GreaterFormula::GreaterFormula(const Expression::Ptr &left_expression,
                               const Expression::Ptr &right_expression)
    : ComparisonFormula<GreaterFormula>(left_expression, right_expression) {}

GreaterFormula::GreaterFormula(const GreaterFormula &other)
    : ComparisonFormula<GreaterFormula>(other) {}

GreaterFormula &GreaterFormula::operator=(const GreaterFormula &other) {
  dynamic_cast<ComparisonFormula<GreaterFormula> &>(*this) = other;
  return *this;
}

GreaterFormula::~GreaterFormula() {}

// === GreaterEqFormula implementation ===

GreaterEqFormula::GreaterEqFormula() {}

GreaterEqFormula::GreaterEqFormula(const Expression::Ptr &left_expression,
                                   const Expression::Ptr &right_expression)
    : ComparisonFormula<GreaterEqFormula>(left_expression, right_expression) {}

GreaterEqFormula::GreaterEqFormula(const GreaterEqFormula &other)
    : ComparisonFormula<GreaterEqFormula>(other) {}

GreaterEqFormula &GreaterEqFormula::operator=(const GreaterEqFormula &other) {
  dynamic_cast<ComparisonFormula<GreaterEqFormula> &>(*this) = other;
  return *this;
}

GreaterEqFormula::~GreaterEqFormula() {}

// === LessEqFormula implementation ===

LessEqFormula::LessEqFormula() {}

LessEqFormula::LessEqFormula(const Expression::Ptr &left_expression,
                             const Expression::Ptr &right_expression)
    : ComparisonFormula<LessEqFormula>(left_expression, right_expression) {}

LessEqFormula::LessEqFormula(const LessEqFormula &other)
    : ComparisonFormula<LessEqFormula>(other) {}

LessEqFormula &LessEqFormula::operator=(const LessEqFormula &other) {
  dynamic_cast<ComparisonFormula<LessEqFormula> &>(*this) = other;
  return *this;
}

LessEqFormula::~LessEqFormula() {}

// === LessFormula implementation ===

LessFormula::LessFormula() {}

LessFormula::LessFormula(const Expression::Ptr &left_expression,
                         const Expression::Ptr &right_expression)
    : ComparisonFormula<LessFormula>(left_expression, right_expression) {}

LessFormula::LessFormula(const LessFormula &other)
    : ComparisonFormula<LessFormula>(other) {}

LessFormula &LessFormula::operator=(const LessFormula &other) {
  dynamic_cast<ComparisonFormula<LessFormula> &>(*this) = other;
  return *this;
}

LessFormula::~LessFormula() {}

// === EqFormula implementation ===

EqFormula::EqFormula() {}

EqFormula::EqFormula(const Expression::Ptr &left_expression,
                     const Expression::Ptr &right_expression)
    : ComparisonFormula<EqFormula>(left_expression, right_expression) {}

EqFormula::EqFormula(const EqFormula &other)
    : ComparisonFormula<EqFormula>(other) {}

EqFormula &EqFormula::operator=(const EqFormula &other) {
  dynamic_cast<ComparisonFormula<EqFormula> &>(*this) = other;
  return *this;
}

EqFormula::~EqFormula() {}

} // namespace pddl

} // namespace skdecide
