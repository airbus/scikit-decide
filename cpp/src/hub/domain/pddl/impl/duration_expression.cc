/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/duration_expression.hh"

namespace skdecide {

namespace pddl {

DurationExpression::DurationExpression() {}

DurationExpression::~DurationExpression() {}

std::ostream &DurationExpression::print(std::ostream &o) const {
  o << "?duration";
  return o;
}

} // namespace pddl

} // namespace skdecide
