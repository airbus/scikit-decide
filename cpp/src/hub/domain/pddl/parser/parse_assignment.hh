/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_ASSIGNMENT_HH
#define SKDECIDE_PDDL_PARSE_ASSIGNMENT_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "parse_operation.hh"
#include "parse_expression.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

template <typename ExpressionRule>
struct assignment
    : pegtl::sor<operation_expression<AssignOperator, ExpressionRule>,
                 operation_expression<IncreaseOperator, ExpressionRule>,
                 operation_expression<DecreaseOperator, ExpressionRule>,
                 operation_expression<ScaleUpOperator, ExpressionRule>,
                 operation_expression<ScaleDownOperator, ExpressionRule>> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_ASSIGNMENT_HH
