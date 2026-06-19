/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_COMPARISON_HH
#define SKDECIDE_PDDL_PARSE_COMPARISON_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "parse_operation.hh"
#include "parse_expression.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct comparison
    : pegtl::sor<operation_expression<GreaterEqOperator, expression>,
                 operation_expression<GreaterOperator, expression>,
                 operation_expression<LessEqOperator, expression>,
                 operation_expression<LessOperator, expression>,
                 operation_expression<EqOperator, expression>> {};

struct duration_comparison
    : pegtl::sor<operation_expression<LessEqOperator, duration_expression>,
                 operation_expression<GreaterEqOperator, duration_expression>,
                 operation_expression<EqOperator, duration_expression>> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_COMPARISON_HH
