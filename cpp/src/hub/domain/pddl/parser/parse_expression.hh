/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_EXPRESSION_HH
#define SKDECIDE_PDDL_PARSE_EXPRESSION_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "parse_operation.hh"
#include "parse_number.hh"
#include "parse_function_head.hh"
#include "timed_expression.hh"
#include "duration_expression.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct expression : pegtl::sor<operation_expression<AddOperator, expression>,
                               operation_expression<SubOperator, expression>,
                               operation_expression<MulOperator, expression>,
                               operation_expression<DivOperator, expression>,
                               number_expression, function_head> {};

struct hash_t : keyword<'#', 't'> {};

template <> struct action<hash_t> {
  static void apply0(state &s) {
    s.expression = std::make_shared<TimeExpression>();
  }
};

struct open_timed_mul
    : pegtl::seq<pegtl::one<'('>, ignored, pegtl::one<'*'>, ignored> {};

template <> struct action<open_timed_mul> {
  static void apply0(state &s) {
    s.expressions.push(std::make_shared<MulExpression>());
  }
};

struct timed_mul_ht : hash_t {};

template <> struct action<timed_mul_ht> {
  static void apply0(state &s) {
    s.expression = std::make_shared<TimeExpression>();
    std::static_pointer_cast<MulExpression>(s.expressions.top())
        ->set_left_expression(s.expression);
  }
};

struct timed_mul_exp : expression {};

template <> struct action<timed_mul_exp> {
  static void apply0(state &s) {
    std::static_pointer_cast<MulExpression>(s.expressions.top())
        ->set_right_expression(s.expression);
  }
};

struct close_timed_mul
    : pegtl::seq<pegtl::sor<pegtl::seq<timed_mul_ht, ignored, timed_mul_exp>,
                            pegtl::seq<timed_mul_exp, ignored, timed_mul_ht>>,
                 ignored, pegtl::one<')'>> {};

template <> struct action<close_timed_mul> {
  static void apply0(state &s) {
    s.expression = s.expressions.top();
    s.expressions.pop();
  }
};

struct timed_mul : pegtl::if_must<open_timed_mul, close_timed_mul> {};

struct timed_expression : pegtl::sor<timed_mul, hash_t> {};

struct duration_expression
    : keyword<'?', 'd', 'u', 'r', 'a', 't', 'i', 'o', 'n'> {};

template <> struct action<duration_expression> {
  static void apply0(state &s) {
    s.expression = std::make_shared<DurationExpression>();
  }
};

struct durative_expression
    : pegtl::sor<operation_expression<AddOperator, durative_expression>,
                 operation_expression<SubOperator, durative_expression>,
                 operation_expression<MulOperator, durative_expression>,
                 operation_expression<DivOperator, durative_expression>,
                 duration_expression, number_expression, function_head> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_EXPRESSION_HH
