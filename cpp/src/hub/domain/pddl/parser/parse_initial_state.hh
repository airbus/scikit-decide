/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_INITIAL_STATE_HH
#define SKDECIDE_PDDL_PARSE_INITIAL_STATE_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "parse_predicate_head.hh"
#include "parse_negation.hh"
#include "parse_operation.hh"
#include "parse_number.hh"

#include "timed_effect.hh"
#include "probabilistic_effect.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct init_els;

struct open_timed_initial_literal
    : pegtl::seq<pegtl::one<'('>, ignored, keyword<'a', 't'>, ignored, number,
                 ignored> {};

template <> struct action<open_timed_initial_literal> {
  template <typename Input> static void apply(const Input &in, state &s) {
    if (!s.global_requirements->has_timed_initial_literals()) {
      throw pegtl::parse_error("using timed initial literal without enabling "
                               ":timed-initial-literals requirement",
                               in.current_position());
    }
    s.effects.push(std::make_shared<ConjunctionEffect>());
  }
};

struct close_timed_initial_literal
    : pegtl::seq<init_els, ignored, pegtl::one<')'>> {};

template <> struct action<close_timed_initial_literal> {
  static void apply0(state &s) {
    s.effect = s.effects.top();
    s.effects.pop();
  }
};

struct timed_initial_literal
    : pegtl::if_must<open_timed_initial_literal, close_timed_initial_literal> {
};

template <> struct action<timed_initial_literal> {
  static void apply0(state &s) {
    s.effect = std::make_shared<AtTimeEffect>(s.number, s.effect);
  }
};

struct a_init_el
    : pegtl::sor<operation_expression<AssignInitOperator, number_expression>,
                 negation<predicate_head<effect>>,
                 timed_initial_literal, // parse before predicate head to
                                        // deconflict 'at' keyword
                 predicate_head<effect>> {};

struct open_probabilistic_init_el
    : pegtl::seq<pegtl::one<'('>, ignored,
                 keyword<'p', 'r', 'o', 'b', 'a', 'b', 'i', 'l', 'i', 's', 't',
                         'i', 'c'>,
                 ignored> {};

template <> struct action<open_probabilistic_init_el> {
  template <typename Input> static void apply(const Input &in, state &s) {
    if (!s.global_requirements->has_probabilistic_effects()) {
      throw pegtl::parse_error(
          "using probabilistic initial element without enabling "
          ":probabilistic-effects requirement",
          in.current_position());
    }
    s.effects.push(std::make_shared<ProbabilisticEffect>());
  }
};

struct prob_init_outcome : pegtl::seq<probability, ignored, a_init_el> {};

template <> struct action<prob_init_outcome> {
  static void apply0(state &s) {
    std::static_pointer_cast<ProbabilisticEffect>(s.effects.top())
        ->append_outcome(s.number->as_double(), s.effect);
  }
};

struct close_probabilistic_init_el
    : pegtl::seq<pegtl::list<prob_init_outcome, ignored>, ignored,
                 pegtl::one<')'>> {};

template <> struct action<close_probabilistic_init_el> {
  static void apply0(state &s) {
    s.effect = s.effects.top();
    s.effects.pop();
  }
};

struct probabilistic_init_el
    : pegtl::if_must<open_probabilistic_init_el, close_probabilistic_init_el> {
};

struct init_el
    : pegtl::sor<operation_expression<AssignInitOperator, number_expression>,
                 negation<predicate_head<effect>>,
                 timed_initial_literal, // parse before predicate head to
                                        // deconflict 'at' keyword
                 probabilistic_init_el, predicate_head<effect>> {};

template <> struct action<init_el> {
  static void apply0(state &s) {
    std::static_pointer_cast<ConjunctionEffect>(s.effects.top())
        ->append_effect(s.effect);
  }
};

struct init_els : pegtl::star<pegtl::seq<init_el, ignored>> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_INITIAL_STATE_HH
