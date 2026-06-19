/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_PROBABILISTIC_EFFECT_HH
#define SKDECIDE_PDDL_PARSE_PROBABILISTIC_EFFECT_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"
#include "parse_number.hh"

#include "probabilistic_effect.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct p_effect;
struct effect;

struct open_probabilistic_effect
    : pegtl::seq<pegtl::one<'('>, ignored,
                 keyword<'p', 'r', 'o', 'b', 'a', 'b', 'i', 'l', 'i', 's', 't',
                         'i', 'c'>,
                 ignored> {};

template <> struct action<open_probabilistic_effect> {
  template <typename Input> static void apply(const Input &in, state &s) {
    if (!s.global_requirements->has_probabilistic_effects()) {
      throw pegtl::parse_error("using probabilistic effect without enabling "
                               ":probabilistic-effects requirement",
                               in.current_position());
    }
    s.effects.push(std::make_shared<ProbabilisticEffect>());
  }
};

struct prob_outcome : pegtl::seq<probability, ignored, effect> {};

template <> struct action<prob_outcome> {
  static void apply0(state &s) {
    std::static_pointer_cast<ProbabilisticEffect>(s.effects.top())
        ->append_outcome(s.number->as_double(), s.effect);
  }
};

struct close_probabilistic_effect
    : pegtl::seq<pegtl::list<prob_outcome, ignored>, ignored, pegtl::one<')'>> {
};

template <> struct action<close_probabilistic_effect> {
  static void apply0(state &s) {
    s.effect = s.effects.top();
    s.effects.pop();
  }
};

struct probabilistic_effect
    : pegtl::if_must<open_probabilistic_effect, close_probabilistic_effect> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_PROBABILISTIC_EFFECT_HH
