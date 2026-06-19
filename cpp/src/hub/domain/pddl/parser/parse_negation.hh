/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_NEG_FORMULA_HH
#define SKDECIDE_PDDL_PARSE_NEG_FORMULA_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "negation_formula.hh"
#include "equality_formula.hh"
#include "negation_effect.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct formula;
struct effect;

template <typename NegatedRule, typename Enable = void> struct NegationProxy;

template <typename NegatedRule>
struct NegationProxy<
    NegatedRule,
    typename std::enable_if<std::is_same<NegatedRule, formula>::value>::type> {
  typedef NegationFormula PDDLType;
  static Formula::Ptr &last_parsed(state &s) { return s.formula; }
  static Formula::Ptr &parsed(state &s) { return s.formula; }

  template <typename Input>
  static void check_requirement(const Input &in, state &s) {
    if (!std::dynamic_pointer_cast<EqualityFormula>(last_parsed(s)) &&
        !s.global_requirements->has_negative_preconditions()) {
      throw pegtl::parse_error(
          "using negation formula without enabling :negative-preconditions",
          in.current_position());
    }
  }
};

template <typename NegatedRule>
struct NegationProxy<NegatedRule,
                     typename std::enable_if<std::is_same<
                         NegatedRule, predicate_head<effect>>::value>::type> {
  typedef NegationEffect PDDLType;
  static PredicateEffect::Ptr last_parsed(state &s) {
    return std::static_pointer_cast<PredicateEffect>(s.effect);
  }
  static Effect::Ptr &parsed(state &s) { return s.effect; }

  template <typename Input>
  static void check_requirement(const Input &in, state &s) {}
};

struct open_negation
    : pegtl::seq<pegtl::one<'('>, ignored, keyword<'n', 'o', 't'>, ignored> {};

template <typename NegatedRule>
struct close_negation
    : pegtl::action<action, pegtl::seq<NegatedRule, ignored, pegtl::one<')'>>> {
  typedef NegationProxy<NegatedRule> NP;
};

template <typename Rule> struct close_negation_action {
  template <typename Input> static void apply(const Input &in, state &s) {
    typedef typename Rule::NP NP;
    NP::check_requirement(in, s);
    NP::parsed(s) = std::make_shared<typename NP::PDDLType>(NP::last_parsed(s));
  }
};

template <typename NegatedRule>
struct negation
    : pegtl::if_must<
          open_negation,
          pegtl::action<close_negation_action, close_negation<NegatedRule>>> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_NEG_FORMULA_HH
