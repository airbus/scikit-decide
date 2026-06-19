/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_IMPLICATION_HH
#define SKDECIDE_PDDL_PARSE_IMPLICATION_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "imply_formula.hh"
#include "conditional_effect.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct formula;
struct timed_formula;
struct cond_effect;
struct cond_effect_da;

template <typename ImpliedRule, typename Enable = void> struct ImplicationProxy;

template <typename ImpliedRule>
struct ImplicationProxy<
    ImpliedRule,
    typename std::enable_if<std::is_same<ImpliedRule, formula>::value>::type> {
  typedef keyword<'i', 'm', 'p', 'l', 'y'> kw;
  typedef ImplyFormula PDDLType;
  static std::stack<Formula::Ptr> &parsing_stack(state &s) {
    return s.formulas;
  }
  static Formula::Ptr &lhs_parsed(state &s) { return s.formula; }
  static Formula::Ptr &rhs_parsed(state &s) { return s.formula; }
  static Formula::Ptr &last_parsed(state &s) { return s.formula; }
  static void set_lhs(const ImplyFormula::Ptr &impf, const Formula::Ptr &f) {
    impf->set_left_formula(f);
  }
  static void set_rhs(const ImplyFormula::Ptr &impf, const Formula::Ptr &f) {
    impf->set_right_formula(f);
  }

  template <typename Input>
  static void check_requirement(const Input &in, state &s) {
    if (!s.global_requirements->has_disjunctive_preconditions()) {
      throw pegtl::parse_error("using 'imply' without enabling "
                               ":disjunctive-preconditions requirement",
                               in.current_position());
    }
  }
};

template <typename ImpliedRule>
struct ImplicationProxy<
    ImpliedRule, typename std::enable_if<
                     std::is_same<ImpliedRule, cond_effect>::value ||
                     std::is_same<ImpliedRule, cond_effect_da>::value>::type> {
  typedef keyword<'w', 'h', 'e', 'n'> kw;
  typedef ConditionalEffect PDDLType;
  static std::stack<Effect::Ptr> &parsing_stack(state &s) { return s.effects; }
  static Formula::Ptr &lhs_parsed(state &s) { return s.formula; }
  static Effect::Ptr &rhs_parsed(state &s) { return s.effect; }
  static Effect::Ptr &last_parsed(state &s) { return s.effect; }
  static void set_lhs(const ConditionalEffect::Ptr &conde,
                      const Formula::Ptr &f) {
    conde->set_condition(f);
  }
  static void set_rhs(const ConditionalEffect::Ptr &conde,
                      const Effect::Ptr &e) {
    conde->set_effect(e);
  }

  template <typename Input>
  static void check_requirement(const Input &in, state &s) {
    if (!s.global_requirements->has_conditional_effects()) {
      throw pegtl::parse_error(
          "using 'when' without enabling :conditional-effects requirement",
          in.current_position());
    }
  }
};

template <typename ImpliedRule>
struct open_implication
    : pegtl::action<
          action,
          pegtl::seq<pegtl::one<'('>, ignored,
                     typename ImplicationProxy<ImpliedRule>::kw, ignored>> {
  typedef ImplicationProxy<ImpliedRule> IP;
};

template <typename Rule> struct open_implication_action {
  template <typename Input> static void apply(const Input &in, state &s) {
    typedef typename Rule::IP IP;
    IP::check_requirement(in, s);
    IP::parsing_stack(s).push(std::make_shared<typename IP::PDDLType>());
  }
};

template <typename ImpliedRule>
struct implication_lhs
    : pegtl::action<action,
                    typename std::conditional<
                        std::is_same<ImpliedRule, cond_effect_da>::value,
                        timed_formula, formula>::type> {
  typedef ImplicationProxy<ImpliedRule> IP;
};

template <typename Rule> struct implication_lhs_action {
  static void apply0(state &s) {
    typedef typename Rule::IP IP;
    IP::set_lhs(std::static_pointer_cast<typename IP::PDDLType>(
                    IP::parsing_stack(s).top()),
                IP::lhs_parsed(s));
  }
};

template <typename ImpliedRule>
struct implication_rhs : pegtl::action<action, ImpliedRule> {
  typedef ImplicationProxy<ImpliedRule> IP;
};

template <typename Rule> struct implication_rhs_action {
  static void apply0(state &s) {
    typedef typename Rule::IP IP;
    IP::set_rhs(std::static_pointer_cast<typename IP::PDDLType>(
                    IP::parsing_stack(s).top()),
                IP::rhs_parsed(s));
  }
};

template <typename ImpliedRule>
struct close_implication
    : pegtl::action<action,
                    pegtl::seq<pegtl::action<implication_lhs_action,
                                             implication_lhs<ImpliedRule>>,
                               ignored,
                               pegtl::action<implication_rhs_action,
                                             implication_rhs<ImpliedRule>>,
                               ignored, pegtl::one<')'>>> {
  typedef ImplicationProxy<ImpliedRule> IP;
};

template <typename Rule> struct close_implication_action {
  static void apply0(state &s) {
    typedef typename Rule::IP IP;
    IP::last_parsed(s) = IP::parsing_stack(s).top();
    IP::parsing_stack(s).pop();
  }
};

template <typename ImpliedRule>
struct implication
    : pegtl::if_must<
          pegtl::action<open_implication_action, open_implication<ImpliedRule>>,
          pegtl::action<close_implication_action,
                        close_implication<ImpliedRule>>> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_IMPLICATION_HH
