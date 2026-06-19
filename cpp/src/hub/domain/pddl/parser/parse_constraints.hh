/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_CONSTRAINTS_HH
#define SKDECIDE_PDDL_PARSE_CONSTRAINTS_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "parse_aggregation.hh"
#include "parse_quantification.hh"
#include "parse_assignment.hh"
#include "parse_number.hh"
#include "parse_preference.hh"

#include "timed_formula.hh"
#include "constraint_formula.hh"
#include "timed_effect.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct formula;
struct effect;
struct durative_expression;
struct duration_comparison;

// Operators

struct AtStartOperator {
  typedef pegtl::seq<keyword<'a', 't'>, ignored,
                     keyword<'s', 't', 'a', 'r', 't'>>
      kw;
  typedef AtStartFormula type;
};

struct AtEndOperator {
  typedef pegtl::seq<keyword<'a', 't'>, ignored, keyword<'e', 'n', 'd'>> kw;
  typedef AtEndFormula type;
};

struct OverAllOperator {
  typedef pegtl::seq<keyword<'o', 'v', 'e', 'r'>, ignored,
                     keyword<'a', 'l', 'l'>>
      kw;
  typedef OverAllFormula type;
};

struct AlwaysOperator {
  typedef keyword<'a', 'l', 'w', 'a', 'y', 's'> kw;
  typedef AlwaysFormula type;
};

struct SometimeOperator {
  typedef keyword<'s', 'o', 'm', 'e', 't', 'i', 'm', 'e'> kw;
  typedef SometimeFormula type;
};

struct WithinOperator {
  typedef keyword<'w', 'i', 't', 'h', 'i', 'n'> kw;
  typedef WithinFormula type;
};

struct AtMostOnceOperator {
  typedef keyword<'a', 't', '-', 'm', 'o', 's', 't', '-', 'o', 'n', 'c', 'e'>
      kw;
  typedef AtMostOnceFormula type;
};

struct SometimeAfterOperator {
  typedef keyword<'s', 'o', 'm', 'e', 't', 'i', 'm', 'e', '-', 'a', 'f', 't',
                  'e', 'r'>
      kw;
  typedef SometimeAfterFormula type;
};

struct SometimeBeforeOperator {
  typedef keyword<'s', 'o', 'm', 'e', 't', 'i', 'm', 'e', '-', 'b', 'e', 'f',
                  'o', 'r', 'e'>
      kw;
  typedef SometimeBeforeFormula type;
};

struct AlwaysWithinOperator {
  typedef keyword<'a', 'l', 'w', 'a', 'y', 's', '-', 'w', 'i', 't', 'h', 'i',
                  'n'>
      kw;
  typedef AlwaysWithinFormula type;
};

struct HoldDuringOperator {
  typedef keyword<'h', 'o', 'l', 'd', '-', 'd', 'u', 'r', 'i', 'n', 'g'> kw;
  typedef HoldDuringFormula type;
};

struct HoldAfterOperator {
  typedef keyword<'h', 'o', 'l', 'd', '-', 'a', 'f', 't', 'e', 'r'> kw;
  typedef HoldAfterFormula type;
};

struct AtStartEffectOperator {
  typedef pegtl::seq<keyword<'a', 't'>, ignored,
                     keyword<'s', 't', 'a', 'r', 't'>>
      kw;
  typedef AtStartEffect type;
};

struct AtEndEffectOperator {
  typedef pegtl::seq<keyword<'a', 't'>, ignored, keyword<'e', 'n', 'd'>> kw;
  typedef AtEndEffect type;
};

struct AtStartDurationOperator {
  typedef pegtl::seq<keyword<'a', 't'>, ignored,
                     keyword<'s', 't', 'a', 'r', 't'>>
      kw;
  typedef AtStartFormula type;
};

struct AtEndDurationOperator {
  typedef pegtl::seq<keyword<'a', 't'>, ignored, keyword<'e', 'n', 'd'>> kw;
  typedef AtEndFormula type;
};

// Unary constraint

template <typename Operator>
struct open_unary_constraint
    : pegtl::action<action, pegtl::seq<pegtl::one<'('>, ignored,
                                       typename Operator::kw, ignored>> {};

template <typename Operator>
struct close_unary_constraint
    : pegtl::action<
          action,
          pegtl::seq<
              typename std::conditional<
                  std::is_same<Operator, AtStartEffectOperator>::value ||
                      std::is_same<Operator, AtEndEffectOperator>::value,
                  // try_catch_return_false used below to avoid if_must rule in
                  // assignment<expression> failing in rule effect whereas we
                  // want assignment<durative_expression>
                  pegtl::sor<pegtl::try_catch_return_false<effect>,
                             assignment<durative_expression>>,
                  typename std::conditional<
                      std::is_same<Operator, AtStartDurationOperator>::value ||
                          std::is_same<Operator, AtEndDurationOperator>::value,
                      duration_comparison, formula>::type>::type,
              ignored, pegtl::one<')'>>> {
  typedef Operator OP;
};

template <typename Rule> struct close_unary_constraint_action {
  template <typename Operator, typename Enable = void> struct Impl;

  template <typename Operator>
  struct Impl<Operator,
              typename std::enable_if<
                  std::is_same<Operator, AtStartOperator>::value ||
                  std::is_same<Operator, OverAllOperator>::value ||
                  std::is_same<Operator, AtEndOperator>::value ||
                  std::is_same<Operator, AlwaysOperator>::value ||
                  std::is_same<Operator, SometimeOperator>::value ||
                  std::is_same<Operator, AtMostOnceOperator>::value ||
                  std::is_same<Operator, AtStartDurationOperator>::value ||
                  std::is_same<Operator, AtEndDurationOperator>::value>::type> {
    static void apply0(state &s) {
      s.formula = std::make_shared<typename Operator::type>(s.formula);
    }
  };

  template <typename Operator>
  struct Impl<Operator,
              typename std::enable_if<
                  std::is_same<Operator, AtStartEffectOperator>::value ||
                  std::is_same<Operator, AtEndEffectOperator>::value>::type> {
    static void apply0(state &s) {
      s.effect = std::make_shared<typename Operator::type>(s.effect);
    }
  };

  static void apply0(state &s) { Impl<typename Rule::OP>::apply0(s); }
};

template <typename Operator>
struct unary_constraint
    : pegtl::if_must<open_unary_constraint<Operator>,
                     pegtl::action<close_unary_constraint_action,
                                   close_unary_constraint<Operator>>> {};

// Binary constraint

template <typename Operator>
struct open_binary_constraint
    : pegtl::action<action, pegtl::seq<pegtl::one<'('>, ignored,
                                       typename Operator::kw, ignored>> {
  typedef Operator OP;
};

template <typename Rule> struct open_binary_constraint_action {
  static void apply0(state &s) {
    s.formulas.push(std::make_shared<typename Rule::OP::type>());
  }
};

template <typename Operator, typename Enable = void> struct constraint_lhs;

template <typename Operator>
struct constraint_lhs<
    Operator, typename std::enable_if<
                  std::is_same<Operator, SometimeAfterOperator>::value ||
                  std::is_same<Operator, SometimeBeforeOperator>::value>::type>
    : pegtl::action<action, formula> {
  typedef Operator OP;
};

template <typename Operator>
struct constraint_lhs<
    Operator, typename std::enable_if<
                  std::is_same<Operator, WithinOperator>::value ||
                  std::is_same<Operator, HoldAfterOperator>::value>::type>
    : pegtl::action<action, number> {
  typedef Operator OP;
};

template <typename Rule, typename Enable = void>
struct constraint_lhs_action {};

template <typename Rule>
struct constraint_lhs_action<
    Rule,
    typename std::enable_if<
        std::is_same<typename Rule::OP, SometimeAfterOperator>::value ||
        std::is_same<typename Rule::OP, SometimeBeforeOperator>::value>::type> {
  static void apply0(state &s) {
    typedef typename Rule::OP OP;
    std::static_pointer_cast<typename OP::type>(s.formulas.top())
        ->set_left_formula(s.formula);
  }
};

template <typename Rule>
struct constraint_lhs_action<
    Rule, typename std::enable_if<
              std::is_same<typename Rule::OP, WithinOperator>::value>::type> {
  static void apply0(state &s) {
    typedef typename Rule::OP OP;
    std::static_pointer_cast<typename OP::type>(s.formulas.top())
        ->set_deadline(s.number);
  }
};

template <typename Rule>
struct constraint_lhs_action<
    Rule, typename std::enable_if<std::is_same<
              typename Rule::OP, HoldAfterOperator>::value>::type> {
  static void apply0(state &s) {
    typedef typename Rule::OP OP;
    std::static_pointer_cast<typename OP::type>(s.formulas.top())
        ->set_from(s.number);
  }
};

template <typename Operator>
struct constraint_rhs : pegtl::action<action, formula> {
  typedef Operator OP;
};

template <typename Rule, typename Enable = void>
struct constraint_rhs_action {};

template <typename Rule>
struct constraint_rhs_action<
    Rule,
    typename std::enable_if<
        std::is_same<typename Rule::OP, SometimeAfterOperator>::value ||
        std::is_same<typename Rule::OP, SometimeBeforeOperator>::value>::type> {
  static void apply0(state &s) {
    typedef typename Rule::OP OP;
    std::static_pointer_cast<typename OP::type>(s.formulas.top())
        ->set_right_formula(s.formula);
  }
};

template <typename Rule>
struct constraint_rhs_action<
    Rule,
    typename std::enable_if<
        std::is_same<typename Rule::OP, WithinOperator>::value ||
        std::is_same<typename Rule::OP, HoldAfterOperator>::value>::type> {
  static void apply0(state &s) {
    typedef typename Rule::OP OP;
    std::static_pointer_cast<typename OP::type>(s.formulas.top())
        ->set_formula(s.formula);
  }
};

template <typename Operator>
struct close_binary_constraint
    : pegtl::action<
          action,
          pegtl::seq<
              pegtl::action<constraint_lhs_action, constraint_lhs<Operator>>,
              ignored,
              pegtl::action<constraint_rhs_action, constraint_rhs<Operator>>,
              ignored, pegtl::one<')'>>> {
  typedef Operator OP;
};

template <typename Rule> struct close_binary_constraint_action {
  static void apply0(state &s) {
    s.formula = s.formulas.top();
    s.formulas.pop();
  }
};

template <typename Operator>
struct binary_constraint
    : pegtl::if_must<pegtl::action<open_binary_constraint_action,
                                   open_binary_constraint<Operator>>,
                     pegtl::action<close_binary_constraint_action,
                                   close_binary_constraint<Operator>>> {};

// Ternary constraint

template <typename Operator>
struct open_ternary_constraint
    : pegtl::action<action, pegtl::seq<pegtl::one<'('>, ignored,
                                       typename Operator::kw, ignored>> {
  typedef Operator OP;
};

template <typename Rule> struct open_ternary_constraint_action {
  static void apply0(state &s) {
    s.formulas.push(std::make_shared<typename Rule::OP::type>());
  }
};

template <typename Operator>
struct constraint_one : pegtl::action<action, number> {
  typedef Operator OP;
};

template <typename Rule, typename Enable = void>
struct constraint_one_action {};

template <typename Rule>
struct constraint_one_action<
    Rule, typename std::enable_if<std::is_same<
              typename Rule::OP, AlwaysWithinOperator>::value>::type> {
  static void apply0(state &s) {
    typedef typename Rule::OP OP;
    std::static_pointer_cast<typename OP::type>(s.formulas.top())
        ->set_deadline(s.number);
  }
};

template <typename Rule>
struct constraint_one_action<
    Rule, typename std::enable_if<std::is_same<
              typename Rule::OP, HoldDuringOperator>::value>::type> {
  static void apply0(state &s) {
    typedef typename Rule::OP OP;
    std::static_pointer_cast<typename OP::type>(s.formulas.top())
        ->set_from(s.number);
  }
};

template <typename Operator, typename Enable = void> struct constraint_two;

template <typename Operator>
struct constraint_two<
    Operator, typename std::enable_if<
                  std::is_same<Operator, AlwaysWithinOperator>::value>::type>
    : pegtl::action<action, formula> {
  typedef Operator OP;
};

template <typename Operator>
struct constraint_two<Operator, typename std::enable_if<std::is_same<
                                    Operator, HoldDuringOperator>::value>::type>
    : pegtl::action<action, number> {
  typedef Operator OP;
};

template <typename Rule, typename Enable = void>
struct constraint_two_action {};

template <typename Rule>
struct constraint_two_action<
    Rule, typename std::enable_if<std::is_same<
              typename Rule::OP, AlwaysWithinOperator>::value>::type> {
  static void apply0(state &s) {
    typedef typename Rule::OP OP;
    std::static_pointer_cast<typename OP::type>(s.formulas.top())
        ->set_left_formula(s.formula);
  }
};

template <typename Rule>
struct constraint_two_action<
    Rule, typename std::enable_if<std::is_same<
              typename Rule::OP, HoldDuringOperator>::value>::type> {
  static void apply0(state &s) {
    typedef typename Rule::OP OP;
    std::static_pointer_cast<typename OP::type>(s.formulas.top())
        ->set_deadline(s.number);
  }
};

template <typename Operator>
struct constraint_three : pegtl::action<action, formula> {
  typedef Operator OP;
};

template <typename Rule, typename Enable = void>
struct constraint_three_action {};

template <typename Rule>
struct constraint_three_action<
    Rule, typename std::enable_if<std::is_same<
              typename Rule::OP, AlwaysWithinOperator>::value>::type> {
  static void apply0(state &s) {
    typedef typename Rule::OP OP;
    std::static_pointer_cast<typename OP::type>(s.formulas.top())
        ->set_right_formula(s.formula);
  }
};

template <typename Rule>
struct constraint_three_action<
    Rule, typename std::enable_if<std::is_same<
              typename Rule::OP, HoldDuringOperator>::value>::type> {
  static void apply0(state &s) {
    typedef typename Rule::OP OP;
    std::static_pointer_cast<typename OP::type>(s.formulas.top())
        ->set_formula(s.formula);
  }
};

template <typename Operator>
struct close_ternary_constraint
    : pegtl::action<
          action,
          pegtl::seq<
              pegtl::action<constraint_one_action, constraint_one<Operator>>,
              ignored,
              pegtl::action<constraint_two_action, constraint_two<Operator>>,
              ignored,
              pegtl::action<constraint_three_action,
                            constraint_three<Operator>>,
              ignored, pegtl::one<')'>>> {
  typedef Operator OP;
};

template <typename Rule> struct close_ternary_constraint_action {
  static void apply0(state &s) {
    s.formula = s.formulas.top();
    s.formulas.pop();
  }
};

template <typename Operator>
struct ternary_constraint
    : pegtl::if_must<pegtl::action<open_ternary_constraint_action,
                                   open_ternary_constraint<Operator>>,
                     pegtl::action<close_ternary_constraint_action,
                                   close_ternary_constraint<Operator>>> {};

// Constraint formula

struct constraint_goal
    : pegtl::sor<
          aggregation<Conjunction, constraint_goal>,
          quantification<Universal, constraint_goal>,
          unary_constraint<AtEndOperator>, unary_constraint<AlwaysOperator>,
          unary_constraint<SometimeOperator>, binary_constraint<WithinOperator>,
          unary_constraint<AtMostOnceOperator>,
          binary_constraint<SometimeAfterOperator>,
          binary_constraint<SometimeBeforeOperator>,
          ternary_constraint<AlwaysWithinOperator>,
          ternary_constraint<HoldDuringOperator>,
          binary_constraint<HoldAfterOperator>> {};

// Preference constraint formula

struct pref_con_goal;

struct pref_goal : pegtl::sor<preference<constraint_goal>,
                              aggregation<Conjunction, pref_con_goal>,
                              quantification<Universal, pref_con_goal>> {};

struct pref_con_goal : pegtl::sor<pref_goal, constraint_goal> {};

// Duration constraint

struct empty_duration_constraint
    : pegtl::seq<pegtl::one<'('>, ignored, keyword<'a', 'n', 'd'>, ignored,
                 pegtl::one<')'>> {};

template <> struct action<empty_duration_constraint> {
  static void apply0(state &s) {
    s.formula = std::make_shared<ConjunctionFormula>();
  }
};

struct duration_constraint
    : pegtl::sor<aggregation<Conjunction, duration_constraint>,
                 duration_comparison, unary_constraint<AtStartDurationOperator>,
                 unary_constraint<AtEndDurationOperator>,
                 empty_duration_constraint> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_CONSTRAINTS_HH
