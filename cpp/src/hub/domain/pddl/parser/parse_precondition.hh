/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_PRECONDITON_HH
#define SKDECIDE_PDDL_PARSE_PRECONDITON_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "parse_formula.hh"
#include "parse_preference.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

template <typename FormulaRule>
struct precondition_preference
    : pegtl::sor<FormulaRule, preference<FormulaRule>> {};

struct empty_precondition
    : pegtl::seq<pegtl::one<'('>, ignored,
                 pegtl::opt<pegtl::seq<keyword<'a', 'n', 'd'>, ignored>>,
                 pegtl::one<')'>> {};

template <> struct action<empty_precondition> {
  static void apply0(state &s) {
    s.formula = std::make_shared<ConjunctionFormula>();
  }
};

struct precondition
    : pegtl::sor<empty_precondition,
                 aggregation<Conjunction, precondition_preference<formula>>,
                 quantification<Universal, precondition_preference<formula>>,
                 precondition_preference<formula>> {};

struct timed_precondition
    : pegtl::sor<empty_precondition,
                 aggregation<Conjunction, timed_precondition>,
                 quantification<Universal, timed_precondition>,
                 precondition_preference<timed_formula>> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_PRECONDITON_HH
