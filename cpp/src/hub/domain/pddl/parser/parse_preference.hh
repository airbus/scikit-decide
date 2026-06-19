/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_PREFERENCE_HH
#define SKDECIDE_PDDL_PARSE_PREFERENCE_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "preference.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct open_preference
    : pegtl::seq<pegtl::one<'('>, ignored,
                 keyword<'p', 'r', 'e', 'f', 'e', 'r', 'e', 'n', 'c', 'e'>,
                 ignored> {};

template <> struct action<open_preference> {
  template <typename Input> static void apply(const Input &in, state &s) {
    if (!s.global_requirements->has_preferences()) {
      throw pegtl::parse_error(
          "using preference formula without enabling :preferences requirement",
          in.current_position());
    }
    s.formulas.push(std::make_shared<Preference>());
  }
};

struct preference_name : name {};

template <> struct action<preference_name> {
  static void apply0(state &s) {
    // remove the default anonymous one and replace with the named one
    s.formulas.pop();
    if (s.domain && !s.problem) {
      s.formulas.push(s.domain->add_preference(s.name));
    } else if (s.problem) {
      s.formulas.push(s.problem->add_preference(s.name));
    }
  }
};

template <typename PreferenceFormula>
struct preference_formula : pegtl::action<action, PreferenceFormula> {};

template <typename Rule> struct preference_formula_action {
  static void apply0(state &s) {
    std::static_pointer_cast<Preference>(s.formulas.top())
        ->set_formula(s.formula);
  }
};

template <typename PreferenceFormula>
struct close_preference
    : pegtl::action<
          action,
          pegtl::seq<pegtl::opt<pegtl::seq<preference_name, ignored>>,
                     pegtl::action<preference_formula_action,
                                   preference_formula<PreferenceFormula>>,
                     ignored, pegtl::one<')'>>> {};

template <typename Rule> struct close_preference_action {
  static void apply0(state &s) {
    s.formula = s.formulas.top();
    s.formulas.pop();
  }
};

template <typename PreferenceFormula>
struct preference
    : pegtl::if_must<open_preference,
                     pegtl::action<close_preference_action,
                                   close_preference<PreferenceFormula>>> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_PREFERENCE_HH
