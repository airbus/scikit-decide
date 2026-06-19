/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_PREDICATES_HH
#define SKDECIDE_PDDL_PARSE_PREDICATES_HH

#include "pegtl.hpp"
#include "parser_state.hh"
#include "parser_action.hh"
#include "parse_terms.hh"
#include "parse_formula.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct predicate_name : name {};
template <> struct action<predicate_name> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    s.predicate = s.domain->add_predicate(s.name);
  }
};

struct predicate_variable : name {};
template <> struct action<predicate_variable> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    s.variable_list.push_back(s.predicate->append_variable(s.name));
  }
};

struct predicate_list : termed_symbols<predicate_name, predicate_variable> {};

struct derived_predicate_name : name {};
template <> struct action<derived_predicate_name> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    s.predicate = s.domain->add_derived_predicate(s.name);
  }
};

struct derived_proposition
    : term_symbol<derived_predicate_name, predicate_variable> {};

struct derived_formula : formula {};

template <> struct action<derived_formula> {
  static void apply0(state &s) {
    std::static_pointer_cast<DerivedPredicate>(s.predicate)
        ->set_formula(s.formula);
  }
};

struct open_derivation_rule
    : pegtl::seq<pegtl::one<'('>, ignored,
                 keyword<':', 'd', 'e', 'r', 'i', 'v', 'e', 'd'>, ignored> {};

template <> struct action<open_derivation_rule> {
  template <typename Input> static void apply(const Input &in, state &s) {
    if (!s.global_requirements->has_derived_predicates()) {
      throw pegtl::parse_error("defining derived predicate without enabling "
                               ":derived-predicates requirement",
                               in.current_position());
    }
  }
};

struct close_derivation_rule
    : pegtl::seq<derived_proposition, ignored, derived_formula, ignored,
                 pegtl::one<')'>> {};

struct derivation_rule
    : pegtl::if_must<open_derivation_rule, close_derivation_rule> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_PREDICATES_HH
