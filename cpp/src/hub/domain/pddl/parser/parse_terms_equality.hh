/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_TERMS_EQUALITY_HH
#define SKDECIDE_PDDL_PARSE_TERMS_EQUALITY_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "parse_name.hh"
#include "parse_terms.hh"

#include "equality_formula.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct terms_equality_constant : name {};

template <> struct action<terms_equality_constant> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    std::unordered_map<std::string, Domain::ObjectPtr>::const_iterator i =
        s.registered_objects.find(StringConverter::tolower(s.name));
    if (i == s.registered_objects.end()) {
      throw pegtl::parse_error("object '" + s.name +
                                   "' unknown to the current parsing context",
                               in.current_position());
    } else {
      std::static_pointer_cast<EqualityFormula>(s.formula)->append_term(
          i->second);
    }
  }
};

struct terms_equality_variable : name {};

template <> struct action<terms_equality_variable> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    std::unordered_map<std::string, Predicate::VariablePtr>::const_iterator i =
        s.registered_variables.find('?' + StringConverter::tolower(s.name));
    if (i == s.registered_variables.end()) {
      throw pegtl::parse_error("variable '?" + s.name +
                                   "' unknown to the current parsing context",
                               in.current_position());
    } else {
      std::static_pointer_cast<EqualityFormula>(s.formula)->append_term(
          i->second);
    }
  }
};

struct try_open_terms_equality
    : pegtl::seq<pegtl::one<'('>, ignored, pegtl::one<'='>, ignored> {};

template <> struct action<try_open_terms_equality> {
  static void apply0(state &s) {
    // will be replaced by a (comparison) EqFormula
    // if it happens to be an expression equality comparison
    s.formula = std::make_shared<EqualityFormula>();
  }
};

struct open_terms_equality
    : pegtl::seq<
          try_open_terms_equality,
          // includes the first term in the if_must test
          // to disambiguate from expression equality operator
          pegtl::sor<terms_equality_constant,
                     pegtl::seq<pegtl::one<'?'>, terms_equality_variable>>,
          ignored> {};

template <> struct action<open_terms_equality> {
  template <typename Input> static void apply(const Input &in, state &s) {
    if (!s.global_requirements->has_equality()) {
      throw pegtl::parse_error("using term equality comparison without "
                               "enabling :equality requirement",
                               in.current_position());
    }
  }
};

struct close_terms_equality
    : pegtl::seq<term_list<terms_equality_constant, terms_equality_variable>,
                 ignored, pegtl::one<')'>> {};

struct terms_equality
    : pegtl::if_must<open_terms_equality, close_terms_equality> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_TERMS_EQUALITY_HH
