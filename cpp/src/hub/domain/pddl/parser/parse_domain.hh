/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_DOMAIN_HH
#define SKDECIDE_PDDL_PARSE_DOMAIN_HH

#include "pegtl.hpp"
#include "parser_state.hh"
#include "parser_action.hh"

#include "parse_name.hh"
#include "parse_requirements.hh"
#include "parse_types.hh"
#include "parse_objects.hh"
#include "parse_predicates.hh"
#include "parse_functions.hh"
#include "parse_constraints.hh"
#include "parse_classes.hh"
#include "parse_operator.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

// parse domain name

struct domain_name
    : pegtl::if_must<pegtl::seq<pegtl::one<'('>, ignored,
                                keyword<'d', 'o', 'm', 'a', 'i', 'n'>, ignored>,
                     pegtl::seq<name, ignored, pegtl::one<')'>>> {};

template <> struct action<domain_name> {
  template <typename Input> static void apply(const Input &in, state &s) {
    auto d = s.domains.emplace(std::make_pair(
        StringConverter::tolower(s.name), std::make_shared<Domain>(s.name)));

    if (!d.second) {
      throw pegtl::parse_error("domain '" + s.name + "' already declared",
                               in.current_position());
    }

    s.domain = d.first->second;
    s.global_requirements =
        std::make_shared<Requirements>(); // exists even if no declared
                                          // requirements
  }
};

// parse domain requirement definition

struct open_domain_requirements
    : pegtl::seq<pegtl::one<'('>, ignored,
                 keyword<':', 'r', 'e', 'q', 'u', 'i', 'r', 'e', 'm', 'e', 'n',
                         't', 's'>,
                 ignored> {};

template <> struct action<open_domain_requirements> {
  static void apply0(state &s) {
    s.requirements = std::make_shared<Requirements>();
  }
};

struct domain_require_def
    : pegtl::if_must<open_domain_requirements,
                     pegtl::seq<pegtl::list<require_key, ignored>, ignored,
                                pegtl::one<')'>>> {};

template <> struct action<domain_require_def> {
  static void apply0(state &s) { s.domain->set_requirements(s.requirements); }
};

// parse domain types

struct type_names
    : pegtl::if_must<pegtl::seq<pegtl::one<'('>, ignored,
                                keyword<':', 't', 'y', 'p', 'e', 's'>, ignored>,
                     pegtl::seq<type_list, ignored, pegtl::one<')'>>> {};

// parse domain constants

struct domain_constants
    : pegtl::if_must<
          pegtl::seq<pegtl::one<'('>, ignored,
                     keyword<':', 'c', 'o', 'n', 's', 't', 'a', 'n', 't', 's'>,
                     ignored>,
          pegtl::seq<constant_list, ignored, pegtl::one<')'>>> {};

// parse predicates

struct predicates
    : pegtl::if_must<pegtl::seq<pegtl::one<'('>, ignored,
                                keyword<':', 'p', 'r', 'e', 'd', 'i', 'c', 'a',
                                        't', 'e', 's'>,
                                ignored>,
                     pegtl::seq<predicate_list, ignored, pegtl::one<')'>>> {};

// parse functions definitions

struct functions_def
    : pegtl::if_must<
          pegtl::seq<pegtl::one<'('>, ignored,
                     keyword<':', 'f', 'u', 'n', 'c', 't', 'i', 'o', 'n', 's'>,
                     ignored>,
          pegtl::seq<pegtl::action<function_list_action,
                                   function_list<domain_function_name>>,
                     ignored, pegtl::one<')'>>> {};

// parse constraints definitions

struct domain_constraint_goal : constraint_goal {};

template <> struct action<domain_constraint_goal> {
  static void apply0(state &s) { s.domain->set_constraints(s.formula); }
};

struct constraints_def
    : pegtl::if_must<
          pegtl::seq<pegtl::one<'('>, ignored,
                     keyword<':', 'c', 'o', 'n', 's', 't', 'r', 'a', 'i', 'n',
                             't', 's'>,
                     ignored>,
          pegtl::seq<domain_constraint_goal, ignored, pegtl::one<')'>>> {};

// parse domain classes

struct classes
    : pegtl::if_must<
          pegtl::seq<pegtl::one<'('>, ignored,
                     keyword<':', 'c', 'l', 'a', 's', 's', 'e', 's'>, ignored>,
          pegtl::seq<class_list, ignored, pegtl::one<')'>>> {};

// parse structures definitions

struct structure_def
    : pegtl::sor<operator_def<ActionOperator>, operator_def<EventOperator>,
                 operator_def<ProcessOperator>,
                 operator_def<DurativeActionOperator>, derivation_rule,
                 class_def> {};

struct structure_defs : pegtl::list<structure_def, ignored> {};

// parse domain preamble

struct preamble_item
    : pegtl::sor<domain_require_def, type_names, domain_constants, predicates,
                 functions_def, constraints_def, classes, structure_defs> {};

struct preamble : pegtl::list<preamble_item, ignored> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_DOMAIN_HH
