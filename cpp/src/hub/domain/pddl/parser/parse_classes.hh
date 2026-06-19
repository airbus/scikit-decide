/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_CLASSES_HH
#define SKDECIDE_PDDL_PARSE_CLASSES_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"
#include "parse_terms.hh"

#include "function.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct class_decl_name : name {};
template <> struct action<class_decl_name> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    try {
      s.domain->add_class(s.name);
    } catch (const std::exception &e) {
      throw pegtl::parse_error(e.what(), in.current_position());
    }
  }
};

struct class_list : pegtl::star<class_decl_name, ignored> {};

struct class_def_name : name {};
template <> struct action<class_def_name> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    try {
      s.cls = s.domain->get_class(s.name);
    } catch (const std::exception &e) {
      throw pegtl::parse_error(e.what(), in.current_position());
    }
  }
};

struct open_class_def
    : pegtl::seq<pegtl::one<'('>, ignored,
                 keyword<':', 'c', 'l', 'a', 's', 's'>, ignored> {};

template <> struct action<open_class_def> {
  template <typename Input> static void apply(const Input &in, state &s) {
    if (!s.global_requirements->has_modules()) {
      throw pegtl::parse_error(
          "defining class without enabling :modules requirement",
          in.current_position());
    }
  }
};

struct close_class_def
    : pegtl::seq<class_def_name, ignored,
                 pegtl::action<function_list_action,
                               function_list<class_function_name>>,
                 ignored, pegtl::one<')'>> {};

struct class_def : pegtl::if_must<open_class_def, close_class_def> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_CLASSES_HH
