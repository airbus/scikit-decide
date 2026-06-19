/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_TYPES_HH
#define SKDECIDE_PDDL_PARSE_TYPES_HH

#include "pegtl.hpp"
#include "parser_state.hh"
#include "parser_action.hh"
#include "parse_terms.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct primitive_type : name {};
template <> struct action<primitive_type> {
  template <typename Input> static void apply(const Input &in, state &s) {
    try {
      s.name = in.string();
      s.type_list.push_back(s.domain->add_type(s.name));
    } catch (const std::exception &e) {
      if (s.name != "object") {
        throw pegtl::parse_error(e.what(), in.current_position());
      }
    }
  }
};

struct type_list : typed_type_list<primitive_type> {};

template <> struct action<type_list> {
  template <typename Input> static void apply(const Input &in, state &s) {
    if (!s.global_requirements->has_typing()) {
      throw pegtl::parse_error(
          "declaring types without enabling :typing requirement",
          in.current_position());
    }
  }
};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_TYPES_HH
