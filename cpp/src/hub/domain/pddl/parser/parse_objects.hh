/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_CONSTANTS_HH
#define SKDECIDE_PDDL_PARSE_CONSTANTS_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"
#include "parse_terms.hh"

#include "object.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct primitive_constant : name {};
template <> struct action<primitive_constant> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    s.object_list.push_back(s.domain->add_object(s.name));
    auto i = s.registered_objects.insert(
        std::make_pair(StringConverter::tolower(s.name), s.object_list.back()));
    if (!i.second) {
      throw pegtl::parse_error("constant '" + s.name +
                                   "' already existing in domain '" +
                                   s.domain->get_name() + "'",
                               in.current_position());
    }
  }
};

struct constant_list : typed_obj_list<primitive_constant> {};

struct primitive_object : name {};
template <> struct action<primitive_object> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    s.object_list.push_back(s.problem->add_object(s.name));
    auto i = s.registered_objects.insert(
        std::make_pair(StringConverter::tolower(s.name), s.object_list.back()));
    if (!i.second) {
      throw pegtl::parse_error("object '" + s.name +
                                   "' already existing in problem '" +
                                   s.problem->get_name() + "'",
                               in.current_position());
    }
  }
};

struct object_list : typed_obj_list<primitive_object> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_CONSTANTS_HH
